import type Anthropic from "@anthropic-ai/sdk";
import type {
  GenerateContentParameters,
  GenerateContentResponse,
  Content,
  Part,
  FunctionCallingConfigMode,
  Candidate,
} from "@google/genai";

type BlockType = "text" | "tool_use" | "thinking" | "web_search_tool_result" | null;

interface StreamState {
  id: string;
  model: string;
  started: boolean;
  currentBlockIndex: number;
  currentBlockType: BlockType;
  prevText: string;
  prevThought: string;
  seenFunctionCallIds: Set<string>;
  toolCallBlockIndices: Map<string, number>;
}

export class GeminiToMessagesConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (Gemini → Messages, forward) ---

  convertRequest(params: GenerateContentParameters): Anthropic.MessageCreateParams {
    const systemParts: string[] = [];
    const messages: Anthropic.MessageParam[] = [];

    const config = params.config;
    if (config?.systemInstruction) {
      const parts =
        typeof config.systemInstruction === "string"
          ? [config.systemInstruction]
          : ((config.systemInstruction as Content).parts?.map(p => p.text ?? "") ?? []);
      systemParts.push(...parts);
    }

    const contents = params.contents;
    if (typeof contents === "string") {
      messages.push({
        role: "user",
        content: [{ type: "text", text: contents }],
      });
    } else if (Array.isArray(contents)) {
      for (const content of contents as Content[]) {
        this.convertContent(messages, content);
      }
    }

    const result: Anthropic.MessageCreateParams = {
      model: params.model,
      max_tokens: config?.maxOutputTokens ?? 4096,
      messages,
    };

    if (systemParts.length > 0) {
      result.system = systemParts.join("\n");
    }
    if (config?.temperature != null) {
      result.temperature = config.temperature;
    }
    if (config?.topP != null) {
      result.top_p = config.topP;
    }
    if (config?.topK != null) {
      result.top_k = config.topK;
    }
    if (config?.stopSequences != null) {
      result.stop_sequences = config.stopSequences;
    }
    if (config?.tools) {
      const { tools, hasWebSearch } = this.convertTools(config.tools);
      if (tools.length > 0) {
        result.tools = tools;
      }
      if (hasWebSearch) {
        result.tools = [...(result.tools ?? []), { type: "web_search_20250305" } as any];
      }
    }
    if (config?.toolConfig?.functionCallingConfig) {
      result.tool_choice = this.convertToolConfig(config.toolConfig.functionCallingConfig);
    }
    if (config?.thinkingConfig) {
      result.thinking = this.convertThinkingConfig(config.thinkingConfig);
    }
    if (config?.responseMimeType === "application/json" && config.responseJsonSchema) {
      result.output_config = {
        format: {
          type: "json_schema",
          schema: config.responseJsonSchema as Record<string, unknown>,
        },
      };
    }

    return result;
  }

  // --- Response conversion (Messages → Gemini, backward) ---

  convertResponse(message: Anthropic.Message): GenerateContentResponse {
    const parts: Part[] = [];

    for (const block of message.content) {
      if (block.type === "thinking") {
        parts.push({
          text: block.thinking,
          thought: true,
          thoughtSignature: block.signature ?? undefined,
        });
      }
    }

    for (const block of message.content) {
      if (block.type === "tool_use" || block.type === "server_tool_use") {
        parts.push({
          functionCall: {
            id: block.id,
            name: block.name,
            args: block.input as Record<string, unknown>,
          },
        });
      }
    }

    for (const block of message.content) {
      if (block.type === "text") {
        parts.push({ text: block.text });
      }
    }

    const hasToolUse = message.content.some(
      b => b.type === "tool_use" || b.type === "server_tool_use"
    );
    const finishReason = this.mapStopReasonToFinishReason(message.stop_reason, hasToolUse);

    const usage = message.usage;
    const result: GenerateContentResponse = {
      candidates: [
        {
          content: { role: "model", parts },
          finishReason,
        } as Candidate,
      ],
      responseId: message.id,
      modelVersion: message.model as string,
    } as GenerateContentResponse;

    result.usageMetadata = {
      promptTokenCount: usage.input_tokens,
      candidatesTokenCount: usage.output_tokens,
      totalTokenCount: usage.input_tokens + usage.output_tokens,
      cachedContentTokenCount: usage.cache_read_input_tokens ?? 0,
    } as any;

    return result;
  }

  // --- Stream conversion (Messages → Gemini, backward) ---

  async *convertStream(
    stream: AsyncIterable<Anthropic.RawMessageStreamEvent>
  ): AsyncIterable<GenerateContentResponse> {
    for await (const event of stream) {
      const chunk = this.convertStreamEvent(event);
      if (chunk) {
        yield chunk;
      }
    }
  }

  convertStreamEvent(event: Anthropic.RawMessageStreamEvent): GenerateContentResponse | null {
    const state = this.streamState;

    switch (event.type) {
      case "message_start": {
        const msg = event.message;
        state.id = msg.id;
        state.model = msg.model;
        state.started = true;
        return this.makeStreamChunk();
      }

      case "content_block_start": {
        const block = event.content_block;

        if (block.type === "thinking") {
          state.currentBlockType = "thinking";
          state.currentBlockIndex++;
        } else if (block.type === "tool_use" || block.type === "server_tool_use") {
          state.currentBlockType = "tool_use";
          state.currentBlockIndex++;
          const fcId = block.id;
          state.seenFunctionCallIds.add(fcId);
          state.toolCallBlockIndices.set(fcId, state.currentBlockIndex);
          return this.makeStreamChunkWithNewFunctionCall(block);
        } else if (block.type === "text") {
          state.currentBlockType = "text";
          state.currentBlockIndex++;
        }
        return null;
      }

      case "content_block_delta": {
        const delta = event.delta;

        if (delta.type === "thinking_delta") {
          state.prevThought += delta.thinking;
          return this.makeStreamChunkAccumulated();
        } else if (delta.type === "input_json_delta") {
          return this.makeStreamChunkAccumulated();
        } else if (delta.type === "text_delta") {
          state.prevText += delta.text;
          return this.makeStreamChunkAccumulated();
        }
        return null;
      }

      case "content_block_stop":
        return null;

      case "message_delta": {
        return null;
      }

      case "message_stop": {
        const chunk = this.makeStreamChunkAccumulated();
        const candidate = chunk.candidates![0] as any;
        const hasToolUse = state.seenFunctionCallIds.size > 0;
        candidate.finishReason = hasToolUse ? "STOP" : "STOP";
        return chunk;
      }

      default:
        return null;
    }
  }

  // --- Private: request helpers ---

  private convertContent(messages: Anthropic.MessageParam[], content: Content): void {
    const role = content.role;
    const parts = content.parts ?? [];

    if (role === "user") {
      const functionResponses = parts.filter(p => p.functionResponse);
      const otherParts = parts.filter(p => !p.functionResponse);

      if (functionResponses.length > 0) {
        const toolResults: Anthropic.ToolResultBlockParam[] = functionResponses.map(fr => {
          const resp = fr.functionResponse!;
          const output =
            typeof resp.response === "string" ? resp.response : JSON.stringify(resp.response ?? {});
          return {
            type: "tool_result" as const,
            tool_use_id: resp.id ?? resp.name ?? "",
            content: [{ type: "text" as const, text: output }],
          };
        });
        messages.push({ role: "user", content: toolResults });
      }

      if (otherParts.length > 0) {
        const blocks = this.convertUserParts(otherParts);
        if (blocks.length > 0) {
          messages.push({ role: "user", content: blocks });
        }
      }
    } else if (role === "model") {
      const blocks: Anthropic.ContentBlockParam[] = [];

      for (const p of parts) {
        if (p.text != null && !p.thought && !p.functionCall) {
          blocks.push({ type: "text", text: p.text });
        } else if (p.functionCall) {
          const fc = p.functionCall;
          blocks.push({
            type: "tool_use",
            id: fc.id ?? fc.name ?? "",
            name: fc.name ?? "",
            input: fc.args ?? {},
          });
        }
      }

      if (blocks.length > 0) {
        messages.push({ role: "assistant", content: blocks });
      }
    }
  }

  private convertUserParts(parts: Part[]): Anthropic.ContentBlockParam[] {
    const blocks: Anthropic.ContentBlockParam[] = [];

    for (const p of parts) {
      if (p.text != null) {
        blocks.push({ type: "text", text: p.text });
      } else if (p.inlineData) {
        blocks.push({
          type: "image",
          source: {
            type: "base64",
            media_type: p.inlineData.mimeType as Anthropic.Base64ImageSource["media_type"],
            data: p.inlineData.data ?? "",
          },
        });
      } else if (p.fileData) {
        blocks.push({
          type: "image",
          source: {
            type: "url",
            url: p.fileData.fileUri ?? "",
          },
        });
      }
    }

    return blocks;
  }

  private convertTools(tools: any[]): {
    tools: Anthropic.ToolUnion[];
    hasWebSearch: boolean;
  } {
    const result: Anthropic.ToolUnion[] = [];
    let hasWebSearch = false;

    for (const t of tools) {
      if (t.functionDeclarations) {
        for (const fd of t.functionDeclarations) {
          result.push({
            name: fd.name ?? "",
            description: fd.description,
            input_schema: (fd.parametersJsonSchema ??
              fd.parameters ?? { type: "object" }) as Anthropic.Tool.InputSchema,
          });
        }
      }
      if (t.googleSearch != null) {
        hasWebSearch = true;
      }
    }

    return { tools: result, hasWebSearch };
  }

  private convertToolConfig(fcc: any): Anthropic.ToolChoice {
    const mode = fcc.mode as FunctionCallingConfigMode | string;
    switch (mode) {
      case "AUTO":
        return { type: "auto" };
      case "ANY":
        if (fcc.allowedFunctionNames?.length === 1) {
          return { type: "tool", name: fcc.allowedFunctionNames[0] };
        }
        return { type: "any" };
      case "NONE":
        return { type: "none" };
      default:
        return { type: "auto" };
    }
  }

  private convertThinkingConfig(tc: any): Anthropic.ThinkingConfigParam {
    if (tc.thinkingBudget === 0) {
      return { type: "disabled" };
    }
    return {
      type: "enabled",
      budget_tokens: tc.thinkingBudget ?? 10240,
    };
  }

  // --- Private: response helpers ---

  private mapStopReasonToFinishReason(
    stopReason: Anthropic.Message["stop_reason"],
    hasToolUse: boolean
  ): string {
    if (hasToolUse) return "STOP";
    switch (stopReason) {
      case "end_turn":
      case "stop_sequence":
      case "tool_use":
        return "STOP";
      case "max_tokens":
        return "MAX_TOKENS";
      case "refusal":
        return "SAFETY";
      default:
        return "STOP";
    }
  }

  // --- Private: stream helpers ---

  private createStreamState(): StreamState {
    return {
      id: "",
      model: "",
      started: false,
      currentBlockIndex: -1,
      currentBlockType: null,
      prevText: "",
      prevThought: "",
      seenFunctionCallIds: new Set(),
      toolCallBlockIndices: new Map(),
    };
  }

  private makeStreamChunk(): GenerateContentResponse {
    return {
      candidates: [
        {
          content: { role: "model", parts: [] },
        } as Candidate,
      ],
      responseId: this.streamState.id,
      modelVersion: this.streamState.model,
    } as GenerateContentResponse;
  }

  private makeStreamChunkWithNewFunctionCall(block: any): GenerateContentResponse {
    const state = this.streamState;
    const parts: Part[] = [];

    if (state.prevThought) {
      parts.push({ text: state.prevThought, thought: true });
    }
    if (state.prevText) {
      parts.push({ text: state.prevText });
    }

    parts.push({
      functionCall: {
        id: block.id,
        name: block.name,
        args: {},
      },
    });

    return {
      candidates: [
        {
          content: { role: "model", parts },
        } as Candidate,
      ],
      responseId: state.id,
      modelVersion: state.model,
    } as GenerateContentResponse;
  }

  private makeStreamChunkAccumulated(): GenerateContentResponse {
    const state = this.streamState;
    const parts: Part[] = [];

    if (state.prevThought) {
      parts.push({ text: state.prevThought, thought: true });
    }
    if (state.prevText) {
      parts.push({ text: state.prevText });
    }

    return {
      candidates: [
        {
          content: { role: "model", parts },
        } as Candidate,
      ],
      responseId: state.id,
      modelVersion: state.model,
    } as GenerateContentResponse;
  }
}
