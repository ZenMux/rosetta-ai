import type Anthropic from "@anthropic-ai/sdk";
import type {
  GenerateContentParameters,
  GenerateContentConfig,
  GenerateContentResponse,
  Content,
  Part,
  FunctionDeclaration,
  FunctionCallingConfigMode,
  FinishReason,
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

export class MessagesToGeminiConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (Messages → Gemini, forward) ---

  convertRequest(params: Anthropic.MessageCreateParams): GenerateContentParameters {
    const systemParts: Part[] = [];
    const contents: Content[] = [];

    if (params.system) {
      if (typeof params.system === "string") {
        systemParts.push({ text: params.system });
      } else {
        for (const block of params.system) {
          systemParts.push({ text: block.text });
        }
      }
    }

    for (const msg of params.messages) {
      if (msg.role === "user") {
        this.convertUserMessage(contents, msg);
      } else if (msg.role === "assistant") {
        this.convertAssistantMessage(contents, msg);
      }
    }

    const config: GenerateContentConfig = {};

    if (systemParts.length > 0) {
      config.systemInstruction = { parts: systemParts };
    }
    if (params.max_tokens != null) {
      config.maxOutputTokens = params.max_tokens;
    }
    if (params.temperature != null) {
      config.temperature = params.temperature;
    }
    if (params.top_p != null) {
      config.topP = params.top_p;
    }
    if (params.top_k != null) {
      config.topK = params.top_k;
    }
    if (params.stop_sequences != null) {
      config.stopSequences = params.stop_sequences;
    }
    if (params.tools) {
      const { tools, hasGoogleSearch } = this.convertTools(params.tools);
      if (tools.length > 0) {
        config.tools = [{ functionDeclarations: tools }];
      }
      if (hasGoogleSearch) {
        config.tools = [...(config.tools ?? []), { googleSearch: {} }];
      }
    }
    if (params.tool_choice != null) {
      config.toolConfig = {
        functionCallingConfig: this.convertToolChoice(params.tool_choice),
      };
    }
    if (params.thinking) {
      config.thinkingConfig = this.convertThinking(params.thinking);
    }
    if (params.output_config?.format?.type === "json_schema") {
      config.responseMimeType = "application/json";
      config.responseJsonSchema = params.output_config.format.schema;
    }
    if (params.stream === true) {
      // stream flag is handled at the call site, not in config
    }

    return {
      model: params.model as string,
      contents,
      config,
    };
  }

  // --- Response conversion (Gemini → Messages, backward) ---

  convertResponse(response: GenerateContentResponse): Anthropic.Message {
    const candidate = response.candidates?.[0];
    const parts = candidate?.content?.parts ?? [];

    const content: Anthropic.ContentBlock[] = [];

    for (const part of parts) {
      if (part.thought && part.text) {
        content.push({
          type: "thinking",
          thinking: part.text,
          signature: part.thoughtSignature ?? "",
        });
      }
    }

    const groundingAnnotations = this.extractGroundingAnnotations(candidate);
    if (groundingAnnotations.length > 0) {
      content.push({
        type: "web_search_tool_result",
        content: groundingAnnotations,
        tool_use_id: `websearch_${this.generateId()}`,
        caller: { type: "direct" },
      });
    }

    for (const part of parts) {
      if (part.text != null && !part.thought) {
        content.push({ type: "text", text: part.text, citations: null });
      }
    }

    for (const part of parts) {
      if (part.functionCall) {
        const fc = part.functionCall;
        content.push({
          type: "tool_use",
          id: fc.id ?? fc.name ?? this.generateId(),
          name: fc.name ?? "",
          input: fc.args ?? {},
          caller: { type: "direct" },
        });
      }
    }

    if (content.length === 0) {
      content.push({ type: "text", text: "", citations: null });
    }

    const hasToolUse = content.some(b => b.type === "tool_use");
    const stopReason = this.mapFinishReasonToStopReason(candidate?.finishReason, hasToolUse);

    const usage = response.usageMetadata;
    const promptTokens = (usage?.promptTokenCount ?? 0) + (usage?.toolUsePromptTokenCount ?? 0);
    const thoughtsTokens = usage?.thoughtsTokenCount ?? 0;
    const candidatesTokens = (usage?.candidatesTokenCount ?? 0) + thoughtsTokens;

    return {
      id: response.responseId ?? `msg_${this.generateId()}`,
      type: "message",
      role: "assistant",
      model: (response.modelVersion ?? "") as Anthropic.Model,
      content,
      stop_reason: stopReason,
      stop_sequence: null,
      usage: {
        input_tokens: promptTokens,
        output_tokens: candidatesTokens,
        cache_read_input_tokens: usage?.cachedContentTokenCount ?? null,
        cache_creation_input_tokens: 0,
        cache_creation: null,
        inference_geo: null,
        server_tool_use: null,
        service_tier: null,
      },
      container: null,
    };
  }

  // --- Stream conversion (Gemini → Messages, backward) ---

  async *convertStream(
    stream: AsyncIterable<GenerateContentResponse>
  ): AsyncIterable<Anthropic.RawMessageStreamEvent> {
    for await (const chunk of stream) {
      const events = this.convertStreamChunk(chunk);
      for (const event of events) {
        yield event;
      }
    }
  }

  convertStreamChunk(chunk: GenerateContentResponse): Anthropic.RawMessageStreamEvent[] {
    const state = this.streamState;
    const events: Anthropic.RawMessageStreamEvent[] = [];
    const candidate = chunk.candidates?.[0];
    const parts = candidate?.content?.parts ?? [];

    if (chunk.modelVersion) {
      state.model = chunk.modelVersion;
    }
    if (chunk.responseId) {
      state.id = chunk.responseId;
    }

    if (!state.started) {
      state.started = true;
      if (!state.id) {
        state.id = `msg_${this.generateId()}`;
      }
      events.push({
        type: "message_start",
        message: {
          id: state.id,
          type: "message",
          role: "assistant",
          model: state.model as Anthropic.Model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            cache_creation: null,
            inference_geo: null,
            server_tool_use: null,
            service_tier: null,
          },
          container: null,
        },
      });
    }

    for (const part of parts) {
      if (part.thought && part.text) {
        const newThought = part.text.slice(state.prevThought.length);
        if (newThought) {
          if (state.currentBlockType !== "thinking") {
            this.transitionBlock(state, events, "thinking");
            events.push({
              type: "content_block_start",
              index: state.currentBlockIndex,
              content_block: { type: "thinking", thinking: "", signature: "" },
            });
          }
          state.prevThought = part.text;
          events.push({
            type: "content_block_delta",
            index: state.currentBlockIndex,
            delta: { type: "thinking_delta", thinking: newThought },
          });
        }
      } else if (part.functionCall) {
        const fc = part.functionCall;
        const fcId = fc.id ?? fc.name ?? "";
        if (!state.seenFunctionCallIds.has(fcId)) {
          state.seenFunctionCallIds.add(fcId);
          this.transitionBlock(state, events, "tool_use");
          state.toolCallBlockIndices.set(fcId, state.currentBlockIndex);

          events.push({
            type: "content_block_start",
            index: state.currentBlockIndex,
            content_block: {
              type: "tool_use",
              id: fc.id ?? fc.name ?? this.generateId(),
              name: fc.name ?? "",
              input: {},
              caller: { type: "direct" },
            },
          });

          const args = JSON.stringify(fc.args ?? {});
          if (args !== "{}") {
            events.push({
              type: "content_block_delta",
              index: state.currentBlockIndex,
              delta: { type: "input_json_delta", partial_json: args },
            });
          }
        }
      } else if (part.text != null) {
        const newText = part.text.slice(state.prevText.length);
        if (newText) {
          if (state.currentBlockType !== "text") {
            this.transitionBlock(state, events, "text");
            events.push({
              type: "content_block_start",
              index: state.currentBlockIndex,
              content_block: { type: "text", text: "", citations: null },
            });
          }
          state.prevText = part.text;
          events.push({
            type: "content_block_delta",
            index: state.currentBlockIndex,
            delta: { type: "text_delta", text: newText },
          });
        }
      }
    }

    const groundingAnnotations = this.extractGroundingAnnotations(candidate);
    if (groundingAnnotations.length > 0) {
      this.transitionBlock(state, events, "web_search_tool_result");
      events.push({
        type: "content_block_start",
        index: state.currentBlockIndex,
        content_block: {
          type: "web_search_tool_result",
          content: groundingAnnotations,
          tool_use_id: `websearch_${this.generateId()}`,
          caller: { type: "direct" },
        },
      });
    }

    if (candidate?.finishReason) {
      if (state.currentBlockType !== null) {
        events.push({
          type: "content_block_stop",
          index: state.currentBlockIndex,
        });
      }

      const hasToolUse = state.seenFunctionCallIds.size > 0;
      const stopReason = this.mapFinishReasonToStopReason(candidate.finishReason, hasToolUse);

      events.push({
        type: "message_delta",
        delta: {
          stop_reason: stopReason,
          stop_sequence: null,
          container: null,
        },
        usage: {
          output_tokens: chunk.usageMetadata?.candidatesTokenCount ?? 0,
          input_tokens: null,
          cache_creation_input_tokens: null,
          cache_read_input_tokens: null,
          server_tool_use: null,
        },
      });

      events.push({ type: "message_stop" });
    }

    return events;
  }

  // --- Private: request helpers ---

  private convertUserMessage(contents: Content[], msg: Anthropic.MessageParam): void {
    if (typeof msg.content === "string") {
      contents.push({ role: "user", parts: [{ text: msg.content }] });
      return;
    }

    const userParts: Part[] = [];
    const toolResponseParts: Part[] = [];

    for (const block of msg.content) {
      switch (block.type) {
        case "text":
          userParts.push({ text: block.text });
          break;
        case "image": {
          const imgBlock = block as Anthropic.ImageBlockParam;
          userParts.push(this.convertImageBlock(imgBlock));
          break;
        }
        case "document": {
          const docBlock = block as Anthropic.DocumentBlockParam;
          userParts.push(this.convertDocumentBlock(docBlock));
          break;
        }
        case "tool_result": {
          const tr = block as Anthropic.ToolResultBlockParam;
          const output =
            typeof tr.content === "string"
              ? tr.content
              : tr.content
                ? tr.content
                    .filter((b): b is Anthropic.TextBlockParam => b.type === "text")
                    .map(b => b.text)
                    .join("\n")
                : "";
          toolResponseParts.push({
            functionResponse: {
              id: tr.tool_use_id,
              name: tr.tool_use_id,
              response: { output },
            },
          });
          break;
        }
      }
    }

    if (toolResponseParts.length > 0) {
      contents.push({ role: "user", parts: toolResponseParts });
      if (userParts.length > 0) {
        contents.push({ role: "user", parts: userParts });
      }
      return;
    }

    if (userParts.length > 0) {
      contents.push({ role: "user", parts: userParts });
    }
  }

  private convertImageBlock(block: Anthropic.ImageBlockParam): Part {
    const source = block.source;

    if (source.type === "base64") {
      return {
        inlineData: {
          mimeType: source.media_type,
          data: source.data,
        },
      };
    }

    return {
      fileData: {
        fileUri: (source as Anthropic.URLImageSource).url,
        mimeType: "image/*",
      },
    };
  }

  private convertDocumentBlock(block: Anthropic.DocumentBlockParam): Part {
    const source = block.source;

    if (source.type === "base64") {
      const src = source as Anthropic.Base64PDFSource;
      return {
        inlineData: {
          mimeType: src.media_type,
          data: src.data,
        },
      };
    }

    if (source.type === "url") {
      const src = source as Anthropic.URLPDFSource;
      return {
        fileData: {
          fileUri: src.url,
          mimeType: "application/pdf",
        },
      };
    }

    if (source.type === "text") {
      const src = source as Anthropic.PlainTextSource;
      return { text: src.data };
    }

    return { text: "[Unsupported document source]" };
  }

  private convertAssistantMessage(contents: Content[], msg: Anthropic.MessageParam): void {
    if (typeof msg.content === "string") {
      contents.push({ role: "model", parts: [{ text: msg.content }] });
      return;
    }

    const parts: Part[] = [];

    for (const block of msg.content) {
      if (block.type === "text") {
        parts.push({ text: block.text });
      } else if (block.type === "tool_use") {
        const tu = block as Anthropic.ToolUseBlockParam;
        parts.push({
          functionCall: {
            id: tu.id,
            name: tu.name,
            args: tu.input as Record<string, unknown>,
          },
        });
      }
    }

    if (parts.length > 0) {
      contents.push({ role: "model", parts });
    }
  }

  private convertTools(tools: Anthropic.ToolUnion[]): {
    tools: FunctionDeclaration[];
    hasGoogleSearch: boolean;
  } {
    const functionTools: FunctionDeclaration[] = [];
    let hasGoogleSearch = false;

    for (const t of tools) {
      if ("input_schema" in t) {
        const tool = t as Anthropic.Tool;
        functionTools.push({
          name: tool.name,
          description: tool.description,
          parametersJsonSchema: tool.input_schema as unknown as Record<string, unknown>,
        });
      } else if (
        "type" in t &&
        (t.type === "web_search_20250305" || t.type === "web_search_20260209")
      ) {
        hasGoogleSearch = true;
      }
    }

    return { tools: functionTools, hasGoogleSearch };
  }

  private convertToolChoice(choice: Anthropic.ToolChoice): {
    mode?: FunctionCallingConfigMode;
    allowedFunctionNames?: string[];
  } {
    switch (choice.type) {
      case "auto":
        return { mode: "AUTO" as FunctionCallingConfigMode };
      case "any":
        return { mode: "ANY" as FunctionCallingConfigMode };
      case "none":
        return { mode: "NONE" as FunctionCallingConfigMode };
      case "tool": {
        const name = (choice as Anthropic.ToolChoiceTool).name;
        return {
          mode: "ANY" as FunctionCallingConfigMode,
          allowedFunctionNames: [name],
        };
      }
      default:
        return { mode: "AUTO" as FunctionCallingConfigMode };
    }
  }

  private convertThinking(thinking: Anthropic.ThinkingConfigParam): {
    includeThoughts?: boolean;
    thinkingBudget?: number;
  } {
    if (thinking.type === "disabled") {
      return { thinkingBudget: 0 };
    }
    if (thinking.type === "enabled") {
      const budget = (thinking as Anthropic.ThinkingConfigEnabled).budget_tokens;
      return { includeThoughts: true, thinkingBudget: budget };
    }
    return { includeThoughts: true, thinkingBudget: -1 };
  }

  // --- Private: response helpers ---

  private mapFinishReasonToStopReason(
    reason: FinishReason | string | null | undefined,
    hasToolUse: boolean
  ): Anthropic.Message["stop_reason"] {
    if (hasToolUse) return "tool_use";
    switch (reason) {
      case "STOP":
        return "end_turn";
      case "MAX_TOKENS":
        return "max_tokens";
      case "SAFETY":
      case "RECITATION":
      case "BLOCKLIST":
      case "PROHIBITED_CONTENT":
        return "refusal";
      default:
        return "end_turn";
    }
  }

  private extractGroundingAnnotations(candidate: any): Array<{
    type: "web_search_result";
    title: string;
    url: string;
    encrypted_content: string;
    page_age: null;
  }> {
    if (!candidate?.groundingMetadata?.groundingChunks) return [];
    const results: Array<{
      type: "web_search_result";
      title: string;
      url: string;
      encrypted_content: string;
      page_age: null;
    }> = [];
    for (const chunk of candidate.groundingMetadata.groundingChunks) {
      const web = chunk.web;
      if (web) {
        results.push({
          type: "web_search_result",
          title: web.title || "",
          url: web.uri || "",
          encrypted_content: "",
          page_age: null,
        });
      }
    }
    return results;
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
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

  private transitionBlock(
    state: StreamState,
    events: Anthropic.RawMessageStreamEvent[],
    newType: BlockType
  ): void {
    if (state.currentBlockType !== null) {
      events.push({
        type: "content_block_stop",
        index: state.currentBlockIndex,
      });
    }
    state.currentBlockIndex++;
    state.currentBlockType = newType;
  }
}
