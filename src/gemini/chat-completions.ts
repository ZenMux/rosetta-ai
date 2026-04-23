import type OpenAI from "openai";
import type {
  GenerateContentParameters,
  GenerateContentResponse,
  Content,
  Part,
  FunctionDeclaration,
  FunctionCallingConfigMode,
  Candidate,
} from "@google/genai";

interface StreamState {
  id: string;
  model: string;
  started: boolean;
  toolCallCounter: number;
  prevText: string;
  prevThought: string;
  seenFunctionCallIds: Set<string>;
}

export class GeminiToChatCompletionConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (Gemini → CC, forward) ---

  convertRequest(params: GenerateContentParameters): OpenAI.ChatCompletionCreateParams {
    const messages: OpenAI.ChatCompletionMessageParam[] = [];

    const config = params.config;
    if (config?.systemInstruction) {
      const sysParts =
        typeof config.systemInstruction === "string"
          ? [config.systemInstruction]
          : ((config.systemInstruction as Content).parts?.map(p => p.text ?? "") ?? []);
      messages.push({ role: "system", content: sysParts.join("\n") });
    }

    const contents = params.contents;
    if (typeof contents === "string") {
      messages.push({ role: "user", content: contents });
    } else if (Array.isArray(contents)) {
      for (const content of contents as Content[]) {
        this.convertContent(messages, content);
      }
    }

    const result: OpenAI.ChatCompletionCreateParams = {
      model: params.model,
      messages,
    };

    if (config?.maxOutputTokens != null) {
      result.max_completion_tokens = config.maxOutputTokens;
    }
    if (config?.temperature != null) {
      result.temperature = config.temperature;
    }
    if (config?.topP != null) {
      result.top_p = config.topP;
    }
    if (config?.stopSequences != null) {
      result.stop = config.stopSequences;
    }
    if (config?.seed != null) {
      result.seed = config.seed;
    }
    if (config?.frequencyPenalty != null) {
      result.frequency_penalty = config.frequencyPenalty;
    }
    if (config?.presencePenalty != null) {
      result.presence_penalty = config.presencePenalty;
    }
    if (config?.candidateCount != null) {
      (result as any).n = config.candidateCount;
    }
    if (config?.responseLogprobs != null) {
      result.logprobs = config.responseLogprobs;
    }
    if (config?.logprobs != null) {
      result.top_logprobs = config.logprobs;
    }
    if (config?.tools) {
      const { tools, hasWebSearch } = this.convertTools(config.tools);
      if (tools.length > 0) {
        result.tools = tools;
      }
      if (hasWebSearch) {
        (result as any).web_search_options = {};
      }
    }
    if (config?.toolConfig?.functionCallingConfig) {
      result.tool_choice = this.convertToolConfig(config.toolConfig.functionCallingConfig);
    }
    if (config?.responseMimeType === "application/json") {
      if (config.responseJsonSchema) {
        result.response_format = {
          type: "json_schema",
          json_schema: {
            name: "response",
            schema: config.responseJsonSchema as Record<string, unknown>,
          },
        };
      } else {
        result.response_format = { type: "json_object" };
      }
    }
    if (config?.thinkingConfig) {
      result.reasoning_effort = this.convertThinkingConfig(config.thinkingConfig) as any;
    }

    return result;
  }

  // --- Response conversion (CC → Gemini, backward) ---

  convertResponse(response: OpenAI.ChatCompletion): GenerateContentResponse {
    const choice = response.choices?.[0];
    const msg = choice?.message;
    const parts: Part[] = [];

    const reasoning = (msg as any)?.reasoning || (msg as any)?.reasoning_content;
    if (reasoning) {
      parts.push({ text: reasoning, thought: true });
    }

    if (msg?.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (tc.type === "function") {
          let args: Record<string, unknown>;
          try {
            args = JSON.parse(tc.function.arguments || "{}");
          } catch {
            args = {};
          }
          parts.push({
            functionCall: {
              id: tc.id,
              name: tc.function.name,
              args,
            },
          });
        }
      }
    }

    if (msg?.content != null) {
      parts.push({ text: msg.content });
    }

    const finishReason = this.mapFinishReasonToGemini(choice?.finish_reason);

    const result: GenerateContentResponse = {
      candidates: [
        {
          content: { role: "model", parts },
          finishReason,
        } as Candidate,
      ],
      responseId: response.id,
      modelVersion: response.model,
    } as GenerateContentResponse;

    if (response.usage) {
      const usage = response.usage;
      result.usageMetadata = {
        promptTokenCount: usage.prompt_tokens,
        candidatesTokenCount: usage.completion_tokens,
        totalTokenCount: usage.total_tokens,
        cachedContentTokenCount: usage.prompt_tokens_details?.cached_tokens ?? 0,
        thoughtsTokenCount: usage.completion_tokens_details?.reasoning_tokens ?? 0,
      } as any;
    }

    return result;
  }

  // --- Stream conversion (CC → Gemini, backward) ---

  async *convertStream(
    stream: AsyncIterable<OpenAI.ChatCompletionChunk>
  ): AsyncIterable<GenerateContentResponse> {
    for await (const chunk of stream) {
      const result = this.convertStreamChunk(chunk);
      if (result) {
        yield result;
      }
    }
  }

  convertStreamChunk(chunk: OpenAI.ChatCompletionChunk): GenerateContentResponse | null {
    const state = this.streamState;
    const choice = chunk.choices?.[0];

    if (chunk.model) state.model = chunk.model;
    if (chunk.id) state.id = chunk.id;

    if (!state.started) {
      state.started = true;
      return this.makeStreamChunk();
    }

    if (!choice) return null;
    const delta = choice.delta;

    const reasoning = (delta as any)?.reasoning || (delta as any)?.reasoning_content;
    if (reasoning) {
      state.prevThought += reasoning;
      return this.makeStreamChunkWithParts();
    }

    if (delta?.tool_calls) {
      for (const tc of delta.tool_calls) {
        if (tc.id && tc.function?.name) {
          const fcId = tc.id;
          if (!state.seenFunctionCallIds.has(fcId)) {
            state.seenFunctionCallIds.add(fcId);
            state.toolCallCounter++;
          }
        }
      }
      return this.makeStreamChunkWithParts(delta);
    }

    if (delta?.content) {
      state.prevText += delta.content;
      return this.makeStreamChunkWithParts();
    }

    if (choice.finish_reason) {
      const finishReason = this.mapFinishReasonToGemini(choice.finish_reason);
      const result = this.makeStreamChunkWithParts();
      const candidate = result.candidates![0] as any;
      candidate.finishReason = finishReason;

      if (chunk.usage) {
        result.usageMetadata = {
          promptTokenCount: chunk.usage.prompt_tokens,
          candidatesTokenCount: chunk.usage.completion_tokens,
          totalTokenCount: chunk.usage.total_tokens,
          cachedContentTokenCount: chunk.usage.prompt_tokens_details?.cached_tokens ?? 0,
          thoughtsTokenCount: chunk.usage.completion_tokens_details?.reasoning_tokens ?? 0,
        } as any;
      }

      return result;
    }

    return null;
  }

  // --- Private: request helpers ---

  private convertContent(messages: OpenAI.ChatCompletionMessageParam[], content: Content): void {
    const role = content.role;
    const parts = content.parts ?? [];

    if (role === "user") {
      const functionResponses = parts.filter(p => p.functionResponse);
      const otherParts = parts.filter(p => !p.functionResponse);

      if (otherParts.length > 0) {
        const ccParts = this.convertUserParts(otherParts);
        if (ccParts.length === 1 && typeof ccParts[0] !== "object") {
          messages.push({ role: "user", content: ccParts[0] as string });
        } else {
          messages.push({
            role: "user",
            content: ccParts as OpenAI.ChatCompletionContentPart[],
          });
        }
      }

      for (const fr of functionResponses) {
        const resp = fr.functionResponse!;
        const output =
          typeof resp.response === "string" ? resp.response : JSON.stringify(resp.response ?? {});
        messages.push({
          role: "tool",
          tool_call_id: resp.id ?? resp.name ?? "",
          content: output,
        });
      }
    } else if (role === "model") {
      const textParts = parts.filter(p => p.text != null && !p.thought && !p.functionCall);
      const functionCalls = parts.filter(p => p.functionCall);

      const msg: OpenAI.ChatCompletionAssistantMessageParam = {
        role: "assistant",
      };

      if (textParts.length > 0) {
        msg.content = textParts.map(p => p.text!).join("");
      }

      if (functionCalls.length > 0) {
        msg.tool_calls = functionCalls.map(fc => {
          const call = fc.functionCall!;
          return {
            id: call.id ?? call.name ?? "",
            type: "function" as const,
            function: {
              name: call.name ?? "",
              arguments: JSON.stringify(call.args ?? {}),
            },
          };
        });
      }

      messages.push(msg);
    }
  }

  private convertUserParts(parts: Part[]): (string | OpenAI.ChatCompletionContentPart)[] {
    if (parts.length === 1 && parts[0].text != null) {
      return [parts[0].text];
    }

    return parts.map(p => {
      if (p.text != null) {
        return { type: "text" as const, text: p.text };
      }
      if (p.inlineData) {
        return {
          type: "image_url" as const,
          image_url: {
            url: `data:${p.inlineData.mimeType};base64,${p.inlineData.data}`,
          },
        };
      }
      if (p.fileData) {
        return {
          type: "image_url" as const,
          image_url: { url: p.fileData.fileUri ?? "" },
        };
      }
      return { type: "text" as const, text: "" };
    });
  }

  private convertTools(tools: any[]): {
    tools: OpenAI.ChatCompletionTool[];
    hasWebSearch: boolean;
  } {
    const ccTools: OpenAI.ChatCompletionTool[] = [];
    let hasWebSearch = false;

    for (const t of tools) {
      if (t.functionDeclarations) {
        for (const fd of t.functionDeclarations as FunctionDeclaration[]) {
          ccTools.push({
            type: "function",
            function: {
              name: fd.name ?? "",
              description: fd.description,
              parameters: (fd.parametersJsonSchema ?? fd.parameters) as Record<string, unknown>,
            },
          });
        }
      }
      if (t.googleSearch != null) {
        hasWebSearch = true;
      }
    }

    return { tools: ccTools, hasWebSearch };
  }

  private convertToolConfig(fcc: any): OpenAI.ChatCompletionToolChoiceOption {
    const mode = fcc.mode as FunctionCallingConfigMode | string;
    switch (mode) {
      case "AUTO":
        return "auto";
      case "ANY":
        if (fcc.allowedFunctionNames?.length === 1) {
          return {
            type: "function",
            function: { name: fcc.allowedFunctionNames[0] },
          };
        }
        return "required";
      case "NONE":
        return "none";
      default:
        return "auto";
    }
  }

  private convertThinkingConfig(tc: any): string | null {
    if (tc.thinkingBudget === 0) return "none";
    const budget = tc.thinkingBudget ?? 10240;
    if (budget <= 2048) return "low";
    if (budget <= 5120) return "medium";
    return "high";
  }

  // --- Private: response helpers ---

  private mapFinishReasonToGemini(reason: string | null | undefined): string {
    switch (reason) {
      case "stop":
      case "tool_calls":
        return "STOP";
      case "length":
        return "MAX_TOKENS";
      case "content_filter":
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
      toolCallCounter: 0,
      prevText: "",
      prevThought: "",
      seenFunctionCallIds: new Set(),
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

  private makeStreamChunkWithParts(
    delta?: OpenAI.ChatCompletionChunk.Choice.Delta
  ): GenerateContentResponse {
    const state = this.streamState;
    const parts: Part[] = [];

    if (state.prevThought) {
      parts.push({ text: state.prevThought, thought: true });
    }

    if (delta?.tool_calls) {
      for (const tc of delta.tool_calls) {
        if (tc.id && tc.function?.name) {
          let args: Record<string, unknown>;
          try {
            args = JSON.parse(tc.function.arguments || "{}");
          } catch {
            args = {};
          }
          parts.push({
            functionCall: {
              id: tc.id,
              name: tc.function.name,
              args,
            },
          });
        }
      }
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
