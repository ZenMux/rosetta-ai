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

type RespResponse = OpenAI.Responses.Response;
type RespStreamEvent = OpenAI.Responses.ResponseStreamEvent;

interface StreamState {
  id: string;
  model: string;
  parts: Part[];
  finishReason: string | null;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cachedTokens: number;
  reasoningTokens: number;
}

export class GeminiToResponsesConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (Gemini → Responses, forward) ---

  convertRequest(params: GenerateContentParameters): OpenAI.Responses.ResponseCreateParams {
    const input: any[] = [];
    let instructions: string | undefined;

    const config = params.config;
    if (config?.systemInstruction) {
      const sysParts =
        typeof config.systemInstruction === "string"
          ? [config.systemInstruction]
          : ((config.systemInstruction as Content).parts?.map(p => p.text ?? "") ?? []);
      instructions = sysParts.join("\n");
    }

    const contents = params.contents;
    if (typeof contents === "string") {
      input.push({
        role: "user",
        type: "message",
        content: [{ type: "input_text", text: contents }],
      });
    } else if (Array.isArray(contents)) {
      for (const content of contents as Content[]) {
        this.convertContent(input, content);
      }
    }

    const result: any = {
      model: params.model,
      input,
    };

    if (instructions) {
      result.instructions = instructions;
    }
    if (config?.maxOutputTokens != null) {
      result.max_output_tokens = config.maxOutputTokens;
    }
    if (config?.temperature != null) {
      result.temperature = config.temperature;
    }
    if (config?.topP != null) {
      result.top_p = config.topP;
    }
    if (config?.tools) {
      result.tools = this.convertTools(config.tools);
    }
    if (config?.toolConfig?.functionCallingConfig) {
      result.tool_choice = this.convertToolConfig(config.toolConfig.functionCallingConfig);
    }
    if (config?.thinkingConfig) {
      result.reasoning = this.convertThinkingConfig(config.thinkingConfig);
    }
    if (config?.responseMimeType === "application/json") {
      if (config.responseJsonSchema) {
        result.text = {
          format: {
            type: "json_schema",
            name: "response",
            schema: config.responseJsonSchema,
          },
        };
      } else {
        result.text = { format: { type: "json_object" } };
      }
    }

    return result as OpenAI.Responses.ResponseCreateParams;
  }

  // --- Response conversion (Responses → Gemini, backward) ---

  convertResponse(response: RespResponse): GenerateContentResponse {
    const parts: Part[] = [];
    let hasToolCall = false;

    for (const item of response.output) {
      if (item.type === "reasoning") {
        const text = (item as any).summary?.map((s: any) => s.text).join("");
        if (text) {
          parts.push({ text, thought: true });
        }
      }
    }

    for (const item of response.output) {
      if (item.type === "function_call") {
        hasToolCall = true;
        const fc = item as any;
        let args: Record<string, unknown>;
        try {
          args = JSON.parse(fc.arguments || "{}");
        } catch {
          args = {};
        }
        parts.push({
          functionCall: {
            id: fc.call_id,
            name: fc.name,
            args,
          },
        });
      }
    }

    for (const item of response.output) {
      if (item.type === "message") {
        const msg = item as any;
        if (Array.isArray(msg.content)) {
          for (const c of msg.content) {
            if (c.type === "output_text" && c.text) {
              parts.push({ text: c.text });
            }
          }
        }
      }
    }

    const finishReason = this.statusToFinishReason(response.status, hasToolCall);

    const usage = response.usage;
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

    if (usage) {
      result.usageMetadata = {
        promptTokenCount: usage.input_tokens,
        candidatesTokenCount: usage.output_tokens,
        totalTokenCount: usage.total_tokens,
        cachedContentTokenCount: usage.input_tokens_details?.cached_tokens ?? 0,
        thoughtsTokenCount: usage.output_tokens_details?.reasoning_tokens ?? 0,
      } as any;
    }

    return result;
  }

  // --- Stream conversion (Responses → Gemini, backward) ---

  async *convertStream(
    stream: AsyncIterable<RespStreamEvent>
  ): AsyncIterable<GenerateContentResponse> {
    for await (const event of stream) {
      const chunk = this.convertStreamEvent(event);
      if (chunk) {
        yield chunk;
      }
    }
  }

  convertStreamEvent(event: RespStreamEvent): GenerateContentResponse | null {
    const state = this.streamState;

    switch (event.type) {
      case "response.created": {
        const resp = (event as any).response;
        state.id = resp?.id ?? "";
        state.model = resp?.model ?? "";
        return this.makeStreamChunk();
      }

      case "response.reasoning_summary_text.delta": {
        const delta = (event as any).delta as string;
        const lastPart = state.parts[state.parts.length - 1];
        if (lastPart?.thought && lastPart.text != null) {
          lastPart.text += delta;
        } else {
          state.parts.push({ text: delta, thought: true });
        }
        return this.makeStreamChunk();
      }

      case "response.output_text.delta": {
        const delta = (event as any).delta as string;
        const lastPart = state.parts[state.parts.length - 1];
        if (lastPart?.text != null && !lastPart.thought && !lastPart.functionCall) {
          lastPart.text += delta;
        } else {
          state.parts.push({ text: delta });
        }
        return this.makeStreamChunk();
      }

      case "response.output_item.added": {
        const item = (event as any).item;
        if (item?.type === "function_call") {
          state.parts.push({
            functionCall: {
              id: item.call_id,
              name: item.name,
              args: {},
            },
          });
          return this.makeStreamChunk();
        }
        return null;
      }

      case "response.function_call_arguments.delta": {
        const delta = (event as any).delta as string;
        const fcPart = [...state.parts].reverse().find(p => p.functionCall);
        if (fcPart?.functionCall) {
          const current = JSON.stringify(fcPart.functionCall.args ?? {});
          const base = current === "{}" ? "" : current;
          try {
            fcPart.functionCall.args = JSON.parse(base + delta);
          } catch {
            // partial JSON — accumulate as string and parse later
          }
        }
        return this.makeStreamChunk();
      }

      case "response.completed":
      case "response.incomplete": {
        const resp = (event as any).response;
        state.finishReason = event.type === "response.incomplete" ? "MAX_TOKENS" : "STOP";

        if (resp?.usage) {
          state.inputTokens = resp.usage.input_tokens ?? 0;
          state.outputTokens = resp.usage.output_tokens ?? 0;
          state.totalTokens = resp.usage.total_tokens ?? 0;
          state.cachedTokens = resp.usage.input_tokens_details?.cached_tokens ?? 0;
          state.reasoningTokens = resp.usage.output_tokens_details?.reasoning_tokens ?? 0;
        }

        const chunk = this.makeStreamChunk();
        const candidate = chunk.candidates![0] as any;
        candidate.finishReason = state.finishReason;

        if (state.inputTokens > 0 || state.outputTokens > 0) {
          chunk.usageMetadata = {
            promptTokenCount: state.inputTokens,
            candidatesTokenCount: state.outputTokens,
            totalTokenCount: state.totalTokens,
            cachedContentTokenCount: state.cachedTokens,
            thoughtsTokenCount: state.reasoningTokens,
          } as any;
        }

        return chunk;
      }

      default:
        return null;
    }
  }

  // --- Private: request helpers ---

  private convertContent(input: any[], content: Content): void {
    const role = content.role;
    const parts = content.parts ?? [];

    if (role === "user") {
      const functionResponses = parts.filter(p => p.functionResponse);
      const otherParts = parts.filter(p => !p.functionResponse);

      if (functionResponses.length > 0) {
        for (const fr of functionResponses) {
          const resp = fr.functionResponse!;
          input.push({
            type: "function_call_output",
            call_id: resp.id ?? resp.name ?? "",
            output:
              typeof resp.response === "string"
                ? resp.response
                : JSON.stringify(resp.response ?? {}),
          });
        }
      }

      if (otherParts.length > 0) {
        const contentParts = otherParts.map(p => {
          if (p.text != null) {
            return { type: "input_text", text: p.text };
          }
          if (p.inlineData) {
            return {
              type: "input_image",
              image_url: `data:${p.inlineData.mimeType};base64,${p.inlineData.data}`,
            };
          }
          if (p.fileData) {
            return { type: "input_image", image_url: p.fileData.fileUri };
          }
          return { type: "input_text", text: "" };
        });
        input.push({
          role: "user",
          type: "message",
          content: contentParts,
        });
      }
    } else if (role === "model") {
      const textParts = parts.filter(p => p.text != null && !p.thought && !p.functionCall);
      const functionCalls = parts.filter(p => p.functionCall);

      if (textParts.length > 0) {
        input.push({
          role: "assistant",
          type: "message",
          content: textParts.map(p => ({
            type: "output_text",
            text: p.text,
          })),
        });
      }

      for (const fc of functionCalls) {
        const call = fc.functionCall!;
        input.push({
          type: "function_call",
          call_id: call.id ?? call.name ?? "",
          name: call.name ?? "",
          arguments: JSON.stringify(call.args ?? {}),
          status: "completed",
        });
      }
    }
  }

  private convertTools(tools: any[]): OpenAI.Responses.Tool[] {
    const result: any[] = [];

    for (const t of tools) {
      if (t.functionDeclarations) {
        for (const fd of t.functionDeclarations as FunctionDeclaration[]) {
          result.push({
            type: "function",
            name: fd.name,
            description: fd.description,
            parameters: fd.parametersJsonSchema ?? fd.parameters,
          });
        }
      }
      if (t.googleSearch != null) {
        result.push({ type: "web_search_preview" });
      }
    }

    return result;
  }

  private convertToolConfig(fcc: any): OpenAI.Responses.ResponseCreateParams["tool_choice"] {
    const mode = fcc.mode as FunctionCallingConfigMode | string;
    switch (mode) {
      case "AUTO":
        return "auto";
      case "ANY":
        if (fcc.allowedFunctionNames?.length === 1) {
          return {
            type: "function",
            name: fcc.allowedFunctionNames[0],
          };
        }
        return "required";
      case "NONE":
        return "none";
      default:
        return "auto";
    }
  }

  private convertThinkingConfig(tc: any): OpenAI.Responses.ResponseCreateParams["reasoning"] {
    if (tc.thinkingBudget === 0) {
      return undefined;
    }
    const budget = tc.thinkingBudget ?? 10240;
    let effort: "low" | "medium" | "high" = "high";
    if (budget <= 2048) effort = "low";
    else if (budget <= 5120) effort = "medium";

    return { effort };
  }

  // --- Private: response helpers ---

  private statusToFinishReason(status: RespResponse["status"], hasToolCall: boolean): string {
    if (hasToolCall) return "STOP";
    switch (status) {
      case "completed":
        return "STOP";
      case "incomplete":
        return "MAX_TOKENS";
      case "failed":
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
      parts: [],
      finishReason: null,
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      cachedTokens: 0,
      reasoningTokens: 0,
    };
  }

  private makeStreamChunk(): GenerateContentResponse {
    return {
      candidates: [
        {
          content: {
            role: "model",
            parts: [...this.streamState.parts],
          },
        } as Candidate,
      ],
      responseId: this.streamState.id,
      modelVersion: this.streamState.model,
    } as GenerateContentResponse;
  }
}
