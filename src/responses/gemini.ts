import type OpenAI from "openai";
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

type RespResponse = OpenAI.Responses.Response;
type RespStreamEvent = OpenAI.Responses.ResponseStreamEvent;

interface StreamState {
  id: string;
  model: string;
  seq: number;
  started: boolean;
  outputIndex: number;
  reasoningStarted: boolean;
  toolCallCount: number;
  textMessageStarted: boolean;
  prevText: string;
  prevThought: string;
  seenFunctionCallIds: Set<string>;
}

export class ResponsesToGeminiConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (Responses → Gemini, forward) ---

  convertRequest(params: OpenAI.Responses.ResponseCreateParams): GenerateContentParameters {
    const systemParts: Part[] = [];
    const contents: Content[] = [];

    if (params.instructions) {
      systemParts.push({ text: params.instructions });
    }

    this.convertInput(systemParts, contents, params.input);

    const config: GenerateContentConfig = {};

    if (systemParts.length > 0) {
      config.systemInstruction = { parts: systemParts };
    }
    if (params.max_output_tokens != null) {
      config.maxOutputTokens = params.max_output_tokens;
    }
    if (params.temperature != null) {
      config.temperature = params.temperature;
    }
    if (params.top_p != null) {
      config.topP = params.top_p;
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
    if (params.reasoning) {
      config.thinkingConfig = this.convertReasoning(params.reasoning);
    }
    if (params.text?.format) {
      this.applyTextFormat(config, params.text.format);
    }
    if (params.include) {
      for (const inc of params.include) {
        if (inc === "message.output_text.logprobs") {
          config.responseLogprobs = true;
          config.logprobs = 20;
          break;
        }
      }
    }

    return {
      model: params.model as string,
      contents,
      config,
    };
  }

  // --- Response conversion (Gemini → Responses, backward) ---

  convertResponse(response: GenerateContentResponse): RespResponse {
    const candidate = response.candidates?.[0];
    const parts = candidate?.content?.parts ?? [];
    const output: OpenAI.Responses.ResponseOutputItem[] = [];

    for (const part of parts) {
      if (part.thought && part.text) {
        output.push({
          type: "reasoning",
          id: `rs_${this.generateId()}`,
          summary: [{ type: "summary_text", text: part.text }],
        });
      }
    }

    for (const part of parts) {
      if (part.functionCall) {
        const fc = part.functionCall;
        output.push({
          type: "function_call",
          id: `fc_${this.generateId()}`,
          call_id: fc.id ?? fc.name ?? `call_${this.generateId()}`,
          name: fc.name ?? "",
          arguments: JSON.stringify(fc.args ?? {}),
          status: "completed",
        });
      }
    }

    const textParts: string[] = [];
    const annotations: any[] = [];
    for (const part of parts) {
      if (part.text != null && !part.thought) {
        textParts.push(part.text);
      }
    }
    this.extractGroundingAnnotations(candidate, annotations);

    if (textParts.length > 0 || annotations.length > 0) {
      const content: any[] = [];
      content.push({
        type: "output_text",
        text: textParts.join(""),
        annotations,
        logprobs: null as any,
      });
      output.push({
        type: "message",
        id: `msg_${this.generateId()}`,
        role: "assistant",
        status: "completed",
        content,
      });
    }

    if (output.length === 0) {
      output.push({
        type: "message",
        id: `msg_${this.generateId()}`,
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "", annotations: [], logprobs: null as any }],
      });
    }

    const hasToolCall = output.some(o => o.type === "function_call");
    const status = this.finishReasonToStatus(candidate?.finishReason, hasToolCall);

    const usage = response.usageMetadata;
    const promptTokens = (usage?.promptTokenCount ?? 0) + (usage?.toolUsePromptTokenCount ?? 0);
    const thoughtsTokens = usage?.thoughtsTokenCount ?? 0;
    const candidatesTokens = (usage?.candidatesTokenCount ?? 0) + thoughtsTokens;

    return {
      id: response.responseId ?? `resp_${this.generateId()}`,
      object: "response",
      created_at: Math.floor(Date.now() / 1000),
      model: response.modelVersion ?? "",
      output,
      status,
      error: null,
      incomplete_details: status === "incomplete" ? { reason: "max_output_tokens" } : null,
      instructions: null,
      metadata: {},
      temperature: null,
      top_p: null,
      max_output_tokens: null,
      previous_response_id: null,
      parallel_tool_calls: true,
      tool_choice: "auto",
      tools: [],
      text: { format: { type: "text" } },
      reasoning: null,
      truncation: null,
      user: undefined,
      usage: {
        input_tokens: promptTokens,
        output_tokens: candidatesTokens,
        total_tokens: usage?.totalTokenCount ?? 0,
        input_tokens_details: {
          cached_tokens: usage?.cachedContentTokenCount ?? 0,
        },
        output_tokens_details: {
          reasoning_tokens: thoughtsTokens,
        },
      },
    } as unknown as RespResponse;
  }

  // --- Stream conversion (Gemini → Responses, backward) ---

  async *convertStream(
    stream: AsyncIterable<GenerateContentResponse>
  ): AsyncIterable<RespStreamEvent> {
    for await (const chunk of stream) {
      const events = this.convertStreamChunk(chunk);
      for (const event of events) {
        yield event;
      }
    }
  }

  convertStreamChunk(chunk: GenerateContentResponse): RespStreamEvent[] {
    const state = this.streamState;
    const events: RespStreamEvent[] = [];
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
        state.id = `resp_${this.generateId()}`;
      }
      const skeleton = this.makeSkeletonResponse();
      events.push({
        type: "response.created",
        response: skeleton,
        sequence_number: state.seq++,
      });
      events.push({
        type: "response.in_progress",
        response: skeleton,
        sequence_number: state.seq++,
      });
    }

    for (const part of parts) {
      if (part.thought && part.text) {
        if (!state.reasoningStarted) {
          state.reasoningStarted = true;
          events.push({
            type: "response.output_item.added",
            item: {
              type: "reasoning",
              id: `rs_${this.generateId()}`,
              summary: [],
            },
            output_index: state.outputIndex,
            sequence_number: state.seq++,
          } as RespStreamEvent);
          events.push({
            type: "response.reasoning_summary_part.added",
            item_id: `rs_${this.generateId()}`,
            output_index: state.outputIndex,
            summary_index: 0,
            part: { type: "summary_text", text: "" },
            sequence_number: state.seq++,
          } as RespStreamEvent);
        }
        events.push({
          type: "response.reasoning_summary_text.delta",
          item_id: `rs_${this.generateId()}`,
          output_index: state.outputIndex,
          summary_index: 0,
          delta: part.text,
          sequence_number: state.seq++,
        } as RespStreamEvent);
      } else if (part.functionCall) {
        const fc = part.functionCall;
        const fcId = fc.id ?? fc.name ?? "";
        if (!state.seenFunctionCallIds.has(fcId)) {
          state.seenFunctionCallIds.add(fcId);
          if (state.reasoningStarted && state.toolCallCount === 0) {
            state.outputIndex++;
          }
          state.toolCallCount++;
          const toolOutputIndex = state.outputIndex + state.toolCallCount - 1;

          events.push({
            type: "response.output_item.added",
            item: {
              type: "function_call",
              id: `fc_${this.generateId()}`,
              call_id: fc.id ?? `call_${this.generateId()}`,
              name: fc.name ?? "",
              arguments: "",
              status: "in_progress",
            },
            output_index: toolOutputIndex,
            sequence_number: state.seq++,
          } as RespStreamEvent);

          const args = JSON.stringify(fc.args ?? {});
          if (args !== "{}") {
            events.push({
              type: "response.function_call_arguments.delta",
              item_id: `fc_${this.generateId()}`,
              output_index: toolOutputIndex,
              delta: args,
              sequence_number: state.seq++,
            } as RespStreamEvent);
          }
        }
      } else if (part.text != null && part.text !== "") {
        if (!state.textMessageStarted) {
          state.textMessageStarted = true;
          const msgOutputIndex =
            state.outputIndex + (state.reasoningStarted ? 1 : 0) + state.toolCallCount;

          events.push({
            type: "response.output_item.added",
            item: {
              type: "message",
              id: `msg_${this.generateId()}`,
              role: "assistant",
              status: "in_progress",
              content: [],
            },
            output_index: msgOutputIndex,
            sequence_number: state.seq++,
          } as RespStreamEvent);
          events.push({
            type: "response.content_part.added",
            item_id: `msg_${this.generateId()}`,
            output_index: msgOutputIndex,
            content_index: 0,
            part: { type: "output_text", text: "", annotations: [] },
            sequence_number: state.seq++,
          } as RespStreamEvent);
        }
        events.push({
          type: "response.output_text.delta",
          item_id: `msg_${this.generateId()}`,
          output_index: state.outputIndex + (state.reasoningStarted ? 1 : 0) + state.toolCallCount,
          content_index: 0,
          delta: part.text,
          sequence_number: state.seq++,
        } as RespStreamEvent);
      }
    }

    if (candidate?.finishReason) {
      const hasToolCall = state.seenFunctionCallIds.size > 0;
      const status = this.finishReasonToStatus(candidate.finishReason, hasToolCall);

      const resp = this.makeSkeletonResponse();
      resp.status = status as any;

      const usage = chunk.usageMetadata;
      if (usage) {
        const promptTokens = (usage.promptTokenCount ?? 0) + (usage.toolUsePromptTokenCount ?? 0);
        const thoughtsTokens = usage.thoughtsTokenCount ?? 0;
        const candidatesTokens = (usage.candidatesTokenCount ?? 0) + thoughtsTokens;

        resp.usage = {
          input_tokens: promptTokens,
          output_tokens: candidatesTokens,
          total_tokens: usage.totalTokenCount ?? 0,
          input_tokens_details: {
            cached_tokens: usage.cachedContentTokenCount ?? 0,
          },
          output_tokens_details: {
            reasoning_tokens: thoughtsTokens,
          },
        };
      }

      if (status === "incomplete") {
        events.push({
          type: "response.incomplete",
          response: resp,
          sequence_number: state.seq++,
        } as RespStreamEvent);
      } else {
        events.push({
          type: "response.completed",
          response: resp,
          sequence_number: state.seq++,
        } as RespStreamEvent);
      }
    }

    return events;
  }

  // --- Private: request helpers ---

  private convertInput(
    systemParts: Part[],
    contents: Content[],
    input: OpenAI.Responses.ResponseCreateParams["input"]
  ): void {
    if (typeof input === "string") {
      contents.push({ role: "user", parts: [{ text: input }] });
      return;
    }

    const pendingFunctionCalls: Part[] = [];

    const flushFunctionCalls = () => {
      if (pendingFunctionCalls.length > 0) {
        contents.push({ role: "model", parts: [...pendingFunctionCalls] });
        pendingFunctionCalls.length = 0;
      }
    };

    for (const item of input!) {
      const typed = item as any;

      if (typed.type === "message") {
        flushFunctionCalls();
        if (typed.role === "system" || typed.role === "developer") {
          const text = this.extractText(typed.content);
          if (text) systemParts.push({ text });
        } else if (typed.role === "user") {
          const parts = this.convertInputContent(typed.content);
          contents.push({ role: "user", parts });
        } else if (typed.role === "assistant") {
          const parts: Part[] = [];
          if (Array.isArray(typed.content)) {
            for (const p of typed.content) {
              if (p.type === "output_text") parts.push({ text: p.text });
            }
          }
          if (parts.length > 0) {
            contents.push({ role: "model", parts });
          }
        }
      } else if (typed.type === "function_call") {
        let args: Record<string, unknown>;
        try {
          args = JSON.parse(typed.arguments);
        } catch {
          args = {};
        }
        pendingFunctionCalls.push({
          functionCall: {
            id: typed.call_id,
            name: typed.name,
            args,
          },
        });
      } else if (typed.type === "function_call_output") {
        flushFunctionCalls();
        const responsePart: Part = {
          functionResponse: {
            id: typed.call_id,
            name: typed.call_id,
            response: { output: typed.output ?? "" },
          },
        };

        const last = contents[contents.length - 1];
        if (last && last.role === "user" && last.parts?.length) {
          const lastPart = last.parts[last.parts.length - 1];
          if (lastPart.functionResponse) {
            last.parts.push(responsePart);
            continue;
          }
        }
        contents.push({ role: "user", parts: [responsePart] });
      } else if ("role" in typed && "content" in typed && typeof typed.content === "string") {
        flushFunctionCalls();
        const role = typed.role;
        if (role === "system" || role === "developer") {
          systemParts.push({ text: typed.content });
        } else if (role === "user") {
          contents.push({
            role: "user",
            parts: [{ text: typed.content }],
          });
        } else if (role === "assistant") {
          contents.push({
            role: "model",
            parts: [{ text: typed.content }],
          });
        }
      }
    }

    flushFunctionCalls();
  }

  private extractText(content: any): string {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .filter((p: any) => p.type === "input_text")
        .map((p: any) => p.text)
        .join("\n");
    }
    return "";
  }

  private convertInputContent(content: any): Part[] {
    if (typeof content === "string") {
      return [{ text: content }];
    }
    if (!Array.isArray(content)) return [{ text: "" }];

    const parts: Part[] = [];
    for (const p of content) {
      if (p.type === "input_text") {
        parts.push({ text: p.text });
      } else if (p.type === "input_image") {
        const url: string = p.image_url || "";
        const match = url.match(/^data:([^;]+);base64,(.+)$/);
        if (match) {
          parts.push({
            inlineData: { mimeType: match[1], data: match[2] },
          });
        } else {
          parts.push({ fileData: { fileUri: url, mimeType: "image/*" } });
        }
      } else if (p.type === "input_file") {
        const data: string = p.file_data || p.file_url || "";
        const match = data.match(/^data:([^;]+);base64,(.+)$/);
        if (match) {
          parts.push({
            inlineData: { mimeType: match[1], data: match[2] },
          });
        } else {
          parts.push({
            fileData: {
              fileUri: data,
              mimeType: "application/octet-stream",
            },
          });
        }
      }
    }
    return parts.length > 0 ? parts : [{ text: "" }];
  }

  private convertTools(tools: OpenAI.Responses.ResponseCreateParams["tools"]): {
    tools: FunctionDeclaration[];
    hasGoogleSearch: boolean;
  } {
    if (!tools) return { tools: [], hasGoogleSearch: false };

    const functionTools: FunctionDeclaration[] = [];
    let hasGoogleSearch = false;

    for (const t of tools) {
      const tt = t as any;
      if (tt.type === "function") {
        functionTools.push({
          name: tt.name,
          description: tt.description,
          parametersJsonSchema: tt.parameters,
        });
      } else if (this.isWebSearch(tt)) {
        hasGoogleSearch = true;
      }
    }

    return { tools: functionTools, hasGoogleSearch };
  }

  private isWebSearch(tool: any): boolean {
    const t = tool.type;
    return (
      t === "web_search" ||
      t === "web_search_2025_08_26" ||
      t === "web_search_preview" ||
      t === "web_search_preview_2025_03_11"
    );
  }

  private convertToolChoice(choice: OpenAI.Responses.ResponseCreateParams["tool_choice"]): {
    mode?: FunctionCallingConfigMode;
    allowedFunctionNames?: string[];
  } {
    if (typeof choice === "string") {
      switch (choice) {
        case "auto":
          return { mode: "AUTO" as FunctionCallingConfigMode };
        case "required":
          return { mode: "ANY" as FunctionCallingConfigMode };
        case "none":
          return { mode: "NONE" as FunctionCallingConfigMode };
        default:
          return { mode: "AUTO" as FunctionCallingConfigMode };
      }
    }
    if (typeof choice === "object" && choice !== null) {
      if (choice.type === "function" && choice.name) {
        return {
          mode: "ANY" as FunctionCallingConfigMode,
          allowedFunctionNames: [choice.name],
        };
      }
    }
    return { mode: "AUTO" as FunctionCallingConfigMode };
  }

  private convertReasoning(reasoning: OpenAI.Responses.ResponseCreateParams["reasoning"]): {
    includeThoughts?: boolean;
    thinkingBudget?: number;
  } {
    if (!reasoning || !reasoning.effort) {
      return { thinkingBudget: 0 };
    }
    const budgetMap: Record<string, number> = {
      low: 2048,
      medium: 5120,
      high: 10240,
    };
    return {
      includeThoughts: true,
      thinkingBudget: budgetMap[reasoning.effort] ?? 10240,
    };
  }

  private applyTextFormat(config: GenerateContentConfig, format: any): void {
    if (format.type === "json_schema") {
      config.responseMimeType = "application/json";
      config.responseJsonSchema = format.schema;
    } else if (format.type === "json_object") {
      config.responseMimeType = "application/json";
    }
  }

  // --- Private: response helpers ---

  private finishReasonToStatus(
    reason: FinishReason | string | null | undefined,
    hasToolCall: boolean
  ): RespResponse["status"] {
    if (hasToolCall) return "completed";
    switch (reason) {
      case "STOP":
        return "completed";
      case "MAX_TOKENS":
        return "incomplete";
      case "SAFETY":
      case "RECITATION":
      case "BLOCKLIST":
      case "PROHIBITED_CONTENT":
        return "failed";
      default:
        return "completed";
    }
  }

  private extractGroundingAnnotations(candidate: any, annotations: any[]): void {
    if (!candidate?.groundingMetadata?.groundingChunks) return;
    for (const chunk of candidate.groundingMetadata.groundingChunks) {
      const web = chunk.web;
      if (web) {
        annotations.push({
          type: "url_citation",
          url: web.uri || "",
          title: web.title || "",
          start_index: 0,
          end_index: 0,
        });
      }
    }
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  // --- Private: stream helpers ---

  private createStreamState(): StreamState {
    return {
      id: "",
      model: "",
      seq: 0,
      started: false,
      outputIndex: 0,
      reasoningStarted: false,
      toolCallCount: 0,
      textMessageStarted: false,
      prevText: "",
      prevThought: "",
      seenFunctionCallIds: new Set(),
    };
  }

  private makeSkeletonResponse(): RespResponse {
    return {
      id: this.streamState.id,
      object: "response",
      created_at: Math.floor(Date.now() / 1000),
      model: this.streamState.model,
      output: [],
      status: "in_progress",
      error: null,
      incomplete_details: null,
      instructions: null,
      metadata: {},
      temperature: null,
      top_p: null,
      max_output_tokens: null,
      previous_response_id: null,
      parallel_tool_calls: true,
      tool_choice: "auto",
      tools: [],
      text: { format: { type: "text" } },
      reasoning: null,
      truncation: null,
      user: undefined,
    } as unknown as RespResponse;
  }
}
