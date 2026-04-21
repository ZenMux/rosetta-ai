import type OpenAI from "openai";

type RespResponse = OpenAI.Responses.Response;
type RespStreamEvent = OpenAI.Responses.ResponseStreamEvent;

interface StreamState {
  id: string;
  model: string;
  created: number;
  seq: number;
  outputIndex: number;
  contentIndex: number;
  toolCallStarted: boolean;
  reasoningStarted: boolean;
  messageStarted: boolean;
  toolCallCount: number;
}

export class ResponsesToChatCompletionConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (Responses → CC, forward) ---

  convertRequest(params: OpenAI.Responses.ResponseCreateParams): OpenAI.Chat.Completions.ChatCompletionCreateParams {
    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];

    if (params.instructions) {
      messages.push({ role: "system", content: params.instructions });
    }

    this.convertInput(messages, params.input);

    const result: OpenAI.Chat.Completions.ChatCompletionCreateParams = {
      model: params.model as string,
      messages,
    };

    if (params.max_output_tokens != null) {
      result.max_completion_tokens = params.max_output_tokens;
    }
    if (params.temperature != null) {
      result.temperature = params.temperature;
    }
    if (params.top_p != null) {
      result.top_p = params.top_p;
    }
    if (params.tools) {
      const { tools, webSearchOptions } = this.convertTools(params.tools);
      if (tools.length > 0) {
        result.tools = tools;
      }
      if (webSearchOptions) {
        result.web_search_options = webSearchOptions;
      }
    }
    if (params.tool_choice != null) {
      result.tool_choice = this.convertToolChoice(params.tool_choice);
    }
    if (params.parallel_tool_calls != null) {
      result.parallel_tool_calls = params.parallel_tool_calls;
    }
    if (params.reasoning) {
      result.reasoning_effort = params.reasoning.effort ?? null;
    }
    if (params.text?.format) {
      result.response_format = this.convertTextFormat(params.text.format);
    }
    if (params.text?.verbosity) {
      result.verbosity = params.text.verbosity;
    }
    if (params.metadata) {
      result.metadata = params.metadata;
    }
    if (params.service_tier != null) {
      result.service_tier = params.service_tier;
    }
    if (params.prompt_cache_key) {
      result.prompt_cache_key = params.prompt_cache_key;
    }
    if (params.prompt_cache_retention != null) {
      result.prompt_cache_retention = params.prompt_cache_retention;
    }
    if (params.safety_identifier) {
      result.safety_identifier = params.safety_identifier;
    }
    if (params.include) {
      for (const inc of params.include) {
        if (inc === "message.output_text.logprobs") {
          result.top_logprobs = 20;
          break;
        }
      }
    }
    if (params.stream === true) {
      (result as any).stream = true;
      if (params.stream_options) {
        result.stream_options = params.stream_options;
      }
    }

    return result;
  }

  // --- Response conversion (CC → Responses, backward) ---

  convertResponse(response: OpenAI.ChatCompletion): RespResponse {
    const choice = response.choices[0];
    const msg = choice?.message;
    const output: OpenAI.Responses.ResponseOutputItem[] = [];

    // Reasoning
    const reasoning = (msg as any)?.reasoning || (msg as any)?.reasoning_content;
    if (reasoning) {
      output.push({
        type: "reasoning",
        id: `rs_${this.generateId()}`,
        summary: [{ type: "summary_text", text: reasoning }],
      });
    }

    // Tool calls
    if (msg?.tool_calls && msg.tool_calls.length > 0) {
      for (const tc of msg.tool_calls) {
        if (tc.type === "function") {
          output.push({
            type: "function_call",
            id: `fc_${this.generateId()}`,
            call_id: tc.id,
            name: tc.function.name,
            arguments: tc.function.arguments,
            status: "completed",
          });
        }
      }
    }

    // Message
    if (msg?.content != null || msg?.refusal != null) {
      const content: any[] = [];
      if (msg.refusal) {
        content.push({ type: "refusal", refusal: msg.refusal });
      } else {
        const annotations = (msg.annotations ?? [])
          .filter((a: any) => a.type === "url_citation")
          .map((a: any) => ({
            type: "url_citation",
            url: a.url_citation.url,
            title: a.url_citation.title,
            start_index: a.url_citation.start_index,
            end_index: a.url_citation.end_index,
          }));
        content.push({
          type: "output_text",
          text: msg.content ?? "",
          annotations,
          logprobs: null,
        });
      }
      output.push({
        type: "message",
        id: `msg_${this.generateId()}`,
        role: "assistant",
        status: "completed",
        content,
      });
    }

    const status = this.finishReasonToStatus(choice?.finish_reason);

    return {
      id: response.id,
      object: "response",
      created_at: response.created,
      model: response.model,
      output,
      status,
      error: null,
      incomplete_details:
        status === "incomplete" ? { reason: "max_output_tokens" } : null,
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
      usage: response.usage
        ? {
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
            total_tokens: response.usage.total_tokens,
            input_tokens_details: {
              cached_tokens:
                response.usage.prompt_tokens_details?.cached_tokens ?? 0,
            },
            output_tokens_details: {
              reasoning_tokens:
                response.usage.completion_tokens_details?.reasoning_tokens ?? 0,
            },
          }
        : undefined,
    } as unknown as RespResponse;
  }

  // --- Stream conversion (CC → Responses, backward) ---

  async *convertStream(
    stream: AsyncIterable<OpenAI.ChatCompletionChunk>,
  ): AsyncIterable<RespStreamEvent> {
    for await (const chunk of stream) {
      const events = this.convertStreamChunk(chunk);
      for (const event of events) {
        yield event;
      }
    }
  }

  convertStreamChunk(chunk: OpenAI.ChatCompletionChunk): RespStreamEvent[] {
    const state = this.streamState;
    const events: RespStreamEvent[] = [];
    const choice = chunk.choices?.[0];

    // First chunk — emit response.created + response.in_progress
    if (state.id === "") {
      state.id = chunk.id;
      state.model = chunk.model;
      state.created = chunk.created;

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

    if (!choice) {
      // Usage-only chunk (empty choices) at the end
      if (chunk.usage) {
        events.push(
          ...this.emitCompleted(chunk.usage)
        );
      }
      return events;
    }

    const delta = choice.delta;

    // Reasoning delta
    const reasoning = (delta as any)?.reasoning || (delta as any)?.reasoning_content;
    if (reasoning) {
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
        delta: reasoning,
        sequence_number: state.seq++,
      } as RespStreamEvent);
    }

    // Tool call start
    if (delta?.tool_calls) {
      for (const tc of delta.tool_calls) {
        if (tc.id && tc.function?.name) {
          if (state.reasoningStarted && !state.toolCallStarted) {
            state.outputIndex++;
          }
          state.toolCallStarted = true;
          state.toolCallCount++;

          events.push({
            type: "response.output_item.added",
            item: {
              type: "function_call",
              id: `fc_${this.generateId()}`,
              call_id: tc.id,
              name: tc.function.name,
              arguments: "",
              status: "in_progress",
            },
            output_index: state.outputIndex + state.toolCallCount - 1,
            sequence_number: state.seq++,
          } as RespStreamEvent);
        }

        // Tool call arguments delta
        if (tc.function?.arguments) {
          events.push({
            type: "response.function_call_arguments.delta",
            item_id: `fc_${this.generateId()}`,
            output_index: state.outputIndex + state.toolCallCount - 1,
            delta: tc.function.arguments,
            sequence_number: state.seq++,
          } as RespStreamEvent);
        }
      }
    }

    // Text content delta
    if (delta?.content && delta.content !== "") {
      if (!state.messageStarted) {
        state.messageStarted = true;
        const msgOutputIndex = state.outputIndex +
          (state.reasoningStarted ? 1 : 0) +
          state.toolCallCount;
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
        output_index: state.outputIndex +
          (state.reasoningStarted ? 1 : 0) +
          state.toolCallCount,
        content_index: 0,
        delta: delta.content,
        sequence_number: state.seq++,
      } as RespStreamEvent);
    }

    // Finish reason
    if (choice.finish_reason) {
      const status = this.finishReasonToStatus(choice.finish_reason);
      if (chunk.usage) {
        events.push(...this.emitCompleted(chunk.usage, status));
      } else {
        events.push(...this.emitCompleted(undefined, status));
      }
    }

    return events;
  }

  // --- Private: request helpers ---

  private convertInput(
    messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
    input: OpenAI.Responses.ResponseCreateParams["input"],
  ): void {
    if (typeof input === "string") {
      messages.push({ role: "user", content: input });
      return;
    }

    const pendingToolCalls: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[] = [];

    const flushToolCalls = () => {
      if (pendingToolCalls.length > 0) {
        messages.push({
          role: "assistant",
          tool_calls: [...pendingToolCalls],
        });
        pendingToolCalls.length = 0;
      }
    };

    for (const item of input!) {
      if ("role" in item && "content" in item && typeof item.content === "string") {
        flushToolCalls();
        const role = item.role;
        if (role === "user" || role === "system" || role === "developer" || role === "assistant") {
          messages.push({
            role,
            content: item.content,
          } as OpenAI.Chat.Completions.ChatCompletionMessageParam);
        }
        continue;
      }

      const typed = item as any;
      if (typed.type === "message") {
        flushToolCalls();
        if (typed.role === "user" || typed.role === "system" || typed.role === "developer") {
          const content = this.convertInputContent(typed.content);
          messages.push({
            role: typed.role,
            content,
          } as OpenAI.Chat.Completions.ChatCompletionMessageParam);
        } else if (typed.role === "assistant") {
          const textParts: string[] = [];
          if (Array.isArray(typed.content)) {
            for (const part of typed.content) {
              if (part.type === "output_text") textParts.push(part.text);
            }
          }
          const msg: OpenAI.Chat.Completions.ChatCompletionAssistantMessageParam = {
            role: "assistant",
          };
          if (textParts.length > 0) msg.content = textParts.join("");
          messages.push(msg);
        }
      } else if (typed.type === "reasoning") {
        flushToolCalls();
        const summaryTexts: string[] = [];
        if (Array.isArray(typed.summary)) {
          for (const s of typed.summary) {
            if (s.type === "summary_text") summaryTexts.push(s.text);
          }
        }
        if (summaryTexts.length > 0) {
          const msg: any = { role: "assistant" };
          msg.reasoning_content = summaryTexts.join("");
          messages.push(msg);
        }
      } else if (typed.type === "function_call") {
        pendingToolCalls.push({
          id: typed.call_id,
          type: "function",
          function: { name: typed.name, arguments: typed.arguments },
        });
      } else if (typed.type === "function_call_output") {
        flushToolCalls();
        messages.push({
          role: "tool",
          tool_call_id: typed.call_id,
          content: typed.output,
        });
      }
    }

    flushToolCalls();
  }

  private convertInputContent(
    content: any[],
  ): string | OpenAI.Chat.Completions.ChatCompletionContentPart[] {
    if (!Array.isArray(content)) return "";

    if (content.length === 1 && content[0].type === "input_text") {
      return content[0].text;
    }

    return content.map((part): OpenAI.Chat.Completions.ChatCompletionContentPart => {
      if (part.type === "input_text") {
        return { type: "text", text: part.text };
      }
      if (part.type === "input_image") {
        return {
          type: "image_url",
          image_url: { url: part.image_url || part.file_id || "" },
        };
      }
      if (part.type === "input_file") {
        return {
          type: "file",
          file: {
            file_data: part.file_data || part.file_url || "",
            filename: part.filename,
            file_id: part.file_id,
          },
        };
      }
      return { type: "text", text: "" };
    });
  }

  private convertTools(tools: OpenAI.Responses.ResponseCreateParams["tools"]): {
    tools: OpenAI.Chat.Completions.ChatCompletionTool[];
    webSearchOptions?: Record<string, unknown>;
  } {
    if (!tools) return { tools: [] };

    const ccTools: OpenAI.Chat.Completions.ChatCompletionTool[] = [];
    let webSearchOptions: Record<string, unknown> | undefined;

    for (const t of tools) {
      const tt = t as any;
      if (tt.type === "function") {
        ccTools.push({
          type: "function",
          function: {
            name: tt.name,
            description: tt.description,
            parameters: tt.parameters ?? { type: "object" },
            strict: tt.strict,
          },
        });
      } else if (this.isWebSearch(tt)) {
        webSearchOptions = {};
        if (tt.search_context_size) {
          webSearchOptions.search_context_size = tt.search_context_size;
        }
        if (tt.user_location) {
          webSearchOptions.user_location = tt.user_location;
        }
      }
    }

    return { tools: ccTools, webSearchOptions };
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

  private convertToolChoice(
    choice: OpenAI.Responses.ResponseCreateParams["tool_choice"],
  ): OpenAI.Chat.Completions.ChatCompletionToolChoiceOption {
    if (typeof choice === "string") {
      if (choice === "auto" || choice === "none" || choice === "required") return choice;
      return "auto";
    }
    if (typeof choice === "object" && choice !== null) {
      const c = choice;
      if (c.type === "function" && c.name) {
        return { type: "function", function: { name: c.name } };
      }
    }
    return "auto";
  }

  private convertTextFormat(
    format: any,
  ): OpenAI.Chat.Completions.ChatCompletionCreateParams["response_format"] {
    if (format.type === "json_schema") {
      return {
        type: "json_schema",
        json_schema: {
          name: format.name || "response",
          schema: format.schema,
          strict: format.strict,
        },
      };
    }
    if (format.type === "json_object") {
      return { type: "json_object" };
    }
    return { type: "text" };
  }

  // --- Private: response helpers ---

  private finishReasonToStatus(
    reason: string | null | undefined,
  ): RespResponse["status"] {
    switch (reason) {
      case "stop":
      case "tool_calls":
        return "completed";
      case "length":
        return "incomplete";
      case "content_filter":
        return "failed";
      default:
        return "completed";
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
      created: 0,
      seq: 0,
      outputIndex: 0,
      contentIndex: 0,
      toolCallStarted: false,
      reasoningStarted: false,
      messageStarted: false,
      toolCallCount: 0,
    };
  }

  private makeSkeletonResponse(): RespResponse {
    return {
      id: this.streamState.id,
      object: "response",
      created_at: this.streamState.created,
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

  private emitCompleted(
    usage?: OpenAI.CompletionUsage,
    status?: RespResponse["status"],
  ): RespStreamEvent[] {
    const state = this.streamState;
    const finalStatus = status ?? "completed";
    const resp = this.makeSkeletonResponse();
    resp.status = finalStatus;

    if (usage) {
      resp.usage = {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
        input_tokens_details: {
          cached_tokens: usage.prompt_tokens_details?.cached_tokens ?? 0,
        },
        output_tokens_details: {
          reasoning_tokens: usage.completion_tokens_details?.reasoning_tokens ?? 0,
        },
      };
    }

    if (finalStatus === "incomplete") {
      return [
        {
          type: "response.incomplete",
          response: resp,
          sequence_number: state.seq++,
        } as RespStreamEvent,
      ];
    }

    return [
      {
        type: "response.completed",
        response: resp,
        sequence_number: state.seq++,
      } as RespStreamEvent,
    ];
  }
}
