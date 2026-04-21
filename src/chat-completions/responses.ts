import type OpenAI from "openai";

type RespResponse = OpenAI.Responses.Response;
type RespStreamEvent = OpenAI.Responses.ResponseStreamEvent;
type RespInputItem = OpenAI.Responses.ResponseInputItem;
type RespTool = OpenAI.Responses.Tool;
type CCParams = OpenAI.Chat.Completions.ChatCompletionCreateParams;

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

export class ChatCompletionToResponsesConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion ---

  convertRequest(params: CCParams): OpenAI.Responses.ResponseCreateParams {
    const result: OpenAI.Responses.ResponseCreateParams = {
      model: params.model as string,
      input: this.convertMessages(params.messages),
    };

    if (params.max_completion_tokens != null) {
      result.max_output_tokens = params.max_completion_tokens;
    } else if (params.max_tokens != null) {
      result.max_output_tokens = params.max_tokens;
    }
    if (params.temperature != null) {
      result.temperature = params.temperature;
    }
    if (params.top_p != null) {
      result.top_p = params.top_p;
    }
    if (params.parallel_tool_calls != null) {
      result.parallel_tool_calls = params.parallel_tool_calls;
    }
    if (params.metadata != null) {
      result.metadata = params.metadata;
    }
    if (params.prompt_cache_key != null) {
      result.prompt_cache_key = params.prompt_cache_key;
    }
    if (params.prompt_cache_retention != null) {
      result.prompt_cache_retention = params.prompt_cache_retention;
    }
    if (params.reasoning_effort != null) {
      result.reasoning = { effort: params.reasoning_effort };
    }
    if (params.top_logprobs != null) {
      result.include = ["message.output_text.logprobs"];
    }
    if (params.service_tier != null) {
      result.service_tier = params.service_tier;
    }
    if (params.tools || params.web_search_options != null) {
      result.tools = this.convertTools(params);
    }
    if (params.tool_choice != null) {
      result.tool_choice = this.convertToolChoice(params.tool_choice);
    }
    if (params.response_format) {
      result.text = this.convertResponseFormat(params.response_format);
    }
    if (params.stream) {
      (result as any).stream = true;
    }

    return result;
  }

  // --- Response conversion ---

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

  // --- Stream conversion ---

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

  private convertMessages(
    messages: CCParams["messages"],
  ): RespInputItem[] {
    const input: RespInputItem[] = [];

    for (const msg of messages) {
      if (msg.role === "system" || msg.role === "developer") {
        if (typeof msg.content === "string") {
          input.push({ role: msg.role, type: "message", content: msg.content });
        } else {
          input.push({
            role: msg.role,
            type: "message",
            content: msg.content.map(p => ({ type: "input_text", text: p.text })),
          });
        }
      } else if (msg.role === "user") {
        if (typeof msg.content === "string") {
          input.push({ role: "user", type: "message", content: msg.content });
        } else {
          input.push({
            role: "user",
            type: "message",
            content: msg.content.map(p => {
              if (p.type === "text") return { type: "input_text", text: p.text };
              if (p.type === "image_url")
                return { type: "input_image", image_url: p.image_url.url, detail: p.image_url.detail ?? "auto" };
              if (p.type === "file")
                return { type: "input_file", file_data: p.file.file_data, file_id: p.file.file_id, filename: p.file.filename };
              return { type: "input_text", text: "" };
            }),
          });
        }
      } else if (msg.role === "assistant") {
        if (msg.tool_calls && msg.tool_calls.length > 0) {
          for (const tc of msg.tool_calls) {
            if (tc.type === "function") {
              input.push({
                type: "function_call",
                name: tc.function.name,
                call_id: tc.id,
                arguments: tc.function.arguments,
              });
            }
          }
        } else {
          const content = msg.content;
          if (typeof content === "string") {
            input.push({
              role: "assistant",
              type: "message",
              content: [{ type: "output_text", text: content }],
            } as any);
          } else if (content == null) {
            input.push({
              role: "assistant",
              type: "message",
              content: content ?? "",
            } as any);
          } else {
            input.push({
              role: "assistant",
              type: "message",
              content: content
                .filter((p: any) => p.type === "text")
                .map((p: any) => ({ type: "output_text", text: p.text })),
            } as any);
          }
        }
      } else if (msg.role === "tool") {
        if (typeof msg.content === "string") {
          input.push({
            type: "function_call_output",
            call_id: msg.tool_call_id,
            output: msg.content,
          });
        } else {
          input.push({
            type: "function_call_output",
            call_id: msg.tool_call_id,
            output: msg.content.map((p: any) => ({ type: "input_text", text: p.text })),
          });
        }
      }
    }

    return input;
  }

  private convertTools(params: CCParams): RespTool[] {
    const tools: RespTool[] = [];

    if (params.tools) {
      for (const t of params.tools) {
        if (t.type === "function") {
          tools.push({
            type: "function",
            name: t.function.name,
            description: t.function.description,
            strict: t.function.strict ?? null,
            parameters: t.function.parameters ?? null,
          });
        }
      }
    }

    if (params.web_search_options != null) {
      tools.push({ type: "web_search" });
    }

    return tools;
  }

  private convertToolChoice(
    choice: CCParams["tool_choice"],
  ): OpenAI.Responses.ResponseCreateParams["tool_choice"] {
    if (typeof choice === "string") {
      if (choice === "auto" || choice === "none" || choice === "required") return choice;
      return "auto";
    }
    if (typeof choice === "object" && choice !== null) {
      const c = choice;
      if (c.type === "function" && c.function?.name) {
        return { type: "function", name: c.function.name };
      }
    }
    return "auto";
  }

  private convertResponseFormat(
    format: CCParams["response_format"],
  ): OpenAI.Responses.ResponseCreateParams["text"] {
    if (!format) return undefined;
    if (format.type === "text" || format.type === "json_object") {
      return { format: { type: format.type } };
    }
    if (format.type === "json_schema") {
      return {
        format: {
          type: "json_schema",
          name: format.json_schema.name,
          schema: format.json_schema.schema ?? {},
        },
      };
    }
    return undefined;
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
