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
  serviceTier: OpenAI.ChatCompletionChunk["service_tier"] | null;
  toolCallCounter: number;
  hasToolCall: boolean;
  webSearchCount: number;
  annotations: OpenAI.ChatCompletionMessage.Annotation[];
  includeUsage: boolean;
}

export class ChatCompletionToResponsesConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (CC → Responses, forward) ---

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

  // --- Response conversion (Responses → CC, backward) ---

  convertResponse(response: RespResponse): OpenAI.ChatCompletion {
    const choices: OpenAI.ChatCompletion.Choice[] = [];
    const toolCalls: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[] = [];
    let reasoningItem: OpenAI.Responses.ResponseReasoningItem | null = null;
    let webSearchCount = 0;

    for (const item of response.output) {
      if (item.type === "message") {
        choices.push(this.convertOutputMessage(
          item as OpenAI.Responses.ResponseOutputMessage,
          reasoningItem,
          response.incomplete_details,
        ));
      } else if (item.type === "function_call") {
        const fc = item as OpenAI.Responses.ResponseFunctionToolCall;
        toolCalls.push({
          id: fc.call_id,
          type: "function",
          function: { name: fc.name, arguments: fc.arguments },
        });
      } else if (item.type === "reasoning") {
        reasoningItem = item as OpenAI.Responses.ResponseReasoningItem;
        if (response.output.length === 1) {
          choices.push({
            index: 0,
            finish_reason: null as any,
            message: {
              role: "assistant",
              content: null,
              refusal: null,
              ...this.convertReasoning(reasoningItem),
            },
            logprobs: null,
          });
        }
      } else if (item.type === "web_search_call") {
        webSearchCount++;
      }
    }

    if (toolCalls.length > 0) {
      choices.push({
        index: 0,
        finish_reason: "tool_calls",
        message: {
          role: "assistant",
          content: null,
          refusal: null,
          tool_calls: toolCalls,
        },
        logprobs: null,
      });
    }

    return {
      id: response.id,
      object: "chat.completion",
      created: response.created_at,
      model: response.model as string,
      service_tier: response.service_tier ?? undefined,
      choices,
      usage: this.convertUsage(response.usage!, webSearchCount),
    };
  }

  private convertOutputMessage(
    message: OpenAI.Responses.ResponseOutputMessage,
    reasoning: OpenAI.Responses.ResponseReasoningItem | null,
    incompleteDetails?: RespResponse["incomplete_details"],
  ): OpenAI.ChatCompletion.Choice {
    const content = message.content[0];
    let finishReason = this.messageStatusToFinishReason(message.status);
    if (finishReason === "length" && incompleteDetails?.reason === "content_filter") {
      finishReason = "content_filter";
    }

    if (content?.type === "refusal") {
      return {
        index: 0,
        finish_reason: finishReason,
        logprobs: null,
        message: {
          role: "assistant",
          content: null,
          refusal: content.refusal,
          ...this.convertReasoning(reasoning),
        },
      };
    }

    const textContent = content?.type === "output_text" ? content : null;
    const annotations = (textContent?.annotations ?? [])
      .filter((a: any) => a.type === "url_citation")
      .map((a: any) => ({
        type: "url_citation" as const,
        url_citation: {
          title: a.title,
          url: a.url,
          start_index: a.start_index,
          end_index: a.end_index,
        },
      }));

    return {
      index: 0,
      finish_reason: finishReason,
      message: {
        role: "assistant",
        content: textContent?.text ?? null,
        refusal: null,
        ...(annotations.length > 0 ? { annotations } : {}),
        ...this.convertReasoning(reasoning),
      },
      logprobs: textContent?.logprobs
        ? {
            content: textContent.logprobs.map((l: any) => ({
              token: l.token,
              logprob: l.logprob,
              bytes: l.bytes,
              top_logprobs: l.top_logprobs,
            })),
            refusal: null,
          }
        : null,
    };
  }

  private convertReasoning(
    reasoning: OpenAI.Responses.ResponseReasoningItem | null,
  ): Record<string, any> {
    if (!reasoning || !reasoning.summary || reasoning.summary.length === 0) return {};

    const summary = reasoning.summary[0].text;
    const details: any[] = [
      {
        index: "0",
        format: "openai-responses-v1",
        type: "reasoning.summary",
        summary,
      },
    ];

    if (reasoning.encrypted_content) {
      details.push({
        id: reasoning.id,
        index: "0",
        format: "openai-responses-v1",
        type: "reasoning.encrypted",
        data: reasoning.encrypted_content,
      });
    }

    return { reasoning: summary, reasoning_details: details };
  }

  private messageStatusToFinishReason(
    status: OpenAI.Responses.ResponseOutputMessage["status"],
  ): OpenAI.ChatCompletion.Choice["finish_reason"] {
    if (status === "completed") return "stop";
    if (status === "incomplete") return "length";
    return "stop";
  }

  // --- Stream conversion (Responses → CC, backward) ---

  async *convertStream(
    stream: AsyncIterable<RespStreamEvent>,
  ): AsyncIterable<OpenAI.ChatCompletionChunk> {
    for await (const event of stream) {
      const result = this.convertStreamEvent(event);
      if (result) {
        if (Array.isArray(result)) {
          for (const chunk of result) yield chunk;
        } else {
          yield result;
        }
      }
    }
  }

  convertStreamEvent(
    event: RespStreamEvent,
  ): OpenAI.ChatCompletionChunk | OpenAI.ChatCompletionChunk[] | null {
    switch (event.type) {
      case "response.created":
        return this.handleResponseCreated(event);
      case "response.output_item.added":
        return this.handleOutputItemAdded(event);
      case "response.output_text.delta":
        return this.handleTextDelta(event);
      case "response.function_call_arguments.delta":
        return this.handleFunctionCallDelta(event);
      case "response.refusal.delta":
        return this.handleRefusalDelta(event);
      case "response.reasoning_summary_text.delta":
        return this.handleReasoningSummaryDelta(event);
      case "response.completed":
        return this.handleCompleted(event);
      case "response.incomplete":
        return this.handleIncomplete(event);
      case "response.failed":
        return this.handleFailed();
      case "response.web_search_call.completed":
        this.streamState.webSearchCount++;
        return null;
      case "response.output_text.annotation.added":
        this.streamState.annotations.push({
          type: "url_citation",
          url_citation: (event as any).annotation,
        });
        return null;
      default:
        return null;
    }
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

  private mapStatus(
    status: RespResponse["status"],
    incompleteDetails?: RespResponse["incomplete_details"],
  ): OpenAI.ChatCompletion.Choice["finish_reason"] {
    switch (status) {
      case "completed":
        return "stop";
      case "incomplete":
        if (incompleteDetails?.reason === "content_filter") return "content_filter";
        return "length";
      case "failed":
      case "cancelled":
      default:
        return "stop";
    }
  }

  private convertUsage(
    usage: OpenAI.Responses.ResponseUsage,
    webSearchCount = 0,
  ): OpenAI.CompletionUsage {
    return {
      prompt_tokens: usage.input_tokens,
      completion_tokens: usage.output_tokens,
      total_tokens: usage.total_tokens,
      prompt_tokens_details: {
        cached_tokens: usage.input_tokens_details?.cached_tokens ?? 0,
        ...(webSearchCount > 0 ? { web_search: webSearchCount } : {}),
      },
      completion_tokens_details: {
        reasoning_tokens: usage.output_tokens_details?.reasoning_tokens ?? 0,
      },
    };
  }

  // --- Private: stream helpers ---

  private createStreamState(): StreamState {
    return {
      id: "",
      model: "",
      created: 0,
      serviceTier: null,
      toolCallCounter: -1,
      hasToolCall: false,
      webSearchCount: 0,
      annotations: [],
      includeUsage: false,
    };
  }

  private makeChunk(
    delta: OpenAI.ChatCompletionChunk.Choice.Delta,
    finish_reason: OpenAI.ChatCompletionChunk.Choice["finish_reason"] = null,
    usage?: OpenAI.CompletionUsage,
  ): OpenAI.ChatCompletionChunk {
    return {
      id: this.streamState.id,
      object: "chat.completion.chunk",
      created: this.streamState.created,
      model: this.streamState.model,
      service_tier: this.streamState.serviceTier,
      choices: [{ index: 0, delta, finish_reason }],
      ...(usage ? { usage } : {}),
    };
  }

  private handleResponseCreated(
    event: OpenAI.Responses.ResponseCreatedEvent,
  ): OpenAI.ChatCompletionChunk {
    const resp = event.response;
    this.streamState.id = resp.id;
    this.streamState.model = resp.model as string;
    this.streamState.created = resp.created_at;
    if (resp.service_tier) {
      this.streamState.serviceTier = resp.service_tier;
    }
    return this.makeChunk({ role: "assistant" });
  }

  private handleOutputItemAdded(
    event: OpenAI.Responses.ResponseOutputItemAddedEvent,
  ): OpenAI.ChatCompletionChunk | null {
    const item = event.item;
    if (item.type === "function_call") {
      this.streamState.hasToolCall = true;
      this.streamState.toolCallCounter++;
      const fc = item as OpenAI.Responses.ResponseFunctionToolCall;
      return this.makeChunk({
        role: "assistant",
        content: null,
        tool_calls: [
          {
            index: this.streamState.toolCallCounter,
            id: fc.call_id,
            type: "function",
            function: { name: fc.name, arguments: "" },
          },
        ],
      });
    }
    return null;
  }

  private handleTextDelta(
    event: OpenAI.Responses.ResponseTextDeltaEvent,
  ): OpenAI.ChatCompletionChunk {
    return this.makeChunk({ role: "assistant", content: event.delta });
  }

  private handleFunctionCallDelta(
    event: OpenAI.Responses.ResponseFunctionCallArgumentsDeltaEvent,
  ): OpenAI.ChatCompletionChunk {
    return this.makeChunk({
      role: "assistant",
      content: "",
      tool_calls: [
        {
          index: this.streamState.toolCallCounter,
          type: "function",
          function: { arguments: event.delta },
        },
      ],
    });
  }

  private handleRefusalDelta(
    event: OpenAI.Responses.ResponseRefusalDeltaEvent,
  ): OpenAI.ChatCompletionChunk {
    return this.makeChunk({ refusal: event.delta });
  }

  private handleReasoningSummaryDelta(
    event: { delta: string; item_id: string },
  ): OpenAI.ChatCompletionChunk {
    return this.makeChunk({
      role: "assistant",
      content: "",
      ...({
        reasoning: event.delta,
        reasoning_details: [
          {
            index: "0",
            type: "reasoning.summary",
            format: "openai-responses-v1",
            summary: event.delta,
          },
        ],
      }),
    });
  }

  private handleCompleted(
    event: OpenAI.Responses.ResponseCompletedEvent,
  ): OpenAI.ChatCompletionChunk | OpenAI.ChatCompletionChunk[] {
    const state = this.streamState;
    const resp = event.response;
    const finishReason = state.hasToolCall
      ? "tool_calls" as const
      : this.mapStatus(resp.status, resp.incomplete_details);

    const delta: any = { content: null };
    if (state.annotations.length > 0) {
      delta.annotations = state.annotations;
    }

    const finishChunk = this.makeChunk(delta, finishReason);
    const chunks: OpenAI.ChatCompletionChunk[] = [finishChunk];

    if (state.includeUsage && resp.usage) {
      const usageChunk: OpenAI.ChatCompletionChunk = {
        id: state.id,
        object: "chat.completion.chunk",
        created: state.created,
        model: state.model,
        service_tier: state.serviceTier,
        choices: [],
        usage: this.convertUsage(resp.usage, state.webSearchCount),
      };
      chunks.push(usageChunk);
    }

    return chunks.length === 1 ? chunks[0] : chunks;
  }

  private handleIncomplete(
    event: OpenAI.Responses.ResponseIncompleteEvent,
  ): OpenAI.ChatCompletionChunk | OpenAI.ChatCompletionChunk[] {
    const state = this.streamState;
    const resp = event.response;
    const finishChunk = this.makeChunk({}, "length");
    const chunks: OpenAI.ChatCompletionChunk[] = [finishChunk];

    if (state.includeUsage && resp.usage) {
      const usageChunk: OpenAI.ChatCompletionChunk = {
        id: state.id,
        object: "chat.completion.chunk",
        created: state.created,
        model: state.model,
        service_tier: state.serviceTier,
        choices: [],
        usage: this.convertUsage(resp.usage, state.webSearchCount),
      };
      chunks.push(usageChunk);
    }

    return chunks.length === 1 ? chunks[0] : chunks;
  }

  private handleFailed(): OpenAI.ChatCompletionChunk {
    return this.makeChunk({ content: "" }, "stop");
  }
}
