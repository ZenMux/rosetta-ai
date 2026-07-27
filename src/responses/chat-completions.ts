import type OpenAI from "openai";

type RespResponse = OpenAI.Responses.Response;
type RespStreamEvent = OpenAI.Responses.ResponseStreamEvent;

type ContentKind = "output_text" | "refusal";

interface ReasoningContext {
  type: "reasoning";
  itemId: string;
  outputIndex: number;
  summaryIndex: number;
  text: string;
}

interface MessageContext {
  type: "message";
  itemId: string;
  outputIndex: number;
  contentIndex: number;
  kind: ContentKind;
  outputText: string;
  refusal: string;
  annotations: Array<{
    type: "url_citation";
    url: string;
    title: string;
    start_index: number;
    end_index: number;
  }>;
}

interface FunctionCallContext {
  type: "function_call";
  itemId: string;
  outputIndex: number;
  callId: string;
  name: string;
  arguments: string;
}

interface CustomToolCallContext {
  type: "custom_tool_call";
  itemId: string;
  outputIndex: number;
  callId: string;
  name: string;
  input: string;
}

/**
 * The currently open output item. At most one is open at a time in a single
 * CC streaming response (choices[0]). When the item finishes — on finish_reason
 * or on transition to a different content kind / a new tool call id — the
 * associated `*.done` / `response.output_item.done` events are emitted and the
 * context is cleared.
 */
type ItemContext = ReasoningContext | MessageContext | FunctionCallContext | CustomToolCallContext;

interface StreamState {
  id: string;
  model: string;
  created: number;
  seq: number;
  outputIndex: number;
  contentIndex: number;
  annotationIndex: number;
  /** Running output items, mirrored so the terminal response carries `output`. */
  output: RespResponse["output"];
  /** The currently open output item, if any. */
  current: ItemContext | null;
  messageStarted: boolean;
  toolCallCount: number;
}

export class ResponsesToChatCompletionConverter {
  private streamState: StreamState;
  /**
   * The original Responses request params, stashed during `convertRequest` so
   * the streaming terminal event can echo request-scoped fields onto the
   * `response.completed` / `response.incomplete` payload. Only populated when
   * `convertRequest` ran before streaming (the normal gateway flow).
   */
  private requestParams?: OpenAI.Responses.ResponseCreateParams;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (Responses → CC, forward) ---

  convertRequest(
    params: OpenAI.Responses.ResponseCreateParams
  ): OpenAI.Chat.Completions.ChatCompletionCreateParams {
    this.requestParams = params;
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
    } else if (params.stream_options?.include_obfuscation != null) {
      // include_obfuscation is meaningful even for non-streaming Responses
      // requests (it governs how the gateway streams back to the client).
      // Forward it so it is not dropped when params.stream is falsy.
      result.stream_options = { include_obfuscation: params.stream_options.include_obfuscation };
    }

    return result;
  }

  // --- Response conversion (CC → Responses, backward) ---

  /**
   * Convert a ChatCompletion response back to a Responses Response.
   *
   * @param response The upstream ChatCompletion.
   * @param params Optional original Responses request params. When provided,
   *   request-scoped fields (instructions, parallel_tool_calls, tool_choice,
   *   tools, reasoning, temperature, text, etc.) are echoed back onto the
   *   response — matching the behavior of a gateway that round-trips the
   *   request through the Responses API. When omitted, these fields fall back
   *   to protocol defaults.
   */
  convertResponse(
    response: OpenAI.ChatCompletion,
    params?: OpenAI.Responses.ResponseCreateParams
  ): RespResponse {
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

    // Request-scoped fields echoed back onto the response. When `params` is
    // absent, fall back to protocol defaults so behavior matches a gateway
    // that does not round-trip the request.
    const echoed = this.echoRequestFields(params);

    return {
      id: response.id,
      object: "response",
      created_at: response.created,
      model: response.model,
      output,
      output_text: this.collectOutputText(output),
      status,
      error: null,
      incomplete_details: status === "incomplete" ? { reason: "max_output_tokens" } : null,
      instructions: echoed.instructions,
      metadata: {},
      temperature: echoed.temperature,
      top_p: echoed.top_p,
      max_output_tokens: echoed.max_output_tokens,
      previous_response_id: echoed.previous_response_id,
      parallel_tool_calls: echoed.parallel_tool_calls,
      tool_choice: echoed.tool_choice,
      tools: echoed.tools,
      text: echoed.text,
      reasoning: echoed.reasoning,
      truncation: echoed.truncation,
      user: undefined,
      // Non-standard echoes (top_logprobs / safety_identifier / service_tier)
      // are attached via cast; the Responses Response type does not declare them.
      ...((echoed as any).top_logprobs !== undefined
        ? { top_logprobs: (echoed as any).top_logprobs }
        : {}),
      ...(echoed.safety_identifier !== undefined
        ? { safety_identifier: echoed.safety_identifier }
        : {}),
      ...(echoed.service_tier !== undefined ? { service_tier: echoed.service_tier } : {}),
      ...(params?.background != null ? { background: params.background } : {}),
      ...(params?.prompt_cache_key !== undefined
        ? { prompt_cache_key: params.prompt_cache_key }
        : {}),
      ...(params?.prompt_cache_retention !== undefined
        ? { prompt_cache_retention: params.prompt_cache_retention }
        : {}),
      usage: response.usage
        ? {
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
            total_tokens: response.usage.total_tokens,
            input_tokens_details: {
              cached_tokens: response.usage.prompt_tokens_details?.cached_tokens ?? 0,
            },
            output_tokens_details: {
              reasoning_tokens: response.usage.completion_tokens_details?.reasoning_tokens ?? 0,
            },
          }
        : undefined,
    } as unknown as RespResponse;
  }

  /**
   * Compute the request-scoped field values to echo onto a converted response,
   * mirroring the legacy `toOpenAIRespResponse(params, result)` mapping. Returns
   * protocol defaults when `params` is absent.
   */
  private echoRequestFields(params?: OpenAI.Responses.ResponseCreateParams): {
    instructions: RespResponse["instructions"];
    temperature: RespResponse["temperature"];
    top_p: RespResponse["top_p"];
    max_output_tokens: RespResponse["max_output_tokens"];
    previous_response_id: RespResponse["previous_response_id"];
    parallel_tool_calls: RespResponse["parallel_tool_calls"];
    tool_choice: RespResponse["tool_choice"];
    tools: RespResponse["tools"];
    text: RespResponse["text"];
    reasoning: RespResponse["reasoning"];
    truncation: RespResponse["truncation"];
    top_logprobs?: number | null;
    safety_identifier?: string;
    service_tier?: RespResponse["service_tier"];
  } {
    if (!params) {
      return {
        instructions: null,
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
      };
    }

    const tools = (params.tools ?? [])
      .map(t => this.toRespTool(t))
      .filter(Boolean) as RespResponse["tools"];

    return {
      instructions: params.instructions ?? null,
      temperature: params.temperature ?? null,
      top_p: params.top_p ?? null,
      max_output_tokens: params.max_output_tokens ?? null,
      previous_response_id: params.previous_response_id ?? null,
      parallel_tool_calls: params.parallel_tool_calls ?? true,
      tool_choice: params.tool_choice ?? "auto",
      tools,
      text: params.text ?? { format: { type: "text" } },
      reasoning: params.reasoning ?? null,
      truncation: params.truncation ?? null,
      top_logprobs: (params as any).top_logprobs,
      safety_identifier: params.safety_identifier,
      // service_tier is normally returned by the upstream model; echo the
      // request's value as a fallback (the legacy converter did the same).
      service_tier: params.service_tier,
    };
  }

  /**
   * Map a Responses request tool back to a Responses response tool, preserving
   * function / web_search / custom tools. Mirrors the legacy `toRespTool`.
   */
  private toRespTool(tool: OpenAI.Responses.Tool): RespResponse["tools"][number] | undefined {
    const t = tool as any;
    if (t.type === "function") {
      return {
        type: "function",
        name: t.name,
        description: t.description,
        parameters: !t.parameters
          ? { additionalProperties: false, type: "object", properties: {}, required: [] }
          : {
              additionalProperties: t.parameters.additionalProperties ?? false,
              type: t.parameters.type ?? "object",
              properties: t.parameters.properties ?? {},
              required: t.parameters.required ?? [],
            },
        strict: t.strict,
      } as any;
    }
    if (
      t.type === "web_search" ||
      t.type === "web_search_preview" ||
      t.type === "web_search_2025_08_26" ||
      t.type === "web_search_preview_2025_03_11" ||
      t.type === "custom"
    ) {
      return t;
    }
    return undefined;
  }

  /**
   * Compute `output_text` — the concatenation of all `output_text` content
   * parts across the response's message output items, matching the Responses
   * API convention. Returns "" when there are no text parts (e.g. tool-only
   * or refusal-only responses), matching the legacy `toOpenAIRespResponse`
   * default of `output_text: ""`.
   */
  private collectOutputText(output: RespResponse["output"]): string {
    let text = "";
    for (const item of output ?? []) {
      if ((item as any).type !== "message") continue;
      const content = (item as any).content;
      if (!Array.isArray(content)) continue;
      for (const part of content) {
        if (part?.type === "output_text" && typeof part.text === "string") {
          text += part.text;
        }
      }
    }
    return text;
  }

  // --- Stream conversion (CC → Responses, backward) ---

  async *convertStream(
    stream: AsyncIterable<OpenAI.ChatCompletionChunk>
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
        events.push(...this.emitCompleted(chunk.usage));
      }
      return events;
    }

    const delta = choice.delta;

    // Reasoning delta — open a reasoning item if not already open.
    const reasoning = (delta as any)?.reasoning || (delta as any)?.reasoning_content;
    if (reasoning) {
      if (!state.current || state.current.type !== "reasoning") {
        // Close any other open item before starting reasoning.
        events.push(...this.closeCurrent());
        const itemId = `rs_${this.generateId()}`;
        state.current = {
          type: "reasoning",
          itemId,
          outputIndex: state.outputIndex,
          summaryIndex: 0,
          text: "",
        };
        events.push({
          type: "response.output_item.added",
          item: { type: "reasoning", id: itemId, summary: [] },
          output_index: state.outputIndex,
          sequence_number: state.seq++,
        } as RespStreamEvent);
        events.push({
          type: "response.reasoning_summary_part.added",
          item_id: itemId,
          output_index: state.outputIndex,
          summary_index: 0,
          part: { type: "summary_text", text: "" },
          sequence_number: state.seq++,
        } as RespStreamEvent);
      }
      state.current.text += reasoning;
      events.push({
        type: "response.reasoning_summary_text.delta",
        item_id: state.current.itemId,
        output_index: state.outputIndex,
        summary_index: 0,
        delta: reasoning,
        sequence_number: state.seq++,
      } as RespStreamEvent);
    }

    // Tool calls — function or custom.
    if (delta?.tool_calls) {
      for (const tc of delta.tool_calls) {
        const isCustom = (tc as any).type === "custom" || (tc as any).custom;
        // A new tool call begins when an id + name arrive.
        if (tc.id && ((tc.function?.name && !isCustom) || ((tc as any).custom?.name && isCustom))) {
          const callId = tc.id;
          const name = isCustom ? (tc as any).custom.name : tc.function!.name;
          // Close any open reasoning/message item before starting a tool call.
          if (
            state.current &&
            state.current.type !== "function_call" &&
            state.current.type !== "custom_tool_call"
          ) {
            events.push(...this.closeCurrent());
          }
          // If a tool call of a different type/id is already open, close it.
          if (
            state.current &&
            ((state.current.type === "function_call" && isCustom) ||
              (state.current.type === "custom_tool_call" && !isCustom) ||
              (state.current.type !== "custom_tool_call" && state.current.type !== "function_call"))
          ) {
            events.push(...this.closeCurrent());
          }
          if (
            state.current &&
            (state.current.type === "function_call" || state.current.type === "custom_tool_call") &&
            state.current.callId !== callId
          ) {
            events.push(...this.closeCurrent());
          }

          state.toolCallCount++;
          // Tool calls live at output_index = outputIndex + toolCallCount - 1,
          // matching the legacy converter's output_index math.
          const outputIndex = state.outputIndex + state.toolCallCount - 1;
          if (isCustom) {
            const itemId = `ct_${this.generateId()}`;
            state.current = {
              type: "custom_tool_call",
              itemId,
              outputIndex,
              callId,
              name,
              input: "",
            };
            events.push({
              type: "response.output_item.added",
              item: {
                type: "custom_tool_call",
                id: itemId,
                call_id: callId,
                name,
                input: "",
                status: "in_progress",
              } as any,
              output_index: outputIndex,
              sequence_number: state.seq++,
            } as RespStreamEvent);
          } else {
            const itemId = `fc_${this.generateId()}`;
            state.current = {
              type: "function_call",
              itemId,
              outputIndex,
              callId,
              name,
              arguments: "",
            };
            events.push({
              type: "response.output_item.added",
              item: {
                type: "function_call",
                id: itemId,
                call_id: callId,
                name,
                arguments: "",
                status: "in_progress",
              },
              output_index: outputIndex,
              sequence_number: state.seq++,
            } as RespStreamEvent);
          }
        }

        // Arguments/input delta.
        if (state.current?.type === "function_call" && tc.function?.arguments) {
          state.current.arguments += tc.function.arguments;
          events.push({
            type: "response.function_call_arguments.delta",
            item_id: state.current.itemId,
            output_index: state.current.outputIndex,
            delta: tc.function.arguments,
            sequence_number: state.seq++,
          } as RespStreamEvent);
        } else if (state.current?.type === "custom_tool_call") {
          const inputDelta =
            (tc as any).custom?.input ?? (tc.function?.arguments as string | undefined) ?? "";
          if (inputDelta) {
            state.current.input += inputDelta;
            events.push({
              type: "response.custom_tool_call_input.delta",
              item_id: state.current.itemId,
              output_index: state.current.outputIndex,
              delta: inputDelta,
              sequence_number: state.seq++,
            } as RespStreamEvent);
          }
        }
      }
    }

    // Refusal delta.
    const refusalDelta = (delta as any)?.refusal;
    if (refusalDelta) {
      if (!state.current || state.current.type !== "message" || state.current.kind !== "refusal") {
        events.push(...this.closeCurrent());
        this.openMessage(state, "refusal", events);
      }
      const msgCtx = state.current as MessageContext;
      msgCtx.refusal += refusalDelta;
      events.push({
        type: "response.refusal.delta",
        item_id: msgCtx.itemId,
        output_index: msgCtx.outputIndex,
        content_index: msgCtx.contentIndex,
        delta: refusalDelta,
        sequence_number: state.seq++,
      } as RespStreamEvent);
    }

    // Text content delta.
    if (delta?.content && delta.content !== "") {
      if (
        !state.current ||
        state.current.type !== "message" ||
        state.current.kind !== "output_text"
      ) {
        events.push(...this.closeCurrent());
        this.openMessage(state, "output_text", events);
      }
      (state.current as MessageContext).outputText += delta.content;
      events.push({
        type: "response.output_text.delta",
        item_id: (state.current as MessageContext).itemId,
        output_index: (state.current as MessageContext).outputIndex,
        content_index: (state.current as MessageContext).contentIndex,
        delta: delta.content,
        logprobs: [],
        sequence_number: state.seq++,
      } as RespStreamEvent);
    }

    // Annotations on the message delta.
    const annotations = (delta as any)?.annotations;
    if (annotations && state.current?.type === "message") {
      for (const a of annotations) {
        if (a?.type !== "url_citation" || !a.url_citation) continue;
        state.annotationIndex++;
        const annotation = {
          type: "url_citation" as const,
          url: a.url_citation.url,
          title: a.url_citation.title,
          start_index: a.url_citation.start_index,
          end_index: a.url_citation.end_index,
        };
        state.current.annotations.push(annotation);
        events.push({
          type: "response.output_text.annotation.added",
          item_id: state.current.itemId,
          output_index: state.current.outputIndex,
          content_index: state.current.contentIndex,
          annotation_index: state.annotationIndex,
          annotation,
          sequence_number: state.seq++,
        } as RespStreamEvent);
      }
    }

    // Finish reason — close the open item, then emit the terminal event.
    if (choice.finish_reason) {
      events.push(...this.closeCurrent());
      const status = this.finishReasonToStatus(choice.finish_reason);
      if (chunk.usage) {
        events.push(...this.emitCompleted(chunk.usage, status));
      } else {
        events.push(...this.emitCompleted(undefined, status));
      }
    }

    return events;
  }

  /** Open a message output item and its first content part. */
  private openMessage(state: StreamState, kind: ContentKind, events: RespStreamEvent[]): void {
    state.messageStarted = true;
    const itemId = `msg_${this.generateId()}`;
    const msgOutputIndex = state.outputIndex + state.toolCallCount;
    state.current = {
      type: "message",
      itemId,
      outputIndex: msgOutputIndex,
      contentIndex: state.contentIndex,
      kind,
      outputText: "",
      refusal: "",
      annotations: [],
    };
    events.push({
      type: "response.output_item.added",
      item: {
        type: "message",
        id: itemId,
        role: "assistant",
        status: "in_progress",
        content: [],
      },
      output_index: msgOutputIndex,
      sequence_number: state.seq++,
    } as RespStreamEvent);
    events.push({
      type: "response.content_part.added",
      item_id: itemId,
      output_index: msgOutputIndex,
      content_index: state.contentIndex,
      part:
        kind === "output_text"
          ? { type: "output_text", text: "", annotations: [] }
          : { type: "refusal", refusal: "" },
      sequence_number: state.seq++,
    } as RespStreamEvent);
  }

  /**
   * Close the currently open output item, emitting the `*.done` and
   * `response.output_item.done` events. Mirrors the legacy converter's
   * stopReasoning / stopOutputTextRefusal / stopFunctionCallArguments /
   * stopCustomToolCallInput helpers.
   */
  private closeCurrent(): RespStreamEvent[] {
    const state = this.streamState;
    const ctx = state.current;
    if (!ctx) return [];
    const events: RespStreamEvent[] = [];

    if (ctx.type === "reasoning") {
      events.push({
        type: "response.reasoning_summary_text.done",
        item_id: ctx.itemId,
        output_index: ctx.outputIndex,
        summary_index: ctx.summaryIndex,
        text: ctx.text,
        sequence_number: state.seq++,
      } as RespStreamEvent);
      events.push({
        type: "response.reasoning_summary_part.done",
        item_id: ctx.itemId,
        output_index: ctx.outputIndex,
        summary_index: ctx.summaryIndex,
        part: { type: "summary_text", text: ctx.text },
        sequence_number: state.seq++,
      } as RespStreamEvent);
      const item = {
        id: ctx.itemId,
        type: "reasoning" as const,
        summary: [{ type: "summary_text" as const, text: ctx.text }],
      };
      state.output.push(item);
      events.push({
        type: "response.output_item.done",
        item,
        output_index: ctx.outputIndex,
        sequence_number: state.seq++,
      } as RespStreamEvent);
      state.outputIndex++;
    } else if (ctx.type === "message") {
      const status: "completed" | "incomplete" = "completed";
      const part =
        ctx.kind === "output_text"
          ? {
              type: "output_text" as const,
              annotations: ctx.annotations,
              logprobs: [] as any[],
              text: ctx.outputText,
            }
          : { type: "refusal" as const, refusal: ctx.refusal };
      events.push(
        ctx.kind === "output_text"
          ? ({
              type: "response.output_text.done",
              item_id: ctx.itemId,
              output_index: ctx.outputIndex,
              content_index: ctx.contentIndex,
              text: ctx.outputText,
              logprobs: [],
              sequence_number: state.seq++,
            } as RespStreamEvent)
          : ({
              type: "response.refusal.done",
              item_id: ctx.itemId,
              output_index: ctx.outputIndex,
              content_index: ctx.contentIndex,
              refusal: ctx.refusal,
              sequence_number: state.seq++,
            } as RespStreamEvent)
      );
      events.push({
        type: "response.content_part.done",
        item_id: ctx.itemId,
        output_index: ctx.outputIndex,
        content_index: ctx.contentIndex,
        part,
        sequence_number: state.seq++,
      } as RespStreamEvent);
      const item = {
        id: ctx.itemId,
        type: "message" as const,
        role: "assistant" as const,
        status,
        content: [part],
      };
      state.output.push(item);
      events.push({
        type: "response.output_item.done",
        item,
        output_index: ctx.outputIndex,
        sequence_number: state.seq++,
      } as RespStreamEvent);
      state.outputIndex++;
    } else if (ctx.type === "function_call") {
      events.push({
        type: "response.function_call_arguments.done",
        item_id: ctx.itemId,
        output_index: ctx.outputIndex,
        name: ctx.name,
        arguments: ctx.arguments,
        sequence_number: state.seq++,
      } as RespStreamEvent);
      const item = {
        id: ctx.itemId,
        type: "function_call" as const,
        status: "completed" as const,
        call_id: ctx.callId,
        name: ctx.name,
        arguments: ctx.arguments,
      };
      state.output.push(item);
      events.push({
        type: "response.output_item.done",
        item,
        output_index: ctx.outputIndex,
        sequence_number: state.seq++,
      } as RespStreamEvent);
      state.outputIndex++;
    } else if (ctx.type === "custom_tool_call") {
      events.push({
        type: "response.custom_tool_call_input.done",
        item_id: ctx.itemId,
        output_index: ctx.outputIndex,
        input: ctx.input,
        sequence_number: state.seq++,
      } as RespStreamEvent);
      const item = {
        id: ctx.itemId,
        type: "custom_tool_call" as const,
        call_id: ctx.callId,
        name: ctx.name,
        input: ctx.input,
      };
      state.output.push(item);
      events.push({
        type: "response.output_item.done",
        item,
        output_index: ctx.outputIndex,
        sequence_number: state.seq++,
      } as RespStreamEvent);
      state.outputIndex++;
    }

    state.current = null;
    return events;
  }

  // --- Private: request helpers ---

  private convertInput(
    messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
    input: OpenAI.Responses.ResponseCreateParams["input"]
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
      if ("role" in item && "content" in item && !("type" in item)) {
        // EasyInputMessage: { role, content } with no `type` field.
        // content may be a string or an array of input parts.
        flushToolCalls();
        const role = item.role;
        if (role === "user" || role === "system" || role === "developer" || role === "assistant") {
          const content =
            typeof item.content === "string"
              ? item.content
              : this.convertInputContent(item.content);
          messages.push({
            role,
            content,
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
          content: this.toToolContent(typed.output),
        });
      } else if (typed.type === "custom_tool_call_output") {
        // A custom tool's output. There is no CC "custom tool" message role;
        // surface it as a tool message keyed by the call id, matching the
        // legacy converter's behavior.
        flushToolCalls();
        messages.push({
          role: "tool",
          tool_call_id: typed.call_id,
          content: this.toToolContent(typed.output),
        } as OpenAI.Chat.Completions.ChatCompletionMessageParam);
      }
    }

    flushToolCalls();
  }

  private convertInputContent(
    content: any[]
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
          image_url: {
            url: part.image_url || part.file_id || "",
            // "original" is not a valid CC image_url.detail; omit it to use the default.
            ...(part.detail && part.detail !== "original" ? { detail: part.detail } : {}),
          },
        };
      }
      if (part.type === "input_file") {
        return {
          type: "file",
          file: {
            file_data: part.file_data || part.file_url || "",
            filename: part.filename,
            file_id: part.file_id || undefined,
          },
        };
      }
      if (part.type === "input_audio") {
        return {
          type: "input_audio",
          input_audio: {
            data: part.input_audio?.data,
            format: part.input_audio?.format,
          },
        } as unknown as OpenAI.Chat.Completions.ChatCompletionContentPart;
      }
      // Non-standard extension: ZenMux supports input_video on Responses user
      // input. Map to a video_url content part (not in the SDK types).
      if (part.type === "input_video") {
        const v = part.input_video;
        const url = v?.data ? `data:video/${v?.format ?? "mp4"};base64,${v.data}` : (v?.url ?? "");
        return {
          type: "video_url",
          video_url: { url: url || null },
        } as unknown as OpenAI.Chat.Completions.ChatCompletionContentPart;
      }
      return { type: "text", text: "" };
    });
  }

  /**
   * Normalize a Responses tool-call output payload to a CC tool message
   * content. A string passes through; an array of input parts is reduced to
   * its `input_text` items as CC text parts. Mirrors the legacy `toToolContent`.
   */
  private toToolContent(
    output: string | Array<{ type: string; text?: string } | Record<string, unknown>>
  ): string | OpenAI.Chat.Completions.ChatCompletionToolMessageParam["content"] {
    if (typeof output === "string") return output;
    if (!Array.isArray(output)) return "";
    const parts = output
      .filter((item: any) => item.type === "input_text")
      .map((item: any) => ({ type: "text" as const, text: item.text }));
    return parts.length ? parts : "";
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
      } else if (tt.type === "custom") {
        // Responses custom tool → CC custom tool.
        const custom: any = { name: tt.name };
        if (tt.description != null) custom.description = tt.description;
        if (tt.format != null) {
          custom.format =
            tt.format.type === "text"
              ? { type: "text" }
              : {
                  type: "grammar",
                  grammar: {
                    definition: tt.format.definition,
                    syntax: tt.format.syntax,
                  },
                };
        }
        ccTools.push({ type: "custom", custom });
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
    choice: OpenAI.Responses.ResponseCreateParams["tool_choice"]
  ): OpenAI.Chat.Completions.ChatCompletionToolChoiceOption {
    if (typeof choice === "string") {
      if (choice === "auto" || choice === "none" || choice === "required") return choice;
      return "auto";
    }
    if (typeof choice === "object" && choice !== null) {
      const c = choice as any;
      if (c.type === "function" && c.name) {
        return { type: "function", function: { name: c.name } };
      }
      if (c.type === "custom" && c.name) {
        return { type: "custom", custom: { name: c.name } };
      }
      if (c.type === "allowed_tools") {
        return {
          type: "allowed_tools",
          allowed_tools: {
            mode: c.mode,
            tools: c.tools,
          },
        };
      }
    }
    return "auto";
  }

  private convertTextFormat(
    format: any
  ): OpenAI.Chat.Completions.ChatCompletionCreateParams["response_format"] {
    if (format.type === "json_schema") {
      const json_schema: any = {
        name: format.name || "response",
        schema: format.schema,
        strict: format.strict,
      };
      // description is optional but supported by the CC json_schema format.
      if (format.description != null) json_schema.description = format.description;
      return {
        type: "json_schema",
        json_schema,
      };
    }
    if (format.type === "json_object") {
      return { type: "json_object" };
    }
    return { type: "text" };
  }

  // --- Private: response helpers ---

  private finishReasonToStatus(reason: string | null | undefined): RespResponse["status"] {
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
      annotationIndex: -1,
      output: [],
      current: null,
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
    status?: RespResponse["status"]
  ): RespStreamEvent[] {
    const state = this.streamState;
    const finalStatus = status ?? "completed";
    const resp = this.makeSkeletonResponse();
    resp.status = finalStatus;
    // Carry the output items accumulated during the stream.
    resp.output = [...state.output];
    resp.output_text = this.collectOutputText(resp.output);

    // Echo request-scoped fields onto the terminal response when available.
    const echoed = this.echoRequestFields(this.requestParams);
    resp.instructions = echoed.instructions;
    resp.temperature = echoed.temperature;
    resp.top_p = echoed.top_p;
    resp.max_output_tokens = echoed.max_output_tokens;
    resp.previous_response_id = echoed.previous_response_id;
    resp.parallel_tool_calls = echoed.parallel_tool_calls;
    resp.tool_choice = echoed.tool_choice;
    resp.tools = echoed.tools;
    resp.text = echoed.text;
    resp.reasoning = echoed.reasoning;
    resp.truncation = echoed.truncation;
    if ((echoed as any).top_logprobs !== undefined)
      (resp as any).top_logprobs = (echoed as any).top_logprobs;
    if (echoed.safety_identifier !== undefined)
      (resp as any).safety_identifier = echoed.safety_identifier;
    if (echoed.service_tier !== undefined) (resp as any).service_tier = echoed.service_tier;
    if (this.requestParams?.background != null)
      (resp as any).background = this.requestParams.background;
    if (this.requestParams?.prompt_cache_key !== undefined)
      (resp as any).prompt_cache_key = this.requestParams.prompt_cache_key;
    if (this.requestParams?.prompt_cache_retention !== undefined)
      (resp as any).prompt_cache_retention = this.requestParams.prompt_cache_retention;

    if (finalStatus === "incomplete") {
      resp.incomplete_details = { reason: "max_output_tokens" };
    }

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

    if (finalStatus === "failed") {
      return [
        {
          type: "response.failed",
          response: resp,
          sequence_number: state.seq++,
        } as RespStreamEvent,
      ];
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
