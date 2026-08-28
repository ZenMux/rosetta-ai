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
  toolCallIndexes: Map<string, number>;
  webSearchCount: number;
  annotations: OpenAI.ChatCompletionMessage.Annotation[];
  includeUsage: boolean;
  includeLogprobs: boolean;
}

export class ChatCompletionToResponsesConverter {
  private streamState: StreamState;
  private legacyFunctionMode = false;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (CC → Responses, forward) ---

  convertRequest(params: CCParams): OpenAI.Responses.ResponseCreateParams {
    this.streamState = this.createStreamState();
    this.streamState.includeUsage = params.stream_options?.include_usage ?? false;
    this.streamState.includeLogprobs = params.top_logprobs != null;
    this.legacyFunctionMode =
      params.function_call != null || (params.tools == null && params.functions != null);

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
    if (params.store != null) {
      result.store = params.store;
    }
    if (params.safety_identifier != null) {
      result.safety_identifier = params.safety_identifier;
    }
    if (params.user != null) {
      result.user = params.user;
    }
    const tools = this.convertTools(params);
    if (tools.length > 0) {
      result.tools = tools;
    }
    if (params.tool_choice != null) {
      result.tool_choice = this.convertToolChoice(params.tool_choice);
    } else if (params.function_call != null) {
      result.tool_choice = this.convertLegacyFunctionChoice(params.function_call);
    }
    if (params.response_format) {
      result.text = this.convertResponseFormat(params.response_format);
    }
    if (params.verbosity != null) {
      result.text = {
        ...(result.text ?? {}),
        verbosity: params.verbosity,
      };
    }
    if (params.stream != null) {
      (result as any).stream = params.stream;
    }
    if (params.stream_options?.include_obfuscation != null) {
      result.stream_options = {
        include_obfuscation: params.stream_options.include_obfuscation,
      };
    }

    return result;
  }

  // --- Response conversion (Responses → CC, backward) ---

  convertResponse(response: RespResponse): OpenAI.ChatCompletion {
    const choices: OpenAI.ChatCompletion.Choice[] = [];
    const toolCalls: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[] = [];
    const messages: OpenAI.Responses.ResponseOutputMessage[] = [];
    const reasoningItems = response.output.filter(
      (item): item is OpenAI.Responses.ResponseReasoningItem => item.type === "reasoning"
    );
    let webSearchCount = 0;

    for (const item of response.output) {
      if (item.type === "message") {
        messages.push(item as OpenAI.Responses.ResponseOutputMessage);
      } else if (item.type === "function_call") {
        const fc = item as OpenAI.Responses.ResponseFunctionToolCall;
        toolCalls.push({
          id: fc.call_id,
          type: "function",
          function: { name: fc.name, arguments: fc.arguments },
        });
      } else if (item.type === "custom_tool_call") {
        const custom = item as OpenAI.Responses.ResponseCustomToolCall;
        toolCalls.push({
          id: custom.call_id,
          type: "custom",
          custom: { name: custom.name, input: custom.input },
        });
      } else if (
        item.type === "web_search_call" &&
        item.status === "completed" &&
        item.action.type === "search"
      ) {
        webSearchCount++;
      }
    }

    if (messages.length > 0) {
      const choice = this.convertOutputMessages(
        messages,
        reasoningItems,
        response.status,
        response.incomplete_details
      );
      this.applyToolCalls(choice, toolCalls, response.status);
      choices.push(choice);
    } else if (toolCalls.length > 0) {
      const choice: OpenAI.ChatCompletion.Choice = {
        index: 0,
        finish_reason: this.mapStatus(response.status, response.incomplete_details),
        message: {
          role: "assistant",
          content: null,
          refusal: null,
        },
        logprobs: null,
      };
      this.applyToolCalls(choice, toolCalls, response.status);
      choices.push(choice);
    } else if (reasoningItems.length > 0) {
      choices.push({
        index: 0,
        finish_reason: this.mapStatus(response.status, response.incomplete_details),
        message: {
          role: "assistant",
          content: null,
          refusal: null,
          ...this.convertReasoning(reasoningItems),
        },
        logprobs: null,
      });
    } else {
      choices.push({
        index: 0,
        finish_reason: this.mapStatus(response.status, response.incomplete_details),
        message: { role: "assistant", content: null, refusal: null },
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
      ...(response.usage ? { usage: this.convertUsage(response.usage, webSearchCount) } : {}),
    };
  }

  private convertOutputMessages(
    messages: OpenAI.Responses.ResponseOutputMessage[],
    reasoningItems: OpenAI.Responses.ResponseReasoningItem[],
    status: RespResponse["status"],
    incompleteDetails?: RespResponse["incomplete_details"]
  ): OpenAI.ChatCompletion.Choice {
    const textParts: string[] = [];
    const refusalParts: string[] = [];
    const annotations: OpenAI.ChatCompletionMessage.Annotation[] = [];
    const logprobs: OpenAI.Chat.Completions.ChatCompletionTokenLogprob[] = [];
    let hasLogprobs = false;
    let textOffset = 0;

    for (const message of messages) {
      for (const part of message.content) {
        if (part.type === "refusal") {
          refusalParts.push(part.refusal);
          continue;
        }

        for (const annotation of part.annotations ?? []) {
          if (annotation.type !== "url_citation") continue;
          annotations.push({
            type: "url_citation",
            url_citation: {
              title: annotation.title,
              url: annotation.url,
              start_index: textOffset + annotation.start_index,
              end_index: textOffset + annotation.end_index,
            },
          });
        }
        if (part.logprobs != null) {
          hasLogprobs = true;
          logprobs.push(...part.logprobs.map(logprob => this.convertTokenLogprob(logprob)));
        }
        textParts.push(part.text);
        textOffset += part.text.length;
      }
    }

    return {
      index: 0,
      finish_reason: this.mapStatus(status, incompleteDetails),
      message: {
        role: "assistant",
        content: textParts.length > 0 ? textParts.join("") : null,
        refusal: refusalParts.length > 0 ? refusalParts.join("") : null,
        annotations,
        ...this.convertReasoning(reasoningItems),
      },
      logprobs: hasLogprobs
        ? {
            content: logprobs,
            refusal: null,
          }
        : null,
    };
  }

  private convertReasoning(
    reasoningItems: OpenAI.Responses.ResponseReasoningItem[]
  ): Record<string, any> {
    const summaries: string[] = [];
    const details: any[] = [];

    for (const reasoning of reasoningItems) {
      if (!reasoning.summary || reasoning.summary.length === 0) {
        continue;
      }

      for (const part of reasoning.summary) {
        summaries.push(part.text);
        details.push({
          index: "0",
          format: "openai-responses-v1",
          type: "reasoning.summary",
          summary: part.text,
        });
      }

      if (reasoning.encrypted_content) {
        details.push({
          id: reasoning.id,
          index: "0",
          format: "openai-responses-v1",
          type: "reasoning.encrypted",
          data: reasoning.encrypted_content,
        });
      }
    }

    if (summaries.length === 0) {
      return {};
    }
    return { reasoning: summaries.join(""), reasoning_details: details };
  }

  // --- Stream conversion (Responses → CC, backward) ---

  async *convertStream(
    stream: AsyncIterable<RespStreamEvent>
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
    event: RespStreamEvent
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
      case "response.custom_tool_call_input.delta":
        return this.handleCustomToolCallDelta(event);
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
        return null;
      case "response.output_item.done":
        this.handleOutputItemDone(event as OpenAI.Responses.ResponseOutputItemDoneEvent);
        return null;
      case "response.output_text.annotation.added":
        this.recordStreamAnnotation(event.annotation);
        return null;
      default:
        return null;
    }
  }

  // --- Private: request helpers ---

  private convertMessages(messages: CCParams["messages"]): RespInputItem[] {
    const input: RespInputItem[] = [];
    const toolCallTypes = new Map<string, "function" | "custom">();
    const legacyCallIds = new Map<string, string[]>();

    for (const [messageIndex, msg] of messages.entries()) {
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
            content: this.convertUserContent(msg.content),
          });
        }
      } else if (msg.role === "assistant") {
        const toolCalls = msg.tool_calls ?? [];
        const hasToolCalls = toolCalls.length > 0;
        const hasLegacyFunctionCall = msg.function_call != null;
        if (
          (!hasToolCalls && !hasLegacyFunctionCall) ||
          msg.content != null ||
          msg.refusal != null
        ) {
          input.push(this.convertAssistantMessage(msg));
        }

        if (hasToolCalls) {
          for (const tc of toolCalls) {
            if (tc.type === "function") {
              toolCallTypes.set(tc.id, "function");
              input.push({
                type: "function_call",
                name: tc.function.name,
                call_id: tc.id,
                arguments: tc.function.arguments,
              });
            } else if (tc.type === "custom") {
              toolCallTypes.set(tc.id, "custom");
              input.push({
                type: "custom_tool_call",
                name: tc.custom.name,
                call_id: tc.id,
                input: tc.custom.input,
              });
            }
          }
        }

        if (msg.function_call != null) {
          const callId = `call_legacy_${messageIndex}`;
          const pendingIds = legacyCallIds.get(msg.function_call.name) ?? [];
          pendingIds.push(callId);
          legacyCallIds.set(msg.function_call.name, pendingIds);
          input.push({
            type: "function_call",
            name: msg.function_call.name,
            call_id: callId,
            arguments: msg.function_call.arguments,
          });
        }
      } else if (msg.role === "tool") {
        const output =
          typeof msg.content === "string"
            ? msg.content
            : msg.content.map(p => ({ type: "input_text" as const, text: p.text }));
        input.push({
          type:
            toolCallTypes.get(msg.tool_call_id) === "custom"
              ? "custom_tool_call_output"
              : "function_call_output",
          call_id: msg.tool_call_id,
          output,
        } as RespInputItem);
      } else if (msg.role === "function") {
        const pendingIds = legacyCallIds.get(msg.name) ?? [];
        const callId = pendingIds.shift() ?? `call_legacy_${messageIndex}`;
        input.push({
          type: "function_call_output",
          call_id: callId,
          output: msg.content ?? "",
        });
      }
    }

    return input;
  }

  private convertAssistantMessage(
    message: OpenAI.Chat.Completions.ChatCompletionAssistantMessageParam
  ): OpenAI.Responses.EasyInputMessage {
    const content = message.content;
    if (typeof content === "string") {
      return {
        role: "assistant",
        type: "message",
        content,
      };
    }

    if (Array.isArray(content)) {
      const parts = content.map(part =>
        part.type === "refusal"
          ? { type: "refusal" as const, refusal: part.refusal }
          : { type: "output_text" as const, text: part.text }
      );
      if (message.refusal != null && !content.some(part => part.type === "refusal")) {
        parts.push({ type: "refusal", refusal: message.refusal });
      }
      return {
        role: "assistant",
        type: "message",
        // The Responses API requires assistant history blocks to use output_text/refusal,
        // while the current SDK types only expose ResponseInputContent for EasyInputMessage.
        content: parts as any,
      };
    }

    if (message.refusal != null) {
      return {
        role: "assistant",
        type: "message",
        // The Responses API models refusal as its own assistant content part.
        content: [{ type: "refusal", refusal: message.refusal }] as any,
      };
    }

    return {
      role: "assistant",
      type: "message",
      content: "",
    };
  }

  private convertUserContent(
    parts: Exclude<OpenAI.Chat.Completions.ChatCompletionUserMessageParam["content"], string>
  ): OpenAI.Responses.ResponseInputContent[] {
    const content: OpenAI.Responses.ResponseInputContent[] = [];
    for (const part of parts) {
      if (part.type === "text") {
        content.push({ type: "input_text", text: part.text });
      } else if (part.type === "image_url") {
        content.push({
          type: "input_image",
          image_url: part.image_url.url,
          detail: part.image_url.detail ?? "auto",
        });
      } else if (part.type === "file") {
        content.push({
          type: "input_file",
          file_data: part.file.file_data,
          file_id: part.file.file_id,
          filename: part.file.filename,
        });
      }
    }
    return content;
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
        } else if (t.type === "custom") {
          tools.push({
            type: "custom",
            name: t.custom.name,
            description: t.custom.description,
            ...(t.custom.format ? { format: this.convertCustomToolFormat(t.custom.format) } : {}),
          });
        }
      }
    } else if (params.functions) {
      for (const fn of params.functions) {
        tools.push({
          type: "function",
          name: fn.name,
          description: fn.description,
          strict: null,
          parameters: fn.parameters ?? null,
        });
      }
    }

    if (params.web_search_options != null) {
      const options = params.web_search_options;
      tools.push({
        type: "web_search",
        ...(options.search_context_size != null
          ? { search_context_size: options.search_context_size }
          : {}),
        ...(options.user_location != null
          ? {
              user_location: {
                type: "approximate",
                ...options.user_location.approximate,
              },
            }
          : {}),
      });
    }

    return tools;
  }

  private convertToolChoice(
    choice: CCParams["tool_choice"]
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
      if (c.type === "custom" && c.custom?.name) {
        return { type: "custom", name: c.custom.name };
      }
      if (c.type === "allowed_tools") {
        return {
          type: "allowed_tools",
          mode: c.allowed_tools.mode,
          tools: c.allowed_tools.tools.map(tool => this.convertAllowedTool(tool)),
        };
      }
    }
    return "auto";
  }

  private convertLegacyFunctionChoice(
    choice: NonNullable<CCParams["function_call"]>
  ): OpenAI.Responses.ResponseCreateParams["tool_choice"] {
    if (choice === "auto" || choice === "none") {
      return choice;
    }
    return { type: "function", name: choice.name };
  }

  private convertAllowedTool(tool: Record<string, unknown>): Record<string, unknown> {
    if (tool.type === "function" && tool.function && typeof tool.function === "object") {
      const fn = tool.function as { name?: unknown };
      if (typeof fn.name === "string") {
        return { type: "function", name: fn.name };
      }
    }
    if (tool.type === "custom" && tool.custom && typeof tool.custom === "object") {
      const custom = tool.custom as { name?: unknown };
      if (typeof custom.name === "string") {
        return { type: "custom", name: custom.name };
      }
    }
    return tool;
  }

  private convertCustomToolFormat(
    format: OpenAI.Chat.Completions.ChatCompletionCustomTool.Custom["format"]
  ): OpenAI.Responses.CustomTool["format"] {
    if (format?.type === "grammar") {
      return {
        type: "grammar",
        definition: format.grammar.definition,
        syntax: format.grammar.syntax,
      };
    }
    return { type: "text" };
  }

  private convertResponseFormat(
    format: CCParams["response_format"]
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
          ...(format.json_schema.description != null
            ? { description: format.json_schema.description }
            : {}),
          ...(format.json_schema.strict != null ? { strict: format.json_schema.strict } : {}),
        },
      };
    }
    return undefined;
  }

  // --- Private: response helpers ---

  private mapStatus(
    status: RespResponse["status"],
    incompleteDetails?: RespResponse["incomplete_details"]
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

  private applyToolCalls(
    choice: OpenAI.ChatCompletion.Choice,
    toolCalls: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[],
    responseStatus: RespResponse["status"]
  ): void {
    if (toolCalls.length === 0) return;

    const firstCall = toolCalls[0];
    if (this.legacyFunctionMode && firstCall.type === "function") {
      choice.message.function_call = { ...firstCall.function };
      if (responseStatus === "completed") {
        choice.finish_reason = "function_call";
      }
      return;
    }

    choice.message.tool_calls = toolCalls;
    if (responseStatus === "completed") {
      choice.finish_reason = "tool_calls";
    }
  }

  private convertUsage(
    usage: OpenAI.Responses.ResponseUsage,
    webSearchCount = 0
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

  private convertTokenLogprob(logprob: {
    token: string;
    logprob: number;
    bytes?: number[] | null;
    top_logprobs?: Array<{
      token?: string;
      logprob?: number;
      bytes?: number[] | null;
    }>;
  }): OpenAI.Chat.Completions.ChatCompletionTokenLogprob {
    return {
      token: logprob.token,
      logprob: logprob.logprob,
      bytes: logprob.bytes ?? null,
      top_logprobs: (logprob.top_logprobs ?? [])
        .filter(
          (top): top is { token: string; logprob: number; bytes?: number[] | null } =>
            typeof top.token === "string" && typeof top.logprob === "number"
        )
        .map(top => ({
          token: top.token,
          logprob: top.logprob,
          bytes: top.bytes ?? null,
        })),
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
      toolCallIndexes: new Map(),
      webSearchCount: 0,
      annotations: [],
      includeUsage: false,
      includeLogprobs: false,
    };
  }

  private makeChunk(
    delta: OpenAI.ChatCompletionChunk.Choice.Delta,
    finish_reason: OpenAI.ChatCompletionChunk.Choice["finish_reason"] = null,
    usage?: OpenAI.CompletionUsage,
    logprobs?: OpenAI.ChatCompletionChunk.Choice.Logprobs | null
  ): OpenAI.ChatCompletionChunk {
    const chunk: OpenAI.ChatCompletionChunk = {
      id: this.streamState.id,
      object: "chat.completion.chunk",
      created: this.streamState.created,
      model: this.streamState.model,
      service_tier: this.streamState.serviceTier,
      choices: [
        {
          index: 0,
          delta,
          finish_reason,
          logprobs: logprobs ?? null,
        },
      ],
    };
    if (this.streamState.includeUsage) {
      chunk.usage = usage ?? null;
    } else if (usage != null) {
      chunk.usage = usage;
    }
    return chunk;
  }

  private handleResponseCreated(
    event: OpenAI.Responses.ResponseCreatedEvent
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
    event: OpenAI.Responses.ResponseOutputItemAddedEvent
  ): OpenAI.ChatCompletionChunk | null {
    const item = event.item;
    if (item.type === "function_call" || item.type === "custom_tool_call") {
      this.streamState.hasToolCall = true;
      this.streamState.toolCallCounter++;
      const index = this.streamState.toolCallCounter;
      this.streamState.toolCallIndexes.set(item.call_id, index);
      if (item.id) {
        this.streamState.toolCallIndexes.set(item.id, index);
      }

      if (item.type === "function_call") {
        const fc = item as OpenAI.Responses.ResponseFunctionToolCall;
        if (this.legacyFunctionMode) {
          return this.makeChunk({
            role: "assistant",
            content: null,
            function_call: { name: fc.name, arguments: "" },
          });
        }
        return this.makeChunk({
          role: "assistant",
          content: null,
          tool_calls: [
            {
              index,
              id: fc.call_id,
              type: "function",
              function: { name: fc.name, arguments: "" },
            },
          ],
        });
      }

      const custom = item as OpenAI.Responses.ResponseCustomToolCall;
      return this.makeChunk({
        role: "assistant",
        content: null,
        tool_calls: [
          {
            index,
            id: custom.call_id,
            type: "custom",
            custom: { name: custom.name, input: "" },
          },
        ],
      } as any);
    }
    return null;
  }

  private handleOutputItemDone(event: OpenAI.Responses.ResponseOutputItemDoneEvent): void {
    const item = event.item;
    if (item?.type === "web_search_call") {
      if (item.status === "completed" && item.action.type === "search") {
        this.streamState.webSearchCount++;
      }
    }
  }

  private handleTextDelta(
    event: OpenAI.Responses.ResponseTextDeltaEvent
  ): OpenAI.ChatCompletionChunk {
    const logprobs = this.streamState.includeLogprobs
      ? {
          content: event.logprobs.map(logprob => this.convertTokenLogprob(logprob)),
          refusal: null,
        }
      : undefined;
    return this.makeChunk({ role: "assistant", content: event.delta }, null, undefined, logprobs);
  }

  private handleFunctionCallDelta(
    event: OpenAI.Responses.ResponseFunctionCallArgumentsDeltaEvent
  ): OpenAI.ChatCompletionChunk {
    const index =
      this.streamState.toolCallIndexes.get(event.item_id) ?? this.streamState.toolCallCounter;
    if (this.legacyFunctionMode) {
      return this.makeChunk({
        role: "assistant",
        content: "",
        function_call: { arguments: event.delta },
      });
    }

    return this.makeChunk({
      role: "assistant",
      content: "",
      tool_calls: [
        {
          index,
          type: "function",
          function: { arguments: event.delta },
        },
      ],
    });
  }

  private handleCustomToolCallDelta(
    event: OpenAI.Responses.ResponseCustomToolCallInputDeltaEvent
  ): OpenAI.ChatCompletionChunk {
    const index =
      this.streamState.toolCallIndexes.get(event.item_id) ?? this.streamState.toolCallCounter;
    return this.makeChunk({
      role: "assistant",
      content: "",
      tool_calls: [
        {
          index,
          type: "custom",
          custom: { input: event.delta },
        },
      ],
    } as any);
  }

  private handleRefusalDelta(
    event: OpenAI.Responses.ResponseRefusalDeltaEvent
  ): OpenAI.ChatCompletionChunk {
    return this.makeChunk({ refusal: event.delta });
  }

  private handleReasoningSummaryDelta(event: {
    delta: string;
    item_id: string;
  }): OpenAI.ChatCompletionChunk {
    return this.makeChunk({
      role: "assistant",
      content: "",
      ...{
        reasoning: event.delta,
        reasoning_details: [
          {
            index: "0",
            type: "reasoning.summary",
            format: "openai-responses-v1",
            summary: event.delta,
          },
        ],
      },
    });
  }

  private handleCompleted(
    event: OpenAI.Responses.ResponseCompletedEvent
  ): OpenAI.ChatCompletionChunk | OpenAI.ChatCompletionChunk[] {
    const state = this.streamState;
    const resp = event.response;
    const finishReason = state.hasToolCall
      ? this.legacyFunctionMode
        ? ("function_call" as const)
        : ("tool_calls" as const)
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
    event: OpenAI.Responses.ResponseIncompleteEvent
  ): OpenAI.ChatCompletionChunk | OpenAI.ChatCompletionChunk[] {
    const state = this.streamState;
    const resp = event.response;
    const finishChunk = this.makeChunk({}, this.mapStatus(resp.status, resp.incomplete_details));
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

  private recordStreamAnnotation(annotation: unknown): void {
    if (
      annotation == null ||
      typeof annotation !== "object" ||
      !("type" in annotation) ||
      annotation.type !== "url_citation" ||
      !("title" in annotation) ||
      typeof annotation.title !== "string" ||
      !("url" in annotation) ||
      typeof annotation.url !== "string" ||
      !("start_index" in annotation) ||
      typeof annotation.start_index !== "number" ||
      !("end_index" in annotation) ||
      typeof annotation.end_index !== "number"
    ) {
      return;
    }

    this.streamState.annotations.push({
      type: "url_citation",
      url_citation: {
        title: annotation.title,
        url: annotation.url,
        start_index: annotation.start_index,
        end_index: annotation.end_index,
      },
    });
  }
}
