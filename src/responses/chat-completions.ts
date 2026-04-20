import type OpenAI from "openai";

interface StreamState {
  id: string;
  model: string;
  created: number;
  toolCallCounter: number;
}

export class ResponsesToChatCompletionConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion ---

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
        (result as any).web_search_options = webSearchOptions;
      }
    }
    if (params.tool_choice != null) {
      result.tool_choice = this.convertToolChoice(params.tool_choice);
    }
    if ((params as any).parallel_tool_calls != null) {
      (result as any).parallel_tool_calls = (params as any).parallel_tool_calls;
    }
    if (params.reasoning) {
      (result as any).reasoning_effort = (params.reasoning as any).effort ?? null;
    }
    if (params.text?.format) {
      result.response_format = this.convertTextFormat(params.text.format);
    }
    if ((params.text as any)?.verbosity) {
      (result as any).verbosity = (params.text as any).verbosity;
    }
    if (params.metadata) {
      result.metadata = params.metadata;
    }
    if (params.service_tier != null) {
      result.service_tier = params.service_tier;
    }
    if ((params as any).prompt_cache_key) {
      (result as any).prompt_cache_key = (params as any).prompt_cache_key;
    }
    if ((params as any).prompt_cache_retention != null) {
      (result as any).prompt_cache_retention = (params as any).prompt_cache_retention;
    }
    if ((params as any).safety_identifier) {
      (result as any).safety_identifier = (params as any).safety_identifier;
    }
    if (params.include) {
      for (const inc of params.include) {
        if (inc === "message.output_text.logprobs") {
          (result as any).top_logprobs = 20;
          break;
        }
      }
    }
    if (params.stream === true) {
      (result as any).stream = true;
      if ((params as any).stream_options) {
        (result as any).stream_options = (params as any).stream_options;
      }
    }

    return result;
  }

  // --- Response conversion ---

  convertResponse(response: OpenAI.Responses.Response): OpenAI.ChatCompletion {
    const textParts: string[] = [];
    const toolCalls: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[] = [];
    const annotations: OpenAI.ChatCompletionMessage.Annotation[] = [];
    let refusal: string | null = null;
    let reasoning: string | undefined;

    for (const item of response.output) {
      if (item.type === "message") {
        for (const part of (item as OpenAI.Responses.ResponseOutputMessage).content) {
          if (part.type === "output_text") {
            textParts.push(part.text);
            if (part.annotations) {
              for (const ann of part.annotations) {
                if (ann.type === "url_citation") {
                  annotations.push({
                    type: "url_citation",
                    url_citation: {
                      title: ann.title,
                      url: ann.url,
                      start_index: ann.start_index,
                      end_index: ann.end_index,
                    },
                  });
                }
              }
            }
          } else if (part.type === "refusal") {
            refusal = part.refusal;
          }
        }
      } else if (item.type === "function_call") {
        const fc = item as OpenAI.Responses.ResponseFunctionToolCall;
        toolCalls.push({
          id: fc.call_id,
          type: "function",
          function: {
            name: fc.name,
            arguments: fc.arguments,
          },
        });
      } else if (item.type === "reasoning") {
        const ri = item as OpenAI.Responses.ResponseReasoningItem;
        if (ri.summary) {
          for (const s of ri.summary) {
            if (s.type === "summary_text") {
              reasoning = reasoning ? reasoning + s.text : s.text;
            }
          }
        }
      }
    }

    const assistantMessage: OpenAI.ChatCompletionMessage = {
      role: "assistant",
      content: textParts.length > 0 ? textParts.join("") : null,
      refusal: refusal,
    };

    if (reasoning) {
      (assistantMessage as any).reasoning = reasoning;
    }
    if (toolCalls.length > 0) {
      assistantMessage.tool_calls = toolCalls;
    }
    if (annotations.length > 0) {
      assistantMessage.annotations = annotations;
    }

    return {
      id: response.id,
      object: "chat.completion",
      created: response.created_at,
      model: response.model as string,
      choices: [
        {
          index: 0,
          message: assistantMessage,
          finish_reason: this.mapStatus(response.status, response.incomplete_details),
          logprobs: null,
        },
      ],
      usage: this.convertUsage(response.usage!),
    };
  }

  // --- Stream conversion ---

  async *convertStream(
    stream: AsyncIterable<OpenAI.Responses.ResponseStreamEvent>,
  ): AsyncIterable<OpenAI.ChatCompletionChunk> {
    for await (const event of stream) {
      const chunk = this.convertStreamEvent(event);
      if (chunk) {
        yield chunk;
      }
    }
  }

  convertStreamEvent(
    event: OpenAI.Responses.ResponseStreamEvent,
  ): OpenAI.ChatCompletionChunk | null {
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
      case "response.reasoning_text.delta":
        return this.handleReasoningDelta(event);
      case "response.completed":
        return this.handleCompleted(event);
      case "response.incomplete":
        return this.handleIncomplete(event);
      case "response.failed":
        return this.handleFailed();
      default:
        return null;
    }
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
        const role = (item as any).role;
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
      const c = choice as any;
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

  private mapStatus(
    status: OpenAI.Responses.Response["status"],
    incompleteDetails?: OpenAI.Responses.Response["incomplete_details"],
  ): OpenAI.Chat.Completions.ChatCompletion.Choice["finish_reason"] {
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

  private convertUsage(usage: OpenAI.Responses.ResponseUsage): OpenAI.CompletionUsage {
    return {
      prompt_tokens: usage.input_tokens,
      completion_tokens: usage.output_tokens,
      total_tokens: usage.total_tokens,
      prompt_tokens_details: {
        cached_tokens: usage.input_tokens_details?.cached_tokens ?? 0,
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
      toolCallCounter: -1,
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
      choices: [{ index: 0, delta, finish_reason }],
      ...(usage ? { usage } : {}),
    };
  }

  private handleResponseCreated(
    event: OpenAI.Responses.ResponseCreatedEvent,
  ): OpenAI.ChatCompletionChunk {
    this.streamState.id = event.response.id;
    this.streamState.model = event.response.model as string;
    this.streamState.created = event.response.created_at;
    return this.makeChunk({ role: "assistant" });
  }

  private handleOutputItemAdded(
    event: OpenAI.Responses.ResponseOutputItemAddedEvent,
  ): OpenAI.ChatCompletionChunk | null {
    const item = event.item;
    if (item.type === "function_call") {
      this.streamState.toolCallCounter++;
      const fc = item as OpenAI.Responses.ResponseFunctionToolCall;
      return this.makeChunk({
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
    return this.makeChunk({ content: event.delta });
  }

  private handleFunctionCallDelta(
    event: OpenAI.Responses.ResponseFunctionCallArgumentsDeltaEvent,
  ): OpenAI.ChatCompletionChunk {
    return this.makeChunk({
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

  private handleReasoningDelta(
    event: OpenAI.Responses.ResponseReasoningTextDeltaEvent,
  ): OpenAI.ChatCompletionChunk {
    return this.makeChunk({
      content: "",
      ...({
        reasoning: event.delta,
      } as any),
    });
  }

  private handleCompleted(
    event: OpenAI.Responses.ResponseCompletedEvent,
  ): OpenAI.ChatCompletionChunk {
    const resp = event.response;
    const finishReason = this.mapStatus(resp.status, resp.incomplete_details);
    const usage = resp.usage ? this.convertUsage(resp.usage) : undefined;
    return this.makeChunk({ content: "" }, finishReason, usage);
  }

  private handleIncomplete(
    event: OpenAI.Responses.ResponseIncompleteEvent,
  ): OpenAI.ChatCompletionChunk {
    const resp = event.response;
    const usage = resp.usage ? this.convertUsage(resp.usage) : undefined;
    return this.makeChunk({ content: "" }, "length", usage);
  }

  private handleFailed(): OpenAI.ChatCompletionChunk {
    return this.makeChunk({ content: "" }, "stop");
  }
}
