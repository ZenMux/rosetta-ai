import type OpenAI from "openai";
import type Anthropic from "@anthropic-ai/sdk";

type RespResponse = OpenAI.Responses.Response;
type RespStreamEvent = OpenAI.Responses.ResponseStreamEvent;

interface StreamState {
  id: string;
  model: string;
  currentBlockIndex: number;
  currentBlockType: string | null;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  webSearchCount: number;
  stopReason: Anthropic.StopReason | null;
}

export class MessagesToResponsesConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (Messages → Responses, forward) ---

  convertRequest(params: Anthropic.MessageCreateParams): OpenAI.Responses.ResponseCreateParams {
    const result: OpenAI.Responses.ResponseCreateParams = {
      model: params.model as string,
      input: this.convertMessages(params.messages),
    };

    if (params.system) {
      if (typeof params.system === "string") {
        result.instructions = params.system;
      } else {
        result.instructions = params.system.map(b => b.text).join("\n");
      }
    }
    if (params.max_tokens != null) {
      result.max_output_tokens = params.max_tokens;
    }
    if (params.temperature != null) {
      result.temperature = params.temperature;
    }
    if (params.top_p != null) {
      result.top_p = params.top_p;
    }
    if (params.tools) {
      result.tools = this.convertTools(params.tools);
    }
    if (params.tool_choice) {
      result.tool_choice = this.convertToolChoice(params.tool_choice);
      if (
        "disable_parallel_tool_use" in params.tool_choice &&
        params.tool_choice.disable_parallel_tool_use === true
      ) {
        result.parallel_tool_calls = false;
      }
    }
    if (params.thinking) {
      result.reasoning = this.convertThinking(params.thinking);
    }
    if (params.output_config?.format?.type === "json_schema") {
      result.text = {
        format: {
          type: "json_schema",
          name: "response",
          schema: params.output_config.format.schema,
        },
      };
    }
    if (params.metadata?.user_id) {
      result.metadata = { user_id: params.metadata.user_id };
    }
    if (params.service_tier != null) {
      result.service_tier = params.service_tier as any;
    }
    if (params.stream === true) {
      (result as any).stream = true;
    }

    return result;
  }

  // --- Response conversion (Responses → Messages, backward) ---

  convertResponse(response: RespResponse): Anthropic.Message {
    const content: Anthropic.ContentBlock[] = [];
    let webSearchCount = 0;

    for (const item of response.output) {
      if (item.type === "reasoning") {
        const ri = item as OpenAI.Responses.ResponseReasoningItem;
        const summaryText = ri.summary?.map(s => s.text).join("") ?? "";
        content.push({
          type: "thinking",
          thinking: summaryText,
          signature: "",
        });
      } else if (
        item.type === "web_search_call" &&
        item.status === "completed" &&
        item.action.type === "search"
      ) {
        webSearchCount++;
      }
    }

    // Tool calls (function_call items)
    for (const item of response.output) {
      if (item.type === "function_call") {
        const fc = item as OpenAI.Responses.ResponseFunctionToolCall;
        let parsedInput: Record<string, unknown>;
        try {
          parsedInput = JSON.parse(fc.arguments);
        } catch {
          parsedInput = {};
        }
        content.push({
          type: "tool_use",
          id: fc.call_id,
          name: fc.name,
          input: parsedInput,
          caller: { type: "direct" },
        });
      }
    }

    // Text / refusal from message items + web search annotations
    for (const item of response.output) {
      if (item.type === "message") {
        const msg = item as OpenAI.Responses.ResponseOutputMessage;
        for (const part of msg.content) {
          if (part.type === "output_text") {
            content.push({
              type: "text",
              text: part.text,
              citations: null,
            });
          }
        }
      }
    }

    if (content.length === 0) {
      content.push({ type: "text", text: "", citations: null });
    }

    const stopReason = this.statusToStopReason(response.status, response.output);
    const usage = this.buildUsage(response.usage, webSearchCount);

    return {
      id: response.id,
      type: "message",
      role: "assistant",
      model: response.model as Anthropic.Model,
      content,
      stop_reason: stopReason,
      stop_sequence: null,
      usage,
      container: null,
    } as any;
  }

  private buildUsage(usage: any, webSearchCount: number): any {
    const u = usage ?? {};
    const inputTokens = u.input_tokens ?? 0;
    const outputTokens = u.output_tokens ?? 0;
    const totalTokens = u.total_tokens ?? inputTokens + outputTokens;
    const cachedTokens = u.input_tokens_details?.cached_tokens ?? 0;
    const reasoningTokens = u.output_tokens_details?.reasoning_tokens ?? 0;

    return {
      completion_tokens: outputTokens,
      prompt_tokens: inputTokens,
      total_tokens: totalTokens,
      completion_tokens_details: {
        reasoning_tokens: reasoningTokens,
      },
      prompt_tokens_details: {
        cached_tokens: cachedTokens,
        ...(webSearchCount > 0 ? { web_search: webSearchCount } : {}),
      },
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      cache_read_input_tokens: cachedTokens,
      server_tool_use:
        webSearchCount > 0
          ? {
              web_fetch_requests: 0,
              web_search_requests: webSearchCount,
            }
          : null,
      cache_creation_input_tokens: 0,
      service_tier: "standard",
      audio_input_tokens: 0,
      audio_cache_read_tokens: 0,
    };
  }

  // --- Stream conversion (Responses → Messages, backward) ---

  async *convertStream(
    stream: AsyncIterable<RespStreamEvent>
  ): AsyncIterable<Anthropic.RawMessageStreamEvent> {
    for await (const event of stream) {
      const events = this.convertStreamEvent(event);
      for (const e of events) {
        yield e;
      }
    }
  }

  convertStreamEvent(event: RespStreamEvent): Anthropic.RawMessageStreamEvent[] {
    const state = this.streamState;
    const events: Anthropic.RawMessageStreamEvent[] = [];

    switch (event.type) {
      case "response.created": {
        const resp = (event as OpenAI.Responses.ResponseCreatedEvent).response;
        state.id = resp.id;
        state.model = resp.model as string;

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
        break;
      }

      case "response.output_item.added": {
        const addedEvent = event as OpenAI.Responses.ResponseOutputItemAddedEvent;
        const item = addedEvent.item;

        if (item.type === "reasoning") {
          state.currentBlockIndex++;
          state.currentBlockType = "thinking";
          events.push({
            type: "content_block_start",
            index: state.currentBlockIndex,
            content_block: { type: "thinking", thinking: "", signature: "" },
          });
        } else if (item.type === "function_call") {
          state.currentBlockIndex++;
          state.currentBlockType = "tool_use";
          const fc = item as OpenAI.Responses.ResponseFunctionToolCall;
          events.push({
            type: "content_block_start",
            index: state.currentBlockIndex,
            content_block: {
              type: "tool_use",
              id: fc.call_id,
              name: fc.name,
              input: {},
              caller: { type: "direct" },
            },
          });
        } else if (item.type === "message") {
          state.currentBlockIndex++;
          state.currentBlockType = "text";
          events.push({
            type: "content_block_start",
            index: state.currentBlockIndex,
            content_block: { type: "text", text: "", citations: null },
          });
        }
        break;
      }

      case "response.reasoning_summary_text.delta": {
        const reasoningEvent = event as any;
        events.push({
          type: "content_block_delta",
          index: state.currentBlockType === "thinking" ? state.currentBlockIndex : 0,
          delta: { type: "thinking_delta", thinking: reasoningEvent.delta } as any,
        });
        break;
      }

      case "response.output_text.delta": {
        const textEvent = event as OpenAI.Responses.ResponseTextDeltaEvent;
        events.push({
          type: "content_block_delta",
          index: state.currentBlockIndex,
          delta: { type: "text_delta", text: textEvent.delta },
        });
        break;
      }

      case "response.function_call_arguments.delta": {
        const fcEvent = event as OpenAI.Responses.ResponseFunctionCallArgumentsDeltaEvent;
        events.push({
          type: "content_block_delta",
          index: state.currentBlockIndex,
          delta: { type: "input_json_delta", partial_json: fcEvent.delta },
        });
        break;
      }

      case "response.output_item.done": {
        const doneEvent = event as OpenAI.Responses.ResponseOutputItemDoneEvent;
        const item = doneEvent.item;
        if (item?.type === "web_search_call") {
          if (item.status === "completed" && item.action.type === "search") {
            state.webSearchCount++;
          }
          break;
        }

        events.push({
          type: "content_block_stop",
          index: state.currentBlockIndex,
        });
        break;
      }

      case "response.completed": {
        const completedEvent = event as OpenAI.Responses.ResponseCompletedEvent;
        const resp = completedEvent.response;
        const stopReason = this.statusToStopReason(resp.status, resp.output);

        events.push({
          type: "message_delta",
          delta: { stop_reason: stopReason, stop_sequence: null, container: null },
          usage: this.buildUsage(resp.usage, state.webSearchCount),
        } as any);
        events.push({ type: "message_stop" });
        break;
      }

      case "response.incomplete": {
        const incompleteEvent = event as OpenAI.Responses.ResponseIncompleteEvent;
        const resp = incompleteEvent.response;

        events.push({
          type: "message_delta",
          delta: { stop_reason: "max_tokens", stop_sequence: null, container: null },
          usage: this.buildUsage(resp.usage, state.webSearchCount),
        } as any);
        events.push({ type: "message_stop" });
        break;
      }

      case "response.failed": {
        events.push({
          type: "message_delta",
          delta: { stop_reason: "end_turn", stop_sequence: null, container: null },
          usage: this.buildUsage(null, state.webSearchCount),
        } as any);
        events.push({ type: "message_stop" });
        break;
      }
    }

    return events;
  }

  // --- Private: request helpers ---

  private convertMessages(
    messages: Anthropic.MessageParam[]
  ): OpenAI.Responses.ResponseInputItem[] {
    const input: OpenAI.Responses.ResponseInputItem[] = [];

    for (const msg of messages) {
      if (msg.role === "user") {
        if (typeof msg.content === "string") {
          input.push({ role: "user", type: "message", content: msg.content });
        } else {
          const parts: any[] = [];
          const toolResults: Anthropic.ToolResultBlockParam[] = [];

          for (const block of msg.content) {
            if (block.type === "text") {
              parts.push({ type: "input_text", text: block.text });
            } else if (block.type === "image") {
              const src = block.source;
              if (src.type === "base64") {
                parts.push({
                  type: "input_image",
                  image_url: `data:${src.media_type};base64,${src.data}`,
                });
              } else if (src.type === "url") {
                parts.push({
                  type: "input_image",
                  image_url: (src as Anthropic.URLImageSource).url,
                });
              }
            } else if (block.type === "document") {
              const src = block.source;
              if (src.type === "base64") {
                parts.push({
                  type: "input_file",
                  file_data: `data:${src.media_type};base64,${src.data}`,
                });
              } else if (src.type === "url") {
                parts.push({ type: "input_file", file_data: src.url });
              } else if (src.type === "text") {
                parts.push({ type: "input_text", text: src.data });
              }
            } else if (block.type === "tool_result") {
              toolResults.push(block as Anthropic.ToolResultBlockParam);
            }
          }

          // Emit tool results as function_call_output items
          for (const tr of toolResults) {
            const output =
              typeof tr.content === "string"
                ? tr.content
                : tr.content
                  ? tr.content
                      .filter((b): b is Anthropic.TextBlockParam => b.type === "text")
                      .map(b => b.text)
                      .join("\n")
                  : "";
            input.push({
              type: "function_call_output",
              call_id: tr.tool_use_id,
              output,
            });
          }

          // Emit user message if there are content parts
          if (parts.length > 0) {
            input.push({ role: "user", type: "message", content: parts });
          }
        }
      } else if (msg.role === "assistant") {
        if (typeof msg.content === "string") {
          input.push({
            role: "assistant",
            type: "message",
            content: [{ type: "output_text", text: msg.content }],
          } as any);
        } else {
          for (const block of msg.content) {
            if (block.type === "text") {
              input.push({
                role: "assistant",
                type: "message",
                content: [{ type: "output_text", text: block.text }],
              } as any);
            } else if (block.type === "tool_use") {
              input.push({
                type: "function_call",
                call_id: block.id,
                name: block.name,
                arguments: JSON.stringify(block.input),
              });
            }
          }
        }
      }
    }

    return input;
  }

  private convertTools(tools: Anthropic.ToolUnion[]): OpenAI.Responses.Tool[] {
    const result: OpenAI.Responses.Tool[] = [];
    for (const t of tools) {
      if ("input_schema" in t) {
        result.push({
          type: "function",
          name: t.name,
          description: t.description,
          parameters: t.input_schema,
          strict: t.strict ?? null,
        });
      } else if (
        "type" in t &&
        (t.type === "web_search_20250305" || t.type === "web_search_20260209")
      ) {
        result.push({ type: "web_search" });
      }
    }
    return result;
  }

  private convertToolChoice(
    choice: Anthropic.ToolChoice
  ): OpenAI.Responses.ResponseCreateParams["tool_choice"] {
    switch (choice.type) {
      case "auto":
        return "auto";
      case "any":
        return "required";
      case "none":
        return "none";
      case "tool":
        return { type: "function", name: (choice as Anthropic.ToolChoiceTool).name };
      default:
        return "auto";
    }
  }

  private convertThinking(
    thinking: Anthropic.ThinkingConfigParam
  ): OpenAI.Responses.ResponseCreateParams["reasoning"] {
    if (thinking.type === "disabled") return { effort: "low" };
    if (thinking.type === "enabled") {
      const budget = (thinking as Anthropic.ThinkingConfigEnabled).budget_tokens;
      if (budget <= 2048) return { effort: "low" };
      if (budget <= 5120) return { effort: "medium" };
      return { effort: "high" };
    }
    return { effort: "medium" };
  }

  // --- Private: response helpers ---

  private statusToStopReason(
    status: RespResponse["status"],
    output?: OpenAI.Responses.ResponseOutputItem[]
  ): Anthropic.StopReason {
    if (output) {
      const hasToolCall = output.some(item => item.type === "function_call");
      if (hasToolCall) return "tool_use";
    }
    switch (status) {
      case "completed":
        return "end_turn";
      case "incomplete":
        return "max_tokens";
      case "failed":
        return "end_turn";
      default:
        return "end_turn";
    }
  }

  // --- Private: stream helpers ---

  private createStreamState(): StreamState {
    return {
      id: "",
      model: "",
      currentBlockIndex: -1,
      currentBlockType: null,
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      webSearchCount: 0,
      stopReason: null,
    };
  }
}
