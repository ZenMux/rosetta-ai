import type OpenAI from "openai";
import type Anthropic from "@anthropic-ai/sdk";

type RespResponse = OpenAI.Responses.Response;
type RespStreamEvent = OpenAI.Responses.ResponseStreamEvent;

interface StreamState {
  id: string;
  model: string;
  created: number;
  seq: number;
  outputIndex: number;
  messageStarted: boolean;
  stopReason: Anthropic.StopReason | null;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  webSearchCount: number;
}

export class MessagesToResponsesConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion ---

  convertRequest(
    params: Anthropic.MessageCreateParams,
  ): OpenAI.Responses.ResponseCreateParams {
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
        (result as any).parallel_tool_calls = false;
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
    if (params.service_tier != null) {
      result.service_tier = params.service_tier as any;
    }
    if (params.stream === true) {
      (result as any).stream = true;
    }

    return result;
  }

  // --- Response conversion ---

  convertResponse(message: Anthropic.Message): RespResponse {
    const output: OpenAI.Responses.ResponseOutputItem[] = [];
    let hasThinking = false;

    // Reasoning (thinking blocks)
    for (const block of message.content) {
      if (block.type === "thinking") {
        if (!hasThinking) {
          hasThinking = true;
          output.push({
            type: "reasoning",
            id: `rs_${this.generateId()}`,
            summary: [{ type: "summary_text", text: block.thinking }],
          } as any);
        }
      }
    }

    // Tool calls (tool_use / server_tool_use)
    for (const block of message.content) {
      if (block.type === "tool_use" || block.type === "server_tool_use") {
        output.push({
          type: "function_call",
          id: `fc_${this.generateId()}`,
          call_id: block.id,
          name: block.name,
          arguments: JSON.stringify(block.input),
          status: "completed",
        } as any);
      }
    }

    // Message (text blocks) + web_search_tool_result annotations
    const textParts: string[] = [];
    const annotations: any[] = [];
    let refusal: string | null = null;
    let webSearchCount = 0;

    for (const block of message.content) {
      if (block.type === "text") {
        textParts.push(block.text);
        if (block.citations) {
          for (const c of block.citations) {
            if ("url" in c) {
              annotations.push({
                type: "url_citation",
                url: (c as any).url,
                title: (c as any).title ?? "",
                start_index: (c as any).start_index ?? 0,
                end_index: (c as any).end_index ?? 0,
              });
            }
          }
        }
      } else if (block.type === "web_search_tool_result") {
        webSearchCount++;
        const content = block.content;
        if (Array.isArray(content)) {
          for (const result of content) {
            annotations.push({
              type: "url_citation",
              url: result.url,
              title: result.title,
              start_index: 0,
              end_index: 0,
            });
          }
        }
      }
    }

    if (textParts.length > 0 || refusal) {
      const content: any[] = [];
      if (refusal) {
        content.push({ type: "refusal", refusal });
      } else {
        content.push({
          type: "output_text",
          text: textParts.join(""),
          annotations,
        });
      }
      output.push({
        type: "message",
        id: `msg_${this.generateId()}`,
        role: "assistant",
        status: this.stopReasonToMessageStatus(message.stop_reason),
        content,
      } as any);
    }

    const status = this.stopReasonToStatus(message.stop_reason);

    return {
      id: message.id,
      object: "response",
      created_at: Math.floor(Date.now() / 1000),
      model: message.model as string,
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
      truncation: null as any,
      user: undefined as any,
      usage: this.convertUsage(message.usage, webSearchCount),
    } as unknown as RespResponse;
  }

  private convertUsage(
    usage: Anthropic.Usage,
    webSearchCount = 0,
  ): OpenAI.Responses.ResponseUsage {
    return {
      input_tokens: usage.input_tokens,
      output_tokens: usage.output_tokens,
      total_tokens: usage.input_tokens + usage.output_tokens,
      input_tokens_details: {
        cached_tokens: usage.cache_read_input_tokens ?? 0,
      },
      output_tokens_details: { reasoning_tokens: 0 },
      ...(webSearchCount > 0 ? { web_search: webSearchCount } as any : {}),
    } as OpenAI.Responses.ResponseUsage;
  }

  // --- Stream conversion ---

  async *convertStream(
    stream: AsyncIterable<Anthropic.RawMessageStreamEvent>,
  ): AsyncIterable<RespStreamEvent> {
    for await (const event of stream) {
      const events = this.convertStreamEvent(event);
      for (const e of events) {
        yield e;
      }
    }
  }

  convertStreamEvent(
    event: Anthropic.RawMessageStreamEvent,
  ): RespStreamEvent[] {
    const state = this.streamState;
    const events: RespStreamEvent[] = [];

    switch (event.type) {
      case "message_start": {
        const msg = event.message;
        state.id = msg.id;
        state.model = msg.model;
        state.created = Math.floor(Date.now() / 1000);

        const usage = msg.usage;
        state.inputTokens = usage.input_tokens;
        state.cacheReadTokens = usage.cache_read_input_tokens ?? 0;

        const skeleton = this.makeSkeletonResponse();
        events.push({
          type: "response.created",
          response: skeleton,
          sequence_number: state.seq++,
        } as unknown as RespStreamEvent);
        events.push({
          type: "response.in_progress",
          response: skeleton,
          sequence_number: state.seq++,
        } as unknown as RespStreamEvent);
        break;
      }

      case "content_block_start": {
        const block = event.content_block;
        if (block.type === "text") {
          if (!state.messageStarted) {
            state.messageStarted = true;
            events.push({
              type: "response.output_item.added",
              item: {
                type: "message",
                id: `msg_${this.generateId()}`,
                role: "assistant",
                status: "in_progress",
                content: [],
              } as any,
              output_index: state.outputIndex,
              sequence_number: state.seq++,
            } as unknown as RespStreamEvent);
            events.push({
              type: "response.content_part.added",
              item_id: `msg_${this.generateId()}`,
              output_index: state.outputIndex,
              content_index: 0,
              part: { type: "output_text", text: "", annotations: [] },
              sequence_number: state.seq++,
            } as unknown as RespStreamEvent);
          }
        } else if (block.type === "tool_use" || block.type === "server_tool_use") {
          state.outputIndex++;
          events.push({
            type: "response.output_item.added",
            item: {
              type: "function_call",
              id: `fc_${this.generateId()}`,
              call_id: block.id,
              name: block.name,
              arguments: "",
              status: "in_progress",
            } as any,
            output_index: state.outputIndex,
            sequence_number: state.seq++,
          } as unknown as RespStreamEvent);
        } else if (block.type === "thinking") {
          events.push({
            type: "response.output_item.added",
            item: {
              type: "reasoning",
              id: `rs_${this.generateId()}`,
              summary: [],
            } as any,
            output_index: state.outputIndex,
            sequence_number: state.seq++,
          } as unknown as RespStreamEvent);
        } else if (block.type === "web_search_tool_result") {
          state.webSearchCount++;
          const content = block.content;
          if (Array.isArray(content)) {
            for (const result of content) {
              events.push({
                type: "response.output_text.annotation.added",
                annotation: {
                  type: "url_citation",
                  url: result.url,
                  title: result.title,
                  start_index: 0,
                  end_index: 0,
                },
              } as unknown as RespStreamEvent);
            }
          }
        }
        break;
      }

      case "content_block_delta": {
        const delta = event.delta;
        if (delta.type === "text_delta") {
          events.push({
            type: "response.output_text.delta",
            item_id: `msg_${this.generateId()}`,
            output_index: state.messageStarted ? 0 : state.outputIndex,
            content_index: 0,
            delta: delta.text,
            sequence_number: state.seq++,
          } as unknown as RespStreamEvent);
        } else if (delta.type === "input_json_delta") {
          events.push({
            type: "response.function_call_arguments.delta",
            item_id: `fc_${this.generateId()}`,
            output_index: state.outputIndex,
            delta: delta.partial_json,
            sequence_number: state.seq++,
          } as unknown as RespStreamEvent);
        } else if (delta.type === "thinking_delta") {
          events.push({
            type: "response.reasoning_summary_text.delta",
            item_id: `rs_${this.generateId()}`,
            output_index: state.outputIndex,
            summary_index: 0,
            delta: delta.thinking,
            sequence_number: state.seq++,
          } as unknown as RespStreamEvent);
        } else if (delta.type === "signature_delta") {
          // Signature is Anthropic-internal; no direct Responses equivalent, skip
        }
        break;
      }

      case "content_block_stop":
        break;

      case "message_delta": {
        state.stopReason = event.delta.stop_reason;
        const usage = event.usage;
        state.outputTokens = usage.output_tokens;
        if (usage.input_tokens != null) {
          state.inputTokens = usage.input_tokens;
        }
        if (usage.server_tool_use?.web_search_requests) {
          state.webSearchCount = usage.server_tool_use.web_search_requests;
        }
        break;
      }

      case "message_stop": {
        const resp = this.makeSkeletonResponse();
        const finalStatus = this.stopReasonToStatus(state.stopReason);
        resp.status = finalStatus;
        (resp as any).usage = {
          input_tokens: state.inputTokens,
          output_tokens: state.outputTokens,
          total_tokens: state.inputTokens + state.outputTokens,
          input_tokens_details: {
            cached_tokens: state.cacheReadTokens,
          },
          output_tokens_details: { reasoning_tokens: 0 },
        } as OpenAI.Responses.ResponseUsage;

        if (finalStatus === "incomplete") {
          events.push({
            type: "response.incomplete",
            response: resp,
            sequence_number: state.seq++,
          } as unknown as RespStreamEvent);
        } else {
          events.push({
            type: "response.completed",
            response: resp,
            sequence_number: state.seq++,
          } as unknown as RespStreamEvent);
        }
        break;
      }
    }

    return events;
  }

  // --- Private helpers ---

  private convertMessages(
    messages: Anthropic.MessageParam[],
  ): OpenAI.Responses.ResponseInputItem[] {
    const input: OpenAI.Responses.ResponseInputItem[] = [];

    for (const msg of messages) {
      if (msg.role === "user") {
        if (typeof msg.content === "string") {
          input.push({ role: "user", type: "message", content: msg.content } as any);
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
                  file_data: `data:${(src as any).media_type};base64,${(src as any).data}`,
                });
              } else if (src.type === "url") {
                parts.push({ type: "input_file", file_data: (src as any).url });
              } else if (src.type === "text") {
                parts.push({ type: "input_text", text: (src as any).data });
              }
            } else if (block.type === "tool_result") {
              toolResults.push(block as Anthropic.ToolResultBlockParam);
            }
          }

          // Emit tool results as function_call_output items
          for (const tr of toolResults) {
            const output = typeof tr.content === "string"
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
            } as any);
          }

          // Emit user message if there are content parts
          if (parts.length > 0) {
            input.push({ role: "user", type: "message", content: parts } as any);
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
              } as any);
            }
          }
        }
      }
    }

    return input;
  }

  private convertTools(
    tools: Anthropic.ToolUnion[],
  ): OpenAI.Responses.Tool[] {
    const result: OpenAI.Responses.Tool[] = [];
    for (const t of tools) {
      if ("input_schema" in t) {
        result.push({
          type: "function",
          name: (t as Anthropic.Tool).name,
          description: (t as Anthropic.Tool).description,
          parameters: (t as Anthropic.Tool).input_schema as any,
          strict: (t as Anthropic.Tool).strict ?? null,
        } as any);
      } else if ("type" in t && ((t as any).type === "web_search_20250305" || (t as any).type === "web_search_20260209")) {
        result.push({ type: "web_search" } as any);
      }
    }
    return result;
  }

  private convertToolChoice(
    choice: Anthropic.ToolChoice,
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
    thinking: Anthropic.ThinkingConfigParam,
  ): OpenAI.Responses.ResponseCreateParams["reasoning"] {
    if (thinking.type === "disabled") return { effort: "low" } as any;
    if (thinking.type === "enabled") {
      const budget = (thinking as Anthropic.ThinkingConfigEnabled).budget_tokens;
      if (budget <= 2048) return { effort: "low" } as any;
      if (budget <= 5120) return { effort: "medium" } as any;
      return { effort: "high" } as any;
    }
    return { effort: "medium" } as any;
  }

  private stopReasonToStatus(
    reason: Anthropic.StopReason | null,
  ): RespResponse["status"] {
    switch (reason) {
      case "end_turn":
      case "stop_sequence":
      case "tool_use":
        return "completed";
      case "max_tokens":
        return "incomplete";
      case "refusal":
        return "failed";
      default:
        return "completed";
    }
  }

  private stopReasonToMessageStatus(
    reason: Anthropic.StopReason | null,
  ): "completed" | "incomplete" {
    if (reason === "max_tokens") return "incomplete";
    return "completed";
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  private createStreamState(): StreamState {
    return {
      id: "",
      model: "",
      created: 0,
      seq: 0,
      outputIndex: 0,
      messageStarted: false,
      stopReason: null,
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      webSearchCount: 0,
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
      truncation: null as any,
      user: undefined as any,
    } as unknown as RespResponse;
  }
}
