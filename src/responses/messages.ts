import type OpenAI from "openai";
import type Anthropic from "@anthropic-ai/sdk";

type RespResponse = OpenAI.Responses.Response;
type RespStreamEvent = OpenAI.Responses.ResponseStreamEvent;

const DEFAULT_MAX_TOKENS = 4096;

interface StreamState {
  id: string;
  model: string;
  seq: number;
  outputIndex: number;
  reasoningStarted: boolean;
  toolCallCount: number;
  textMessageStarted: boolean;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  webSearchCount: number;
  stopReason: string | null;
}

export class ResponsesToMessagesConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (Responses → Messages, forward) ---

  convertRequest(params: OpenAI.Responses.ResponseCreateParams): Anthropic.MessageCreateParams {
    const systemBlocks: string[] = [];
    const messages: Anthropic.MessageParam[] = [];

    if (params.instructions) {
      systemBlocks.push(params.instructions);
    }

    this.convertInput(systemBlocks, messages, params.input);

    const result: Anthropic.MessageCreateParams = {
      model: params.model as string,
      max_tokens: params.max_output_tokens ?? DEFAULT_MAX_TOKENS,
      messages,
    };

    if (systemBlocks.length > 0) {
      result.system = systemBlocks.join("\n");
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
    if (params.tool_choice != null) {
      result.tool_choice = this.convertToolChoice(params.tool_choice, params.parallel_tool_calls);
    }
    if (params.reasoning) {
      result.thinking = this.convertReasoning(params.reasoning);
    }
    if (params.text?.format?.type === "json_schema") {
      const fmt = params.text.format as any;
      result.output_config = {
        format: {
          type: "json_schema",
          schema: fmt.schema ?? {},
        },
      };
    }
    if (params.service_tier != null) {
      const tier = params.service_tier as string;
      if (tier === "auto" || tier === "standard_only") {
        result.service_tier = tier;
      }
    }
    if (params.stream === true) {
      (result as any).stream = true;
    }

    return result;
  }

  // --- Response conversion (Messages → Responses, backward) ---

  convertResponse(message: Anthropic.Message): RespResponse {
    const output: OpenAI.Responses.ResponseOutputItem[] = [];

    for (const block of message.content) {
      if (block.type === "thinking") {
        output.push({
          type: "reasoning",
          id: `rs_${this.generateId()}`,
          summary: [{ type: "summary_text", text: block.thinking }],
        });
      }
    }

    for (const block of message.content) {
      if (block.type === "tool_use" || block.type === "server_tool_use") {
        output.push({
          type: "function_call",
          id: `fc_${this.generateId()}`,
          call_id: block.id,
          name: block.name,
          arguments: JSON.stringify(block.input),
          status: "completed",
        });
      }
    }

    const textParts: string[] = [];
    for (const block of message.content) {
      if (block.type === "text") {
        textParts.push(block.text);
      }
    }

    if (textParts.length > 0) {
      output.push({
        type: "message",
        id: `msg_${this.generateId()}`,
        role: "assistant",
        status: "completed",
        content: [
          {
            type: "output_text",
            text: textParts.join(""),
            annotations: [],
            logprobs: null as any,
          },
        ],
      });
    }

    if (output.length === 0) {
      output.push({
        type: "message",
        id: `msg_${this.generateId()}`,
        role: "assistant",
        status: "completed",
        content: [
          {
            type: "output_text",
            text: "",
            annotations: [],
            logprobs: null as any,
          },
        ],
      });
    }

    const status = this.stopReasonToStatus(message.stop_reason, message.content);
    const usage = message.usage;

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
      truncation: null,
      user: undefined,
      usage: {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.input_tokens + usage.output_tokens,
        input_tokens_details: {
          cached_tokens: usage.cache_read_input_tokens ?? 0,
        },
        output_tokens_details: {
          reasoning_tokens: 0,
        },
      },
    } as unknown as RespResponse;
  }

  // --- Stream conversion (Messages → Responses, backward) ---

  async *convertStream(
    stream: AsyncIterable<Anthropic.RawMessageStreamEvent>
  ): AsyncIterable<RespStreamEvent> {
    for await (const event of stream) {
      const events = this.convertStreamEvent(event);
      for (const e of events) {
        yield e;
      }
    }
  }

  convertStreamEvent(event: Anthropic.RawMessageStreamEvent): RespStreamEvent[] {
    const state = this.streamState;
    const events: RespStreamEvent[] = [];

    switch (event.type) {
      case "message_start": {
        const msg = event.message;
        state.id = msg.id;
        state.model = msg.model;
        state.inputTokens = msg.usage.input_tokens;
        state.cacheReadTokens = msg.usage.cache_read_input_tokens ?? 0;
        state.cacheCreationTokens = msg.usage.cache_creation_input_tokens ?? 0;

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
        break;
      }

      case "content_block_start": {
        const block = event.content_block;

        if (block.type === "thinking") {
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
        } else if (block.type === "tool_use" || block.type === "server_tool_use") {
          if (state.toolCallCount === 0 && state.reasoningStarted) {
            state.outputIndex++;
          }
          state.toolCallCount++;
          const toolOutputIndex = state.outputIndex + state.toolCallCount - 1;

          events.push({
            type: "response.output_item.added",
            item: {
              type: "function_call",
              id: `fc_${this.generateId()}`,
              call_id: block.id,
              name: block.name,
              arguments: "",
              status: "in_progress",
            },
            output_index: toolOutputIndex,
            sequence_number: state.seq++,
          } as RespStreamEvent);
        } else if (block.type === "text") {
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
        } else if (block.type === "web_search_tool_result") {
          state.webSearchCount++;
        }
        break;
      }

      case "content_block_delta": {
        const delta = event.delta;

        if (delta.type === "thinking_delta") {
          events.push({
            type: "response.reasoning_summary_text.delta",
            item_id: `rs_${this.generateId()}`,
            output_index: state.outputIndex,
            summary_index: 0,
            delta: delta.thinking,
            sequence_number: state.seq++,
          } as RespStreamEvent);
        } else if (delta.type === "input_json_delta") {
          const toolOutputIndex =
            state.outputIndex + (state.reasoningStarted ? 1 : 0) + state.toolCallCount - 1;
          events.push({
            type: "response.function_call_arguments.delta",
            item_id: `fc_${this.generateId()}`,
            output_index: toolOutputIndex,
            delta: delta.partial_json,
            sequence_number: state.seq++,
          } as RespStreamEvent);
        } else if (delta.type === "text_delta") {
          const msgOutputIndex =
            state.outputIndex + (state.reasoningStarted ? 1 : 0) + state.toolCallCount;
          events.push({
            type: "response.output_text.delta",
            item_id: `msg_${this.generateId()}`,
            output_index: msgOutputIndex,
            content_index: 0,
            delta: delta.text,
            sequence_number: state.seq++,
          } as RespStreamEvent);
        }
        break;
      }

      case "content_block_stop": {
        break;
      }

      case "message_delta": {
        const usage = event.usage;
        state.outputTokens = usage.output_tokens;
        if (usage.input_tokens != null) {
          state.inputTokens = usage.input_tokens;
        }
        state.stopReason = event.delta.stop_reason;
        state.webSearchCount = usage.server_tool_use?.web_search_requests ?? state.webSearchCount;
        break;
      }

      case "message_stop": {
        const status = this.stopReasonToStatus(state.stopReason as any, undefined);
        const resp = this.makeSkeletonResponse();
        resp.status = status as any;
        resp.usage = {
          input_tokens: state.inputTokens,
          output_tokens: state.outputTokens,
          total_tokens: state.inputTokens + state.outputTokens,
          input_tokens_details: {
            cached_tokens: state.cacheReadTokens,
          },
          output_tokens_details: {
            reasoning_tokens: 0,
          },
        };

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
        break;
      }
    }

    return events;
  }

  // --- Private: request helpers ---

  private convertInput(
    systemBlocks: string[],
    messages: Anthropic.MessageParam[],
    input: OpenAI.Responses.ResponseCreateParams["input"]
  ): void {
    if (typeof input === "string") {
      messages.push({
        role: "user",
        content: [{ type: "text", text: input }],
      });
      return;
    }

    const pendingToolUses: Anthropic.ToolUseBlockParam[] = [];

    const flushToolUses = () => {
      if (pendingToolUses.length > 0) {
        messages.push({
          role: "assistant",
          content: [...pendingToolUses],
        });
        pendingToolUses.length = 0;
      }
    };

    for (const item of input!) {
      const typed = item as any;

      if (typed.type === "message") {
        flushToolUses();
        if (typed.role === "system" || typed.role === "developer") {
          const text = this.extractText(typed.content);
          if (text) systemBlocks.push(text);
        } else if (typed.role === "user") {
          const content = this.convertInputContent(typed.content);
          messages.push({ role: "user", content });
        } else if (typed.role === "assistant") {
          const blocks: Anthropic.ContentBlockParam[] = [];
          if (Array.isArray(typed.content)) {
            for (const part of typed.content) {
              if (part.type === "output_text") {
                blocks.push({ type: "text", text: part.text });
              }
            }
          }
          if (blocks.length > 0) {
            messages.push({ role: "assistant", content: blocks });
          }
        }
      } else if (typed.type === "function_call") {
        let parsedInput: Record<string, unknown>;
        try {
          parsedInput = JSON.parse(typed.arguments);
        } catch {
          parsedInput = {};
        }
        pendingToolUses.push({
          type: "tool_use",
          id: typed.call_id,
          name: typed.name,
          input: parsedInput,
        });
      } else if (typed.type === "function_call_output") {
        flushToolUses();
        const toolResult: Anthropic.ToolResultBlockParam = {
          type: "tool_result",
          tool_use_id: typed.call_id,
          content: [{ type: "text", text: typed.output ?? "" }],
        };
        const last = messages[messages.length - 1];
        if (last && last.role === "user" && Array.isArray(last.content)) {
          const lastContent = last.content as Anthropic.ContentBlockParam[];
          if (lastContent.length > 0 && (lastContent[0] as any).type === "tool_result") {
            lastContent.push(toolResult);
            continue;
          }
        }
        messages.push({ role: "user", content: [toolResult] });
      } else if ("role" in typed && "content" in typed && typeof typed.content === "string") {
        flushToolUses();
        const role = typed.role;
        if (role === "system" || role === "developer") {
          systemBlocks.push(typed.content);
        } else if (role === "user") {
          messages.push({ role: "user", content: [{ type: "text", text: typed.content }] });
        } else if (role === "assistant") {
          messages.push({ role: "assistant", content: [{ type: "text", text: typed.content }] });
        }
      }
    }

    flushToolUses();
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

  private convertInputContent(content: any): Anthropic.ContentBlockParam[] {
    if (typeof content === "string") {
      return [{ type: "text", text: content }];
    }
    if (!Array.isArray(content)) return [{ type: "text", text: "" }];

    const blocks: Anthropic.ContentBlockParam[] = [];
    for (const part of content) {
      if (part.type === "input_text") {
        blocks.push({ type: "text", text: part.text });
      } else if (part.type === "input_image") {
        blocks.push(this.convertInputImage(part));
      } else if (part.type === "input_file") {
        blocks.push(this.convertInputFile(part));
      }
    }
    return blocks.length > 0 ? blocks : [{ type: "text", text: "" }];
  }

  private convertInputImage(part: any): Anthropic.ImageBlockParam {
    const url: string = part.image_url || "";
    const dataUriMatch = url.match(/^data:(image\/[a-z+]+);base64,(.+)$/);

    if (dataUriMatch) {
      return {
        type: "image",
        source: {
          type: "base64",
          media_type: dataUriMatch[1] as Anthropic.Base64ImageSource["media_type"],
          data: dataUriMatch[2],
        },
      };
    }

    return {
      type: "image",
      source: { type: "url", url },
    };
  }

  private convertInputFile(part: any): Anthropic.DocumentBlockParam {
    const data: string = part.file_data || part.file_url || "";
    const dataUriMatch = data.match(/^data:([^;]+);base64,(.+)$/);

    if (dataUriMatch) {
      return {
        type: "document",
        source: {
          type: "base64",
          media_type: dataUriMatch[1] as Anthropic.Base64PDFSource["media_type"],
          data: dataUriMatch[2],
        },
      };
    }

    if (data.startsWith("http")) {
      return {
        type: "document",
        source: { type: "url", url: data },
      };
    }

    return {
      type: "document",
      source: { type: "text", data, media_type: "text/plain" } as Anthropic.PlainTextSource,
    };
  }

  private convertTools(
    tools: OpenAI.Responses.ResponseCreateParams["tools"]
  ): Anthropic.ToolUnion[] {
    if (!tools) return [];

    const result: Anthropic.ToolUnion[] = [];
    for (const t of tools) {
      const tt = t as any;
      if (tt.type === "function") {
        result.push({
          name: tt.name,
          description: tt.description,
          input_schema: (tt.parameters ?? { type: "object" }) as Anthropic.Tool.InputSchema,
        });
      } else if (this.isWebSearch(tt)) {
        result.push({ type: "web_search_20250305" } as any);
      }
    }
    return result;
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
    parallelToolCalls?: boolean | null
  ): Anthropic.ToolChoice {
    const disableParallel = parallelToolCalls === false ? true : undefined;

    if (typeof choice === "string") {
      switch (choice) {
        case "auto":
          return disableParallel
            ? { type: "auto", disable_parallel_tool_use: true }
            : { type: "auto" };
        case "required":
          return disableParallel
            ? { type: "any", disable_parallel_tool_use: true }
            : { type: "any" };
        case "none":
          return { type: "none" };
        default:
          return { type: "auto" };
      }
    }

    if (typeof choice === "object" && choice !== null) {
      if (choice.type === "function" && choice.name) {
        return disableParallel
          ? { type: "tool", name: choice.name, disable_parallel_tool_use: true }
          : { type: "tool", name: choice.name };
      }
    }

    return { type: "auto" };
  }

  private convertReasoning(
    reasoning: OpenAI.Responses.ResponseCreateParams["reasoning"]
  ): Anthropic.ThinkingConfigParam {
    if (!reasoning || !reasoning.effort) {
      return { type: "disabled" };
    }

    const budgetMap: Record<string, number> = {
      low: 2048,
      medium: 5120,
      high: 10240,
    };

    const budget = budgetMap[reasoning.effort] ?? 10240;
    return { type: "enabled", budget_tokens: budget };
  }

  // --- Private: response helpers ---

  private stopReasonToStatus(
    stopReason: Anthropic.Message["stop_reason"] | string | null,
    content?: Anthropic.ContentBlock[]
  ): RespResponse["status"] {
    if (content) {
      const hasToolCall = content.some(b => b.type === "tool_use" || b.type === "server_tool_use");
      if (hasToolCall) return "completed";
    }
    switch (stopReason) {
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

  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  // --- Private: stream helpers ---

  private createStreamState(): StreamState {
    return {
      id: "",
      model: "",
      seq: 0,
      outputIndex: 0,
      reasoningStarted: false,
      toolCallCount: 0,
      textMessageStarted: false,
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      cacheCreationTokens: 0,
      webSearchCount: 0,
      stopReason: null,
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
