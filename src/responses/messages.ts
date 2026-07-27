import type OpenAI from "openai";
import type Anthropic from "@anthropic-ai/sdk";
import { APIError } from "@anthropic-ai/sdk";

type RespResponse = OpenAI.Responses.Response;
type RespStreamEvent = OpenAI.Responses.ResponseStreamEvent;

const DEFAULT_MAX_TOKENS = 4096;

interface StreamState {
  id: string;
  model: string;
  seq: number;
  // Single counter shared across all item types; each new item claims and increments it once.
  nextOutputIndex: number;
  reasoningItemId: string;
  reasoningOutputIndex: number;
  reasoningText: string;
  toolCallCount: number;
  // Stores full per-call info so output_item.done can carry the finalized item.
  // `done` tracks whether content_block_stop fired, so message_stop can tell
  // complete tool_calls from ones truncated by max_tokens.
  toolCalls: Array<{
    id: string;
    call_id: string;
    name: string;
    arguments: string;
    outputIndex: number;
    done: boolean;
  }>;
  webSearchCallCount: number;
  // Keyed by Anthropic's server_tool_use block.id to pair with the later web_search_tool_result block.
  // `query` accumulates input_json_delta partial_json so done events can carry the actual search query;
  // `done` marks whether output_item.done has fired, so message_stop can backfill truncated items.
  webSearchByToolUseId: Map<
    string,
    {
      itemId: string;
      outputIndex: number;
      query: string;
      done: boolean;
    }
  >;
  // Maps a content block's stream index to its web_search tool_use_id, so input_json_delta
  // events can route to the right entry by event.index rather than assuming one open block.
  webSearchToolUseIdByBlockIndex: Map<number, string>;
  // Needed because input_json_delta is overloaded between tool_use args and web_search query.
  currentBlockType: string;
  // Anthropic may emit multiple text blocks; they all merge into one Responses message item.
  textMessageStarted: boolean;
  textMessageItemId: string;
  textMessageOutputIndex: number;
  textMessageText: string;
  // Annotations collected from citations_delta, emitted on the message's done events.
  textAnnotations: any[];
  // Char offset where the current text block starts within the merged textMessageText;
  // needed because per-block citation indices must shift when blocks merge.
  textBlockStartOffset: number;
  // Accumulates finalized items for response.completed/incomplete.output. Also covers
  // truncated tool_calls (status="incomplete") pushed at message_stop. Items are pushed in
  // completion order (which differs from output order, e.g. text finalizes at message_stop
  // after a later tool_use), so each carries its `index` for a final sort before output.
  outputItems: Array<{ index: number; item: any }>;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  // Distinct from webSearchCallCount: this tracks usage.server_tool_use.web_search_requests, not emitted items.
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
      result.thinking = this.convertReasoning(params.reasoning, result.max_tokens);
    }
    if (params.text?.format?.type === "json_schema") {
      const fmt = params.text.format as any;
      result.output_config = {
        format: {
          type: "json_schema",
          schema: fmt.schema ?? {},
        },
      };
    } else if (params.text?.format?.type === "json_object") {
      // Anthropic has no bare "json_object" mode; approximate with a permissive object schema.
      result.output_config = {
        format: {
          type: "json_schema",
          schema: { type: "object" },
        },
      };
    }
    if (params.metadata) {
      const userId =
        typeof params.metadata === "object"
          ? (params.metadata as Record<string, string>).user_id
          : undefined;
      if (userId) {
        result.metadata = { user_id: userId };
      }
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
    const textParts: string[] = [];
    const annotations: any[] = [];
    let messageItem: any = null;
    // Offset of the current text block within the merged text; citations are per-block.
    let textOffset = 0;

    // Single pass preserving Anthropic's generation order, which matches OpenAI's
    // typical ordering (reasoning → web_search_call → message → function_call).
    for (const block of message.content) {
      if (block.type === "thinking") {
        output.push({
          type: "reasoning",
          id: `rs_${this.generateId()}`,
          summary: [{ type: "summary_text", text: block.thinking }],
        });
      } else if (block.type === "server_tool_use" && block.name === "web_search") {
        const input = (block as any).input as any;
        const query = typeof input === "object" && input !== null ? String(input.query ?? "") : "";
        output.push({
          type: "web_search_call",
          id: `ws_${this.generateId()}`,
          status: "completed",
          action: { type: "search", query },
        });
      } else if (block.type === "tool_use" || block.type === "server_tool_use") {
        output.push({
          type: "function_call",
          id: `fc_${this.generateId()}`,
          call_id: block.id,
          name: block.name,
          arguments: JSON.stringify(block.input),
          status: "completed",
        });
      } else if (block.type === "text") {
        textParts.push(block.text);
        const citations = (block as any).citations as any[] | null | undefined;
        if (citations) {
          for (const c of citations) {
            annotations.push(this.citationToAnnotation(c, textOffset));
          }
        }
        // Multiple text blocks merge into one message item at the first text block's position.
        if (!messageItem) {
          messageItem = {
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
          };
          output.push(messageItem);
        }
        textOffset += block.text.length;
      }
      // web_search_tool_result and other result blocks: skipped; the corresponding
      // server_tool_use block already produced the web_search_call item.
    }

    if (messageItem) {
      messageItem.content[0].text = textParts.join("");
      messageItem.content[0].annotations = annotations;
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
    // Responses input_tokens is inclusive of cached tokens (cached ⊆ input), but Anthropic's
    // input_tokens counts only uncached tokens, with cache read/creation as separate buckets.
    const cacheRead = usage.cache_read_input_tokens ?? 0;
    const totalInputTokens =
      usage.input_tokens + cacheRead + (usage.cache_creation_input_tokens ?? 0);

    return {
      id: message.id,
      object: "response",
      created_at: Math.floor(Date.now() / 1000),
      model: message.model as string,
      output,
      status,
      // refusal → status "failed"; surface an error so clients don't read it as a normal completion.
      error:
        status === "failed"
          ? { code: "content_filter", message: "Content refused by the model." }
          : null,
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
        input_tokens: totalInputTokens,
        output_tokens: usage.output_tokens,
        total_tokens: totalInputTokens + usage.output_tokens,
        input_tokens_details: {
          cached_tokens: cacheRead,
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
    try {
      for await (const event of stream) {
        const events = this.convertStreamEvent(event);
        for (const e of events) {
          yield e;
        }
      }
    } catch (err) {
      // Only Anthropic API errors are converted to OpenAI-style failure events;
      // unexpected errors (programming bugs, network layer, etc.) propagate.
      if (!(err instanceof APIError)) {
        throw err;
      }
      const message = err.message;
      const code: "server_error" | "rate_limit_exceeded" =
        err.status === 429 ? "rate_limit_exceeded" : "server_error";

      const resp = this.makeSkeletonResponse();
      resp.status = "failed" as any;
      resp.error = { code, message } as any;

      yield {
        type: "response.failed",
        response: resp,
        sequence_number: this.streamState.seq++,
      } as RespStreamEvent;
      yield {
        type: "error",
        code,
        message,
        param: null,
        sequence_number: this.streamState.seq++,
      } as RespStreamEvent;
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
          state.currentBlockType = "thinking";
          const reasoningItemId = `rs_${this.generateId()}`;
          state.reasoningItemId = reasoningItemId;
          state.reasoningOutputIndex = state.nextOutputIndex++;
          // Reset per-item so done events don't carry text from prior reasoning items.
          state.reasoningText = "";
          events.push({
            type: "response.output_item.added",
            item: {
              type: "reasoning",
              id: reasoningItemId,
              summary: [],
            },
            output_index: state.reasoningOutputIndex,
            sequence_number: state.seq++,
          } as RespStreamEvent);
          events.push({
            type: "response.reasoning_summary_part.added",
            item_id: reasoningItemId,
            output_index: state.reasoningOutputIndex,
            summary_index: 0,
            part: { type: "summary_text", text: "" },
            sequence_number: state.seq++,
          } as RespStreamEvent);
        } else if (block.type === "server_tool_use" && block.name === "web_search") {
          state.webSearchCallCount++;
          const wsOutputIndex = state.nextOutputIndex++;
          const wsItemId = `ws_${this.generateId()}`;
          // web_search_tool_result arrives as a separate later block; key by tool_use_id to pair them.
          state.webSearchByToolUseId.set(block.id, {
            itemId: wsItemId,
            outputIndex: wsOutputIndex,
            query: "",
            done: false,
          });
          state.webSearchToolUseIdByBlockIndex.set(event.index, block.id);
          state.currentBlockType = "server_tool_use_web_search";

          events.push({
            type: "response.output_item.added",
            item: {
              type: "web_search_call",
              id: wsItemId,
              status: "in_progress",
              action: { type: "search", query: "" },
            },
            output_index: wsOutputIndex,
            sequence_number: state.seq++,
          } as RespStreamEvent);
          events.push({
            type: "response.web_search_call.in_progress",
            item_id: wsItemId,
            output_index: wsOutputIndex,
            sequence_number: state.seq++,
          } as RespStreamEvent);
          // searching fires once when the search begins, not on every query delta.
          events.push({
            type: "response.web_search_call.searching",
            item_id: wsItemId,
            output_index: wsOutputIndex,
            sequence_number: state.seq++,
          } as RespStreamEvent);
        } else if (block.type === "tool_use" || block.type === "server_tool_use") {
          state.toolCallCount++;
          const toolCallItemId = `fc_${this.generateId()}`;
          const toolOutputIndex = state.nextOutputIndex++;
          state.toolCalls.push({
            id: toolCallItemId,
            call_id: block.id,
            name: block.name,
            arguments: "",
            outputIndex: toolOutputIndex,
            done: false,
          });
          state.currentBlockType = "tool_use";

          events.push({
            type: "response.output_item.added",
            item: {
              type: "function_call",
              id: toolCallItemId,
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
            const textMessageItemId = `msg_${this.generateId()}`;
            state.textMessageItemId = textMessageItemId;
            state.textMessageOutputIndex = state.nextOutputIndex++;
            state.textBlockStartOffset = 0;

            events.push({
              type: "response.output_item.added",
              item: {
                type: "message",
                id: textMessageItemId,
                role: "assistant",
                status: "in_progress",
                content: [],
              },
              output_index: state.textMessageOutputIndex,
              sequence_number: state.seq++,
            } as RespStreamEvent);
            events.push({
              type: "response.content_part.added",
              item_id: textMessageItemId,
              output_index: state.textMessageOutputIndex,
              content_index: 0,
              part: { type: "output_text", text: "", annotations: [] },
              sequence_number: state.seq++,
            } as RespStreamEvent);
          } else {
            // Subsequent text block — its citations are relative to this block's text,
            // so shift by the accumulated length of prior merged blocks.
            state.textBlockStartOffset = state.textMessageText.length;
          }
          state.currentBlockType = "text";
        } else if (block.type === "web_search_tool_result") {
          state.webSearchCount++;
          state.currentBlockType = "web_search_tool_result";
          const entry = state.webSearchByToolUseId.get((block as any).tool_use_id as string);
          if (entry && !entry.done) {
            entry.done = true;
            const query = this.extractWebSearchQuery(entry.query);
            events.push({
              type: "response.web_search_call.completed",
              item_id: entry.itemId,
              output_index: entry.outputIndex,
              sequence_number: state.seq++,
            } as RespStreamEvent);
            const wsItem = {
              type: "web_search_call",
              id: entry.itemId,
              status: "completed",
              action: { type: "search", query },
            };
            events.push({
              type: "response.output_item.done",
              item: wsItem,
              output_index: entry.outputIndex,
              sequence_number: state.seq++,
            } as RespStreamEvent);
            state.outputItems.push({ index: entry.outputIndex, item: wsItem });
          }
        }
        break;
      }

      case "content_block_delta": {
        const delta = event.delta;

        if (delta.type === "thinking_delta") {
          state.reasoningText += delta.thinking;
          events.push({
            type: "response.reasoning_summary_text.delta",
            item_id: state.reasoningItemId,
            output_index: state.reasoningOutputIndex,
            summary_index: 0,
            delta: delta.thinking,
            sequence_number: state.seq++,
          } as RespStreamEvent);
        } else if (delta.type === "input_json_delta") {
          // Same delta type carries tool_use args and web_search query; route by current block.
          if (state.currentBlockType === "server_tool_use_web_search") {
            const toolUseId = state.webSearchToolUseIdByBlockIndex.get(event.index);
            const entry = toolUseId ? state.webSearchByToolUseId.get(toolUseId) : undefined;
            if (entry) {
              entry.query += delta.partial_json;
            }
            // searching already emitted at content_block_start; deltas only accumulate query.
          } else {
            const lastToolCall = state.toolCalls[state.toolCalls.length - 1];
            if (lastToolCall) {
              lastToolCall.arguments += delta.partial_json;
              events.push({
                type: "response.function_call_arguments.delta",
                item_id: lastToolCall.id,
                output_index: lastToolCall.outputIndex,
                delta: delta.partial_json,
                sequence_number: state.seq++,
              } as RespStreamEvent);
            }
          }
        } else if (delta.type === "text_delta") {
          state.textMessageText += delta.text;
          events.push({
            type: "response.output_text.delta",
            item_id: state.textMessageItemId,
            output_index: state.textMessageOutputIndex,
            content_index: 0,
            delta: delta.text,
            sequence_number: state.seq++,
          } as RespStreamEvent);
        } else if (delta.type === "citations_delta") {
          // Anthropic streams citations as deltas; collect and emit on the message's done events.
          const citation = (delta as any).citation;
          if (citation) {
            state.textAnnotations.push(
              this.citationToAnnotation(citation, state.textBlockStartOffset)
            );
          }
        }
        break;
      }

      case "content_block_stop": {
        const finishedType = state.currentBlockType;
        state.currentBlockType = "";

        if (finishedType === "thinking") {
          events.push({
            type: "response.reasoning_summary_text.done",
            item_id: state.reasoningItemId,
            output_index: state.reasoningOutputIndex,
            summary_index: 0,
            text: state.reasoningText,
            sequence_number: state.seq++,
          } as RespStreamEvent);
          events.push({
            type: "response.reasoning_summary_part.done",
            item_id: state.reasoningItemId,
            output_index: state.reasoningOutputIndex,
            summary_index: 0,
            part: { type: "summary_text", text: state.reasoningText },
            sequence_number: state.seq++,
          } as RespStreamEvent);
          events.push({
            type: "response.output_item.done",
            item: {
              type: "reasoning",
              id: state.reasoningItemId,
              summary: [{ type: "summary_text", text: state.reasoningText }],
            },
            output_index: state.reasoningOutputIndex,
            sequence_number: state.seq++,
          } as RespStreamEvent);
          state.outputItems.push({
            index: state.reasoningOutputIndex,
            item: {
              type: "reasoning",
              id: state.reasoningItemId,
              summary: [{ type: "summary_text", text: state.reasoningText }],
            },
          });
        } else if (finishedType === "tool_use") {
          const lastToolCall = state.toolCalls[state.toolCalls.length - 1];
          if (lastToolCall) {
            lastToolCall.done = true;
            events.push({
              type: "response.function_call_arguments.done",
              item_id: lastToolCall.id,
              output_index: lastToolCall.outputIndex,
              arguments: lastToolCall.arguments,
              sequence_number: state.seq++,
            } as RespStreamEvent);
            const toolItem = {
              type: "function_call",
              id: lastToolCall.id,
              call_id: lastToolCall.call_id,
              name: lastToolCall.name,
              arguments: lastToolCall.arguments,
              status: "completed" as const,
            };
            events.push({
              type: "response.output_item.done",
              item: toolItem,
              output_index: lastToolCall.outputIndex,
              sequence_number: state.seq++,
            } as RespStreamEvent);
            state.outputItems.push({ index: lastToolCall.outputIndex, item: toolItem });
          }
        }
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
        // Pass a synthetic content array when any tool_call completed its block, so the
        // streaming path matches non-streaming behavior (max_tokens + complete tool_use
        // → "completed", not "incomplete").
        const hasCompleteToolCall = state.toolCalls.some(t => t.done);
        const status = this.stopReasonToStatus(
          state.stopReason as any,
          hasCompleteToolCall ? ([{ type: "tool_use" }] as any) : undefined
        );
        const resp = this.makeSkeletonResponse();
        resp.status = status as any;
        // Responses input_tokens is inclusive of cached tokens (cached ⊆ input), but Anthropic's
        // input_tokens counts only uncached tokens, with cache read/creation as separate buckets.
        const totalInputTokens =
          state.inputTokens + state.cacheReadTokens + state.cacheCreationTokens;
        resp.usage = {
          input_tokens: totalInputTokens,
          output_tokens: state.outputTokens,
          total_tokens: totalInputTokens + state.outputTokens,
          input_tokens_details: {
            cached_tokens: state.cacheReadTokens,
          },
          output_tokens_details: {
            reasoning_tokens: 0,
          },
        };

        // Text may span multiple Anthropic blocks; emit the message's done events once at the end.
        if (state.textMessageStarted) {
          events.push({
            type: "response.output_text.done",
            item_id: state.textMessageItemId,
            output_index: state.textMessageOutputIndex,
            content_index: 0,
            text: state.textMessageText,
            sequence_number: state.seq++,
          } as RespStreamEvent);
          events.push({
            type: "response.content_part.done",
            item_id: state.textMessageItemId,
            output_index: state.textMessageOutputIndex,
            content_index: 0,
            part: {
              type: "output_text",
              text: state.textMessageText,
              annotations: state.textAnnotations,
            },
            sequence_number: state.seq++,
          } as RespStreamEvent);
          const messageItem = {
            type: "message",
            id: state.textMessageItemId,
            role: "assistant",
            status: "completed",
            content: [
              {
                type: "output_text",
                text: state.textMessageText,
                annotations: state.textAnnotations,
                logprobs: null as any,
              },
            ],
          };
          events.push({
            type: "response.output_item.done",
            item: messageItem,
            output_index: state.textMessageOutputIndex,
            sequence_number: state.seq++,
          } as RespStreamEvent);
          state.outputItems.push({ index: state.textMessageOutputIndex, item: messageItem });
        }

        // Truncated tool_calls (no content_block_stop) still belong in output with incomplete status.
        for (const tc of state.toolCalls) {
          if (!tc.done) {
            state.outputItems.push({
              index: tc.outputIndex,
              item: {
                type: "function_call",
                id: tc.id,
                call_id: tc.call_id,
                name: tc.name,
                arguments: tc.arguments,
                status: "incomplete",
              },
            });
          }
        }

        // Truncated web_search_calls (web_search_tool_result never arrived) still need a
        // completed + output_item.done so clients see the item finalize.
        for (const entry of state.webSearchByToolUseId.values()) {
          if (entry.done) continue;
          entry.done = true;
          const query = this.extractWebSearchQuery(entry.query);
          const wsStatus = status === "incomplete" ? "incomplete" : "completed";
          events.push({
            type: "response.web_search_call.completed",
            item_id: entry.itemId,
            output_index: entry.outputIndex,
            sequence_number: state.seq++,
          } as RespStreamEvent);
          const wsItem = {
            type: "web_search_call",
            id: entry.itemId,
            status: wsStatus,
            action: { type: "search", query },
          };
          events.push({
            type: "response.output_item.done",
            item: wsItem,
            output_index: entry.outputIndex,
            sequence_number: state.seq++,
          } as RespStreamEvent);
          state.outputItems.push({ index: entry.outputIndex, item: wsItem });
        }

        // Items accumulate in completion order, not output order (e.g. text finalizes here,
        // after an earlier-emitted tool_use with a higher index). Sort by index to match
        // the output_index sequence clients saw during streaming.
        resp.output = [...state.outputItems].sort((a, b) => a.index - b.index).map(o => o.item);

        if (status === "incomplete") {
          events.push({
            type: "response.incomplete",
            response: resp,
            sequence_number: state.seq++,
          } as RespStreamEvent);
        } else if (status === "failed") {
          // refusal → mirror the catch block's failure shape so clients see a terminal
          // response.failed + error instead of a response.completed carrying status "failed".
          const error = { code: "content_filter", message: "Content refused by the model." };
          resp.error = error as any;
          events.push({
            type: "response.failed",
            response: resp,
            sequence_number: state.seq++,
          } as RespStreamEvent);
          events.push({
            type: "error",
            code: error.code,
            message: error.message,
            param: null,
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

    // Buffer assistant text and tool_use blocks together so a single assistant turn
    // (text + following function_calls) lands in one message, as Anthropic requires.
    const pendingAssistantBlocks: Anthropic.ContentBlockParam[] = [];

    const flushAssistant = () => {
      if (pendingAssistantBlocks.length > 0) {
        messages.push({
          role: "assistant",
          content: [...pendingAssistantBlocks],
        });
        pendingAssistantBlocks.length = 0;
      }
    };

    for (const item of input!) {
      const typed = item as any;

      if (typed.type === "message" || !typed.type) {
        if (typed.role === "system" || typed.role === "developer") {
          flushAssistant();
          const text = this.extractText(typed.content);
          if (text) systemBlocks.push(text);
        } else if (typed.role === "user") {
          flushAssistant();
          const content = this.convertInputContent(typed.content);
          messages.push({ role: "user", content });
        } else if (typed.role === "assistant") {
          // Append to pending; may merge with following function_call tool_use blocks.
          if (typeof typed.content === "string") {
            pendingAssistantBlocks.push({ type: "text", text: typed.content });
          } else if (Array.isArray(typed.content)) {
            for (const part of typed.content) {
              if (part.type === "output_text") {
                pendingAssistantBlocks.push({ type: "text", text: part.text });
              }
            }
          }
        }
      } else if (typed.type === "function_call") {
        let parsedInput: Record<string, unknown>;
        try {
          parsedInput = JSON.parse(typed.arguments);
        } catch {
          parsedInput = {};
        }
        pendingAssistantBlocks.push({
          type: "tool_use",
          id: typed.call_id,
          name: typed.name,
          input: parsedInput,
        });
      } else if (typed.type === "function_call_output") {
        flushAssistant();
        const output = typed.output;
        // Array content-part outputs are lossy: serialized as raw JSON rather than
        // unwrapping input_text/image/file parts. Acceptable since Anthropic tool_result
        // text expects a string and array outputs are rare on this path.
        const outputText =
          typeof output === "string" ? output : output == null ? "" : JSON.stringify(output);
        const toolResult: Anthropic.ToolResultBlockParam = {
          type: "tool_result",
          tool_use_id: typed.call_id,
          content: [{ type: "text", text: outputText }],
        };
        // Anthropic requires parallel tool_results grouped in one user message,
        // so append to the prior user message if it already holds tool_results.
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
        const role = typed.role;
        if (role === "system" || role === "developer") {
          flushAssistant();
          systemBlocks.push(typed.content);
        } else if (role === "user") {
          flushAssistant();
          messages.push({ role: "user", content: [{ type: "text", text: typed.content }] });
        } else if (role === "assistant") {
          pendingAssistantBlocks.push({ type: "text", text: typed.content });
        }
      }
    }

    flushAssistant();
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
        result.push({ type: "web_search_20250305", name: "web_search" } as any);
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
          // Anthropic's name for "required" is "any".
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
    reasoning: OpenAI.Responses.ResponseCreateParams["reasoning"],
    maxOutputTokens: number
  ): Anthropic.ThinkingConfigParam | undefined {
    if (!reasoning || !reasoning.effort) {
      return undefined;
    } else if (reasoning.effort === "none") {
      return { type: "disabled" };
    }

    const effortRatioMap: Record<string, number> = {
      minimal: 0.1,
      low: 0.2,
      medium: 0.5,
      high: 0.8,
      xhigh: 0.9,
      max: 0.95,
    };

    const ratio = effortRatioMap[reasoning.effort] ?? effortRatioMap.minimal;
    const budget = Math.floor(maxOutputTokens * ratio);
    return { type: "enabled", budget_tokens: Math.max(budget, 1024) };
  }

  // --- Private: response helpers ---

  private stopReasonToStatus(
    stopReason: Anthropic.Message["stop_reason"] | string | null,
    content?: Anthropic.ContentBlock[]
  ): RespResponse["status"] {
    // Any tool_use in content means the turn ended for tool dispatch, not incompleteness.
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

  // Parses the query from accumulated web_search input_json_delta. Handles both complete
  // JSON (`{"query":"x"}`) and partial JSON truncated by max_tokens (`{"query":"lat`).
  private extractWebSearchQuery(raw: string): string {
    if (!raw) return "";
    try {
      const parsed = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null && typeof parsed.query === "string") {
        return parsed.query;
      }
    } catch {
      // partial JSON; fall through to regex
    }
    const match = raw.match(/"query"\s*:\s*"((?:[^"\\]|\\.)*)/);
    return match ? match[1].replace(/\\"/g, '"').replace(/\\\\/g, "\\") : "";
  }

  // Maps an Anthropic citation to an OpenAI output_text annotation. Anthropic's
  // web_search_result citation maps to OpenAI's url_citation; indices are shifted by
  // `offset` when merging multiple text blocks into one output_text.
  private citationToAnnotation(citation: any, offset: number = 0): any {
    const start = (citation.start_index ?? 0) + offset;
    const end = (citation.end_index ?? 0) + offset;
    return {
      type: "url_citation",
      start_index: start,
      end_index: end,
      url: citation.url ?? "",
      title: citation.title ?? "",
    };
  }

  // --- Private: stream helpers ---

  private createStreamState(): StreamState {
    return {
      id: "",
      model: "",
      seq: 0,
      nextOutputIndex: 0,
      reasoningItemId: "",
      reasoningOutputIndex: 0,
      reasoningText: "",
      toolCallCount: 0,
      toolCalls: [],
      webSearchCallCount: 0,
      webSearchByToolUseId: new Map(),
      webSearchToolUseIdByBlockIndex: new Map(),
      currentBlockType: "",
      textMessageStarted: false,
      textMessageItemId: "",
      textMessageOutputIndex: 0,
      textMessageText: "",
      textAnnotations: [],
      textBlockStartOffset: 0,
      outputItems: [],
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
