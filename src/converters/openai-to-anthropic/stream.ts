import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';

interface StreamState {
  messageStarted: boolean;
  currentBlockIndex: number;
  currentBlockType: 'text' | 'tool_use' | null;
  id: string;
  model: string;
  toolCallIndexMap: Map<number, number>; // openai tool_call index -> anthropic block index
}

export function createOpenAIToAnthropicStreamConverter(): (
  chunk: OpenAI.ChatCompletionChunk,
) => Anthropic.RawMessageStreamEvent[] {
  const state: StreamState = {
    messageStarted: false,
    currentBlockIndex: -1,
    currentBlockType: null,
    id: '',
    model: '',
    toolCallIndexMap: new Map(),
  };

  return (chunk: OpenAI.ChatCompletionChunk): Anthropic.RawMessageStreamEvent[] => {
    const events: Anthropic.RawMessageStreamEvent[] = [];
    const choice = chunk.choices[0];
    if (!choice) return events;

    state.id = chunk.id;
    state.model = chunk.model;

    const delta = choice.delta;

    // First chunk with role - emit message_start
    if (!state.messageStarted && delta.role === 'assistant') {
      state.messageStarted = true;
      events.push({
        type: 'message_start',
        message: {
          id: chunk.id,
          type: 'message',
          role: 'assistant',
          model: chunk.model as Anthropic.Model,
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

      // Start a text block by default
      state.currentBlockIndex = 0;
      state.currentBlockType = 'text';
      events.push({
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '', citations: null },
      });
    }

    // Text content delta
    if (delta.content) {
      events.push({
        type: 'content_block_delta',
        index: state.currentBlockIndex,
        delta: { type: 'text_delta', text: delta.content },
      });
    }

    // Tool call deltas
    if (delta.tool_calls) {
      for (const tc of delta.tool_calls) {
        if (tc.id && tc.function?.name) {
          // New tool call starting - close previous block
          if (state.currentBlockType !== null) {
            events.push({
              type: 'content_block_stop',
              index: state.currentBlockIndex,
            });
          }

          state.currentBlockIndex++;
          state.currentBlockType = 'tool_use';
          state.toolCallIndexMap.set(tc.index, state.currentBlockIndex);

          events.push({
            type: 'content_block_start',
            index: state.currentBlockIndex,
            content_block: {
              type: 'tool_use',
              id: tc.id,
              name: tc.function.name,
              input: {},
              caller: { type: 'direct' },
            },
          });

          // If there are initial arguments, emit them
          if (tc.function.arguments) {
            events.push({
              type: 'content_block_delta',
              index: state.currentBlockIndex,
              delta: {
                type: 'input_json_delta',
                partial_json: tc.function.arguments,
              },
            });
          }
        } else if (tc.function?.arguments) {
          // Continuation of existing tool call arguments
          const blockIndex = state.toolCallIndexMap.get(tc.index) ?? state.currentBlockIndex;
          events.push({
            type: 'content_block_delta',
            index: blockIndex,
            delta: {
              type: 'input_json_delta',
              partial_json: tc.function.arguments,
            },
          });
        }
      }
    }

    // Finish reason - emit stop events
    if (choice.finish_reason) {
      if (state.currentBlockType !== null) {
        events.push({
          type: 'content_block_stop',
          index: state.currentBlockIndex,
        });
      }

      events.push({
        type: 'message_delta',
        delta: {
          stop_reason: mapFinishReason(choice.finish_reason),
          stop_sequence: null,
          container: null,
        },
        usage: {
          output_tokens: 0,
          input_tokens: null,
          cache_creation_input_tokens: null,
          cache_read_input_tokens: null,
          server_tool_use: null,
        },
      });

      events.push({ type: 'message_stop' });
    }

    return events;
  };
}

function mapFinishReason(
  reason: string,
): Anthropic.Message['stop_reason'] {
  switch (reason) {
    case 'stop':
      return 'end_turn';
    case 'tool_calls':
      return 'tool_use';
    case 'length':
      return 'max_tokens';
    default:
      return 'end_turn';
  }
}
