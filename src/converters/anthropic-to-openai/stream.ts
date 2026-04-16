import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';

interface StreamState {
  id: string;
  model: string;
  toolCallCounter: number;
}

export function createAnthropicToOpenAIStreamConverter(): (
  event: Anthropic.RawMessageStreamEvent,
) => OpenAI.ChatCompletionChunk | null {
  const state: StreamState = {
    id: '',
    model: '',
    toolCallCounter: 0,
  };

  return (event: Anthropic.RawMessageStreamEvent): OpenAI.ChatCompletionChunk | null => {
    switch (event.type) {
      case 'message_start':
        return handleMessageStart(state, event);
      case 'content_block_start':
        return handleContentBlockStart(state, event);
      case 'content_block_delta':
        return handleContentBlockDelta(state, event);
      case 'content_block_stop':
        return null;
      case 'message_delta':
        return handleMessageDelta(state, event);
      case 'message_stop':
        return null;
      default:
        return null;
    }
  };
}

function makeChunk(
  state: StreamState,
  delta: OpenAI.ChatCompletionChunk.Choice.Delta,
  finish_reason: OpenAI.ChatCompletionChunk.Choice['finish_reason'] = null,
): OpenAI.ChatCompletionChunk {
  return {
    id: state.id,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: state.model,
    choices: [
      {
        index: 0,
        delta,
        finish_reason,
      },
    ],
  };
}

function handleMessageStart(
  state: StreamState,
  event: Anthropic.RawMessageStartEvent,
): OpenAI.ChatCompletionChunk {
  state.id = event.message.id;
  state.model = event.message.model;

  return makeChunk(state, { role: 'assistant' });
}

function handleContentBlockStart(
  state: StreamState,
  event: Anthropic.RawContentBlockStartEvent,
): OpenAI.ChatCompletionChunk | null {
  const block = event.content_block;

  if (block.type === 'tool_use') {
    const index = state.toolCallCounter++;
    return makeChunk(state, {
      tool_calls: [
        {
          index,
          id: block.id,
          type: 'function',
          function: {
            name: block.name,
            arguments: '',
          },
        },
      ],
    });
  }

  // Text block start - no OpenAI equivalent
  return null;
}

function handleContentBlockDelta(
  state: StreamState,
  event: Anthropic.RawContentBlockDeltaEvent,
): OpenAI.ChatCompletionChunk | null {
  const delta = event.delta;

  if (delta.type === 'text_delta') {
    return makeChunk(state, { content: delta.text });
  }

  if (delta.type === 'input_json_delta') {
    // Map back to the correct tool call index
    const toolIndex = state.toolCallCounter - 1;
    return makeChunk(state, {
      tool_calls: [
        {
          index: toolIndex,
          function: { arguments: delta.partial_json },
        },
      ],
    });
  }

  return null;
}

function handleMessageDelta(
  state: StreamState,
  event: Anthropic.RawMessageDeltaEvent,
): OpenAI.ChatCompletionChunk {
  return makeChunk(
    state,
    {},
    mapStopReason(event.delta.stop_reason),
  );
}

function mapStopReason(
  reason: Anthropic.StopReason | null,
): OpenAI.ChatCompletionChunk.Choice['finish_reason'] {
  switch (reason) {
    case 'end_turn':
    case 'stop_sequence':
      return 'stop';
    case 'tool_use':
      return 'tool_calls';
    case 'max_tokens':
      return 'length';
    default:
      return 'stop';
  }
}
