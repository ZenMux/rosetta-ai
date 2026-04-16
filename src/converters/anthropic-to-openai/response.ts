import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';

export function convertAnthropicResponseToOpenAI(
  message: Anthropic.Message,
): OpenAI.ChatCompletion {
  const textParts: string[] = [];
  const toolCalls: OpenAI.ChatCompletionMessageToolCall[] = [];

  for (const block of message.content) {
    if (block.type === 'text') {
      textParts.push(block.text);
    } else if (block.type === 'tool_use') {
      toolCalls.push({
        id: block.id,
        type: 'function',
        function: {
          name: block.name,
          arguments: JSON.stringify(block.input),
        },
      });
    }
  }

  const assistantMessage: OpenAI.ChatCompletionMessage = {
    role: 'assistant',
    content: textParts.length > 0 ? textParts.join('\n') : null,
    refusal: null,
  };

  if (toolCalls.length > 0) {
    assistantMessage.tool_calls = toolCalls;
  }

  const promptTokens = message.usage.input_tokens;
  const completionTokens = message.usage.output_tokens;

  return {
    id: message.id,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: message.model,
    choices: [
      {
        index: 0,
        message: assistantMessage,
        finish_reason: mapStopReason(message.stop_reason),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
  };
}

function mapStopReason(
  reason: Anthropic.Message['stop_reason'],
): OpenAI.ChatCompletion.Choice['finish_reason'] {
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
