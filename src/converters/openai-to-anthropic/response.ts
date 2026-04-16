import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';

export function convertOpenAIResponseToAnthropic(
  response: OpenAI.ChatCompletion,
): Anthropic.Message {
  const choice = response.choices[0];
  const msg = choice?.message;

  const content: Anthropic.ContentBlock[] = [];

  if (msg?.content) {
    content.push({ type: 'text', text: msg.content, citations: null });
  }

  if (msg?.tool_calls) {
    for (const tc of msg.tool_calls) {
      if (tc.type === 'function') {
        content.push({
          type: 'tool_use',
          id: tc.id,
          name: tc.function.name,
          input: JSON.parse(tc.function.arguments || '{}'),
          caller: { type: 'direct' },
        });
      }
    }
  }

  if (content.length === 0) {
    content.push({ type: 'text', text: '', citations: null });
  }

  return {
    id: response.id,
    type: 'message',
    role: 'assistant',
    model: response.model as Anthropic.Model,
    content,
    stop_reason: mapFinishReason(choice?.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: response.usage?.prompt_tokens ?? 0,
      output_tokens: response.usage?.completion_tokens ?? 0,
      cache_creation_input_tokens: null,
      cache_read_input_tokens: null,
      cache_creation: null,
      inference_geo: null,
      server_tool_use: null,
      service_tier: null,
    },
    container: null,
  };
}

function mapFinishReason(
  reason: OpenAI.ChatCompletion.Choice['finish_reason'] | undefined,
): Anthropic.Message['stop_reason'] {
  switch (reason) {
    case 'stop':
      return 'end_turn';
    case 'tool_calls':
      return 'tool_use';
    case 'length':
      return 'max_tokens';
    case 'content_filter':
    case 'function_call':
    default:
      return 'end_turn';
  }
}
