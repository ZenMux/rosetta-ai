import type OpenAI from 'openai';
import { convertOpenAIResponseToAnthropic } from '../response';

describe('convertOpenAIResponseToAnthropic', () => {
  it('converts a basic text response', () => {
    const input: OpenAI.ChatCompletion = {
      id: 'chatcmpl-123',
      object: 'chat.completion',
      created: 1700000000,
      model: 'gpt-4o',
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content: 'Hello!', refusal: null },
          finish_reason: 'stop',
          logprobs: null,
        },
      ],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    };

    const result = convertOpenAIResponseToAnthropic(input);

    expect(result.id).toBe('chatcmpl-123');
    expect(result.model).toBe('gpt-4o');
    expect(result.role).toBe('assistant');
    expect(result.type).toBe('message');
    expect(result.content).toEqual([{ type: 'text', text: 'Hello!', citations: null }]);
    expect(result.stop_reason).toBe('end_turn');
    expect(result.usage.input_tokens).toBe(10);
    expect(result.usage.output_tokens).toBe(5);
  });

  it('converts a tool call response', () => {
    const input: OpenAI.ChatCompletion = {
      id: 'chatcmpl-456',
      object: 'chat.completion',
      created: 1700000000,
      model: 'gpt-4o',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: null,
            refusal: null,
            tool_calls: [
              {
                id: 'call_abc',
                type: 'function',
                function: {
                  name: 'get_weather',
                  arguments: '{"location":"SF"}',
                },
              },
            ],
          },
          finish_reason: 'tool_calls',
          logprobs: null,
        },
      ],
    };

    const result = convertOpenAIResponseToAnthropic(input);

    expect(result.stop_reason).toBe('tool_use');
    expect(result.content).toEqual([
      {
        type: 'tool_use',
        id: 'call_abc',
        name: 'get_weather',
        input: { location: 'SF' },
        caller: { type: 'direct' },
      },
    ]);
  });

  it('converts mixed content + tool calls', () => {
    const input: OpenAI.ChatCompletion = {
      id: 'chatcmpl-789',
      object: 'chat.completion',
      created: 1700000000,
      model: 'gpt-4o',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: 'Let me check.',
            refusal: null,
            tool_calls: [
              {
                id: 'call_1',
                type: 'function',
                function: { name: 'get_weather', arguments: '{}' },
              },
            ],
          },
          finish_reason: 'tool_calls',
          logprobs: null,
        },
      ],
    };

    const result = convertOpenAIResponseToAnthropic(input);

    expect(result.content).toEqual([
      { type: 'text', text: 'Let me check.', citations: null },
      { type: 'tool_use', id: 'call_1', name: 'get_weather', input: {}, caller: { type: 'direct' } },
    ]);
  });

  it('maps finish_reason "length" to "max_tokens"', () => {
    const input: OpenAI.ChatCompletion = {
      id: 'id',
      object: 'chat.completion',
      created: 0,
      model: 'gpt-4o',
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content: 'truncated', refusal: null },
          finish_reason: 'length',
          logprobs: null,
        },
      ],
    };

    const result = convertOpenAIResponseToAnthropic(input);
    expect(result.stop_reason).toBe('max_tokens');
  });

  it('maps finish_reason "content_filter" to "end_turn"', () => {
    const input: OpenAI.ChatCompletion = {
      id: 'id',
      object: 'chat.completion',
      created: 0,
      model: 'gpt-4o',
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content: '', refusal: null },
          finish_reason: 'content_filter',
          logprobs: null,
        },
      ],
    };

    const result = convertOpenAIResponseToAnthropic(input);
    expect(result.stop_reason).toBe('end_turn');
  });

  it('handles null content with no tool calls', () => {
    const input: OpenAI.ChatCompletion = {
      id: 'id',
      object: 'chat.completion',
      created: 0,
      model: 'gpt-4o',
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content: null, refusal: null },
          finish_reason: 'stop',
          logprobs: null,
        },
      ],
    };

    const result = convertOpenAIResponseToAnthropic(input);
    expect(result.content).toEqual([{ type: 'text', text: '', citations: null }]);
  });
});
