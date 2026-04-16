import type Anthropic from '@anthropic-ai/sdk';
import { convertAnthropicResponseToOpenAI } from '../response';

function makeMessage(overrides: Partial<Anthropic.Message> = {}): Anthropic.Message {
  return {
    id: 'msg_123',
    type: 'message',
    role: 'assistant',
    model: 'claude-sonnet-4-20250514',
    content: [{ type: 'text', text: 'Hello!', citations: null }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_creation_input_tokens: null,
      cache_read_input_tokens: null,
      cache_creation: null,
      inference_geo: null,
      server_tool_use: null,
      service_tier: null,
    },
    container: null,
    ...overrides,
  };
}

describe('convertAnthropicResponseToOpenAI', () => {
  it('converts a basic text response', () => {
    const input = makeMessage();
    const result = convertAnthropicResponseToOpenAI(input);

    expect(result.id).toBe('msg_123');
    expect(result.model).toBe('claude-sonnet-4-20250514');
    expect(result.object).toBe('chat.completion');
    expect(result.choices[0].message.role).toBe('assistant');
    expect(result.choices[0].message.content).toBe('Hello!');
    expect(result.choices[0].finish_reason).toBe('stop');
    expect(result.usage?.prompt_tokens).toBe(10);
    expect(result.usage?.completion_tokens).toBe(5);
    expect(result.usage?.total_tokens).toBe(15);
  });

  it('converts a tool_use response', () => {
    const input = makeMessage({
      content: [
        {
          type: 'tool_use',
          id: 'toolu_123',
          name: 'get_weather',
          input: { location: 'SF' },
          caller: { type: 'direct' },
        },
      ],
      stop_reason: 'tool_use',
    });

    const result = convertAnthropicResponseToOpenAI(input);

    expect(result.choices[0].finish_reason).toBe('tool_calls');
    expect(result.choices[0].message.content).toBeNull();
    expect(result.choices[0].message.tool_calls).toEqual([
      {
        id: 'toolu_123',
        type: 'function',
        function: {
          name: 'get_weather',
          arguments: '{"location":"SF"}',
        },
      },
    ]);
  });

  it('converts mixed text + tool_use response', () => {
    const input = makeMessage({
      content: [
        { type: 'text', text: 'Let me check.', citations: null },
        {
          type: 'tool_use',
          id: 'toolu_1',
          name: 'get_weather',
          input: {},
          caller: { type: 'direct' },
        },
      ],
      stop_reason: 'tool_use',
    });

    const result = convertAnthropicResponseToOpenAI(input);

    expect(result.choices[0].message.content).toBe('Let me check.');
    expect(result.choices[0].message.tool_calls).toEqual([
      {
        id: 'toolu_1',
        type: 'function',
        function: { name: 'get_weather', arguments: '{}' },
      },
    ]);
  });

  it('maps stop_reason "end_turn" to finish_reason "stop"', () => {
    const result = convertAnthropicResponseToOpenAI(makeMessage({ stop_reason: 'end_turn' }));
    expect(result.choices[0].finish_reason).toBe('stop');
  });

  it('maps stop_reason "max_tokens" to finish_reason "length"', () => {
    const result = convertAnthropicResponseToOpenAI(makeMessage({ stop_reason: 'max_tokens' }));
    expect(result.choices[0].finish_reason).toBe('length');
  });

  it('maps stop_reason "tool_use" to finish_reason "tool_calls"', () => {
    const result = convertAnthropicResponseToOpenAI(makeMessage({ stop_reason: 'tool_use' }));
    expect(result.choices[0].finish_reason).toBe('tool_calls');
  });

  it('joins multiple text blocks', () => {
    const input = makeMessage({
      content: [
        { type: 'text', text: 'Hello', citations: null },
        { type: 'text', text: 'World', citations: null },
      ],
    });

    const result = convertAnthropicResponseToOpenAI(input);
    expect(result.choices[0].message.content).toBe('Hello\nWorld');
  });
});
