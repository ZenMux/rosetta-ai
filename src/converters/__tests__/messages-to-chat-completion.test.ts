import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';
import { MessagesToChatCompletionConverter } from '../messages-to-chat-completion';

const converter = new MessagesToChatCompletionConverter();

// ─── convertRequest ────────────────────────────────────────────────

describe('MessagesToChatCompletionConverter.convertRequest', () => {
  it('converts system string to system message', () => {
    const result = converter.convertRequest({
      model: 'claude-sonnet-4-20250514', max_tokens: 1024,
      system: 'You are helpful.',
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(result.messages[0]).toEqual({ role: 'system', content: 'You are helpful.' });
    expect(result.messages[1]).toEqual({ role: 'user', content: 'Hello' });
  });

  it('converts system content blocks to system message', () => {
    const result = converter.convertRequest({
      model: 'claude-sonnet-4-20250514', max_tokens: 1024,
      system: [{ type: 'text', text: 'You are helpful.' }, { type: 'text', text: 'Be concise.' }],
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(result.messages[0]).toEqual({ role: 'system', content: 'You are helpful.\nBe concise.' });
  });

  it('converts user content blocks to string when text only', () => {
    const result = converter.convertRequest({
      model: 'claude-sonnet-4-20250514', max_tokens: 1024,
      messages: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
    });

    expect(result.messages[0]).toEqual({ role: 'user', content: 'Hello' });
  });

  describe('tool calling', () => {
    it('converts tool_use blocks to assistant tool_calls', () => {
      const result = converter.convertRequest({
        model: 'claude-sonnet-4-20250514', max_tokens: 1024,
        messages: [
          { role: 'user', content: 'What is the weather?' },
          {
            role: 'assistant',
            content: [{ type: 'tool_use', id: 'toolu_123', name: 'get_weather', input: { location: 'SF' } }],
          },
          {
            role: 'user',
            content: [{ type: 'tool_result', tool_use_id: 'toolu_123', content: '72°F' }],
          },
        ],
      });

      expect(result.messages[1]).toEqual({
        role: 'assistant',
        tool_calls: [{ id: 'toolu_123', type: 'function', function: { name: 'get_weather', arguments: '{"location":"SF"}' } }],
      });
      expect(result.messages[2]).toEqual({ role: 'tool', tool_call_id: 'toolu_123', content: '72°F' });
    });

    it('splits user message with multiple tool_results', () => {
      const result = converter.convertRequest({
        model: 'claude-sonnet-4-20250514', max_tokens: 1024,
        messages: [
          { role: 'user', content: 'Do two things' },
          {
            role: 'assistant',
            content: [
              { type: 'tool_use', id: 'c1', name: 'fn1', input: {} },
              { type: 'tool_use', id: 'c2', name: 'fn2', input: {} },
            ],
          },
          {
            role: 'user',
            content: [
              { type: 'tool_result', tool_use_id: 'c1', content: 'result1' },
              { type: 'tool_result', tool_use_id: 'c2', content: 'result2' },
            ],
          },
        ],
      });

      expect(result.messages[2]).toEqual({ role: 'tool', tool_call_id: 'c1', content: 'result1' });
      expect(result.messages[3]).toEqual({ role: 'tool', tool_call_id: 'c2', content: 'result2' });
    });
  });

  describe('multimodal content', () => {
    it('converts image block with URL source', () => {
      const result = converter.convertRequest({
        model: 'claude-sonnet-4-20250514', max_tokens: 1024,
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'What is this?' },
            { type: 'image', source: { type: 'url', url: 'https://example.com/img.png' } },
          ],
        }],
      });

      const msg = result.messages[0] as OpenAI.ChatCompletionUserMessageParam;
      const content = msg.content as OpenAI.ChatCompletionContentPart[];
      expect(content[1]).toEqual({ type: 'image_url', image_url: { url: 'https://example.com/img.png' } });
    });

    it('converts image block with base64 source', () => {
      const result = converter.convertRequest({
        model: 'claude-sonnet-4-20250514', max_tokens: 1024,
        messages: [{
          role: 'user',
          content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgo=' } }],
        }],
      });

      const msg = result.messages[0] as OpenAI.ChatCompletionUserMessageParam;
      const content = msg.content as OpenAI.ChatCompletionContentPart[];
      expect(content[0]).toEqual({ type: 'image_url', image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' } });
    });
  });

  describe('params mapping', () => {
    it('maps temperature, top_p, max_tokens, stop_sequences', () => {
      const result = converter.convertRequest({
        model: 'claude-sonnet-4-20250514', max_tokens: 1000,
        messages: [{ role: 'user', content: 'Hi' }],
        temperature: 0.7, top_p: 0.9, stop_sequences: ['END', 'STOP'],
      });

      expect(result.temperature).toBe(0.7);
      expect(result.top_p).toBe(0.9);
      expect(result.max_tokens).toBe(1000);
      expect(result.stop).toEqual(['END', 'STOP']);
    });
  });

  describe('tools mapping', () => {
    it('converts Anthropic tools to OpenAI function tools', () => {
      const result = converter.convertRequest({
        model: 'claude-sonnet-4-20250514', max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [{
          name: 'get_weather', description: 'Get weather info',
          input_schema: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] },
        }],
      });

      expect(result.tools).toEqual([{
        type: 'function',
        function: {
          name: 'get_weather', description: 'Get weather info',
          parameters: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] },
        },
      }]);
    });

    it('maps tool_choice {type: "auto"} to "auto"', () => {
      const result = converter.convertRequest({
        model: 'claude-sonnet-4-20250514', max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tool_choice: { type: 'auto' },
      });
      expect(result.tool_choice).toBe('auto');
    });

    it('maps tool_choice {type: "any"} to "required"', () => {
      const result = converter.convertRequest({
        model: 'claude-sonnet-4-20250514', max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tool_choice: { type: 'any' },
      });
      expect(result.tool_choice).toBe('required');
    });

    it('maps tool_choice {type: "tool", name} to named choice', () => {
      const result = converter.convertRequest({
        model: 'claude-sonnet-4-20250514', max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tool_choice: { type: 'tool', name: 'get_weather' },
      });
      expect(result.tool_choice).toEqual({ type: 'function', function: { name: 'get_weather' } });
    });
  });
});

// ─── convertResponse ───────────────────────────────────────────────

function makeMessage(overrides: Partial<Anthropic.Message> = {}): Anthropic.Message {
  return {
    id: 'msg_123', type: 'message', role: 'assistant',
    model: 'claude-sonnet-4-20250514',
    content: [{ type: 'text', text: 'Hello!', citations: null }],
    stop_reason: 'end_turn', stop_sequence: null,
    usage: {
      input_tokens: 10, output_tokens: 5,
      cache_creation_input_tokens: null, cache_read_input_tokens: null,
      cache_creation: null, inference_geo: null, server_tool_use: null, service_tier: null,
    },
    container: null,
    ...overrides,
  };
}

describe('MessagesToChatCompletionConverter.convertResponse', () => {
  it('converts a basic text response', () => {
    const result = converter.convertResponse(makeMessage());

    expect(result.id).toBe('msg_123');
    expect(result.object).toBe('chat.completion');
    expect(result.choices[0].message.content).toBe('Hello!');
    expect(result.choices[0].finish_reason).toBe('stop');
    expect(result.usage?.prompt_tokens).toBe(10);
    expect(result.usage?.completion_tokens).toBe(5);
    expect(result.usage?.total_tokens).toBe(15);
  });

  it('converts a tool_use response', () => {
    const result = converter.convertResponse(makeMessage({
      content: [{ type: 'tool_use', id: 'toolu_1', name: 'get_weather', input: { location: 'SF' }, caller: { type: 'direct' } }],
      stop_reason: 'tool_use',
    }));

    expect(result.choices[0].finish_reason).toBe('tool_calls');
    expect(result.choices[0].message.tool_calls).toEqual([
      { id: 'toolu_1', type: 'function', function: { name: 'get_weather', arguments: '{"location":"SF"}' } },
    ]);
  });

  it('maps stop_reason "max_tokens" to finish_reason "length"', () => {
    const result = converter.convertResponse(makeMessage({ stop_reason: 'max_tokens' }));
    expect(result.choices[0].finish_reason).toBe('length');
  });

  it('joins multiple text blocks', () => {
    const result = converter.convertResponse(makeMessage({
      content: [
        { type: 'text', text: 'Hello', citations: null },
        { type: 'text', text: 'World', citations: null },
      ],
    }));
    expect(result.choices[0].message.content).toBe('Hello\nWorld');
  });
});

// ─── createStreamConverter ─────────────────────────────────────────

const baseUsage: Anthropic.Usage = {
  input_tokens: 10, output_tokens: 0,
  cache_creation_input_tokens: null, cache_read_input_tokens: null,
  cache_creation: null, inference_geo: null, server_tool_use: null, service_tier: null,
};

const baseDeltaUsage: Anthropic.MessageDeltaUsage = {
  output_tokens: 5, input_tokens: null,
  cache_creation_input_tokens: null, cache_read_input_tokens: null, server_tool_use: null,
};

function messageStart(): Anthropic.RawMessageStartEvent {
  return {
    type: 'message_start',
    message: {
      id: 'msg_123', type: 'message', role: 'assistant',
      model: 'claude-sonnet-4-20250514', content: [],
      stop_reason: null, stop_sequence: null, usage: baseUsage, container: null,
    },
  };
}

describe('MessagesToChatCompletionConverter.createStreamConverter', () => {
  it('emits first chunk with role on message_start', () => {
    const convert = converter.createStreamConverter();
    const result = convert(messageStart());

    expect(result).not.toBeNull();
    expect(result!.id).toBe('msg_123');
    expect(result!.choices[0].delta.role).toBe('assistant');
  });

  it('returns null for content_block_start (text)', () => {
    const convert = converter.createStreamConverter();
    convert(messageStart());
    const result = convert({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '', citations: null } });
    expect(result).toBeNull();
  });

  it('emits text content for text_delta', () => {
    const convert = converter.createStreamConverter();
    convert(messageStart());
    const result = convert({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello' } });

    expect(result).not.toBeNull();
    expect(result!.choices[0].delta.content).toBe('Hello');
  });

  it('emits tool_call start on content_block_start (tool_use)', () => {
    const convert = converter.createStreamConverter();
    convert(messageStart());
    const result = convert({
      type: 'content_block_start', index: 0,
      content_block: { type: 'tool_use', id: 'toolu_1', name: 'get_weather', input: {}, caller: { type: 'direct' } },
    });

    expect(result).not.toBeNull();
    const tc = result!.choices[0].delta.tool_calls![0];
    expect(tc.id).toBe('toolu_1');
    expect(tc.function!.name).toBe('get_weather');
  });

  it('emits tool_call arguments for input_json_delta', () => {
    const convert = converter.createStreamConverter();
    convert(messageStart());
    convert({
      type: 'content_block_start', index: 0,
      content_block: { type: 'tool_use', id: 'toolu_1', name: 'fn', input: {}, caller: { type: 'direct' } },
    });
    const result = convert({ type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"loc' } });

    expect(result!.choices[0].delta.tool_calls![0].function!.arguments).toBe('{"loc');
  });

  it('emits finish_reason on message_delta', () => {
    const convert = converter.createStreamConverter();
    convert(messageStart());
    const result = convert({
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null, container: null },
      usage: baseDeltaUsage,
    });

    expect(result!.choices[0].finish_reason).toBe('stop');
  });

  it('returns null for message_stop and content_block_stop', () => {
    const convert = converter.createStreamConverter();
    convert(messageStart());
    expect(convert({ type: 'message_stop' })).toBeNull();
    expect(convert({ type: 'content_block_stop', index: 0 })).toBeNull();
  });
});
