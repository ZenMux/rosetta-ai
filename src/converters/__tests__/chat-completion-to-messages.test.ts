import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';
import { ChatCompletionToMessagesConverter } from '../chat-completion-to-messages';

const converter = new ChatCompletionToMessagesConverter();

// ─── convertRequest ────────────────────────────────────────────────

describe('ChatCompletionToMessagesConverter.convertRequest', () => {
  describe('basic text conversation', () => {
    it('converts system + user messages', () => {
      const result = converter.convertRequest({
        model: 'gpt-4o',
        messages: [
          { role: 'system', content: 'You are helpful.' },
          { role: 'user', content: 'Hello' },
        ],
      });

      expect(result.model).toBe('gpt-4o');
      expect(result.system).toEqual([{ type: 'text', text: 'You are helpful.' }]);
      expect(result.messages).toEqual([
        { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
      ]);
    });

    it('converts developer role as system', () => {
      const result = converter.convertRequest({
        model: 'gpt-4o',
        messages: [
          { role: 'developer', content: 'Be concise.' },
          { role: 'user', content: 'Hi' },
        ],
      });

      expect(result.system).toEqual([{ type: 'text', text: 'Be concise.' }]);
    });

    it('concatenates multiple system messages', () => {
      const result = converter.convertRequest({
        model: 'gpt-4o',
        messages: [
          { role: 'system', content: 'You are helpful.' },
          { role: 'system', content: 'Be concise.' },
          { role: 'user', content: 'Hi' },
        ],
      });

      expect(result.system).toEqual([
        { type: 'text', text: 'You are helpful.' },
        { type: 'text', text: 'Be concise.' },
      ]);
    });
  });

  describe('tool calling', () => {
    it('converts assistant tool_calls to tool_use blocks', () => {
      const result = converter.convertRequest({
        model: 'gpt-4o',
        messages: [
          { role: 'user', content: 'What is the weather?' },
          {
            role: 'assistant',
            tool_calls: [
              { id: 'call_123', type: 'function', function: { name: 'get_weather', arguments: '{"location":"SF"}' } },
            ],
          },
          { role: 'tool', tool_call_id: 'call_123', content: '72°F' },
        ],
      });

      expect(result.messages[1]).toEqual({
        role: 'assistant',
        content: [{ type: 'tool_use', id: 'call_123', name: 'get_weather', input: { location: 'SF' } }],
      });
      expect(result.messages[2]).toEqual({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'call_123', content: [{ type: 'text', text: '72°F' }] }],
      });
    });

    it('merges consecutive tool messages into one user message', () => {
      const result = converter.convertRequest({
        model: 'gpt-4o',
        messages: [
          { role: 'user', content: 'Do two things' },
          {
            role: 'assistant',
            tool_calls: [
              { id: 'c1', type: 'function', function: { name: 'fn1', arguments: '{}' } },
              { id: 'c2', type: 'function', function: { name: 'fn2', arguments: '{}' } },
            ],
          },
          { role: 'tool', tool_call_id: 'c1', content: 'result1' },
          { role: 'tool', tool_call_id: 'c2', content: 'result2' },
        ],
      });

      expect(result.messages[2]).toEqual({
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'c1', content: [{ type: 'text', text: 'result1' }] },
          { type: 'tool_result', tool_use_id: 'c2', content: [{ type: 'text', text: 'result2' }] },
        ],
      });
    });
  });

  describe('multimodal content', () => {
    it('converts image_url with regular URL', () => {
      const result = converter.convertRequest({
        model: 'gpt-4o',
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'What is this?' },
            { type: 'image_url', image_url: { url: 'https://example.com/img.png' } },
          ],
        }],
      });

      expect(result.messages[0]).toEqual({
        role: 'user',
        content: [
          { type: 'text', text: 'What is this?' },
          { type: 'image', source: { type: 'url', url: 'https://example.com/img.png' } },
        ],
      });
    });

    it('converts image_url with base64 data URI', () => {
      const result = converter.convertRequest({
        model: 'gpt-4o',
        messages: [{
          role: 'user',
          content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' } }],
        }],
      });

      const content = result.messages[0].content as Anthropic.ImageBlockParam[];
      expect(content[0]).toEqual({
        type: 'image',
        source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgo=' },
      });
    });
  });

  describe('params mapping', () => {
    it('maps temperature, top_p, max_tokens, stop', () => {
      const result = converter.convertRequest({
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'Hi' }],
        temperature: 0.7, top_p: 0.9, max_tokens: 1000, stop: ['END', 'STOP'],
      });

      expect(result.temperature).toBe(0.7);
      expect(result.top_p).toBe(0.9);
      expect(result.max_tokens).toBe(1000);
      expect(result.stop_sequences).toEqual(['END', 'STOP']);
    });

    it('defaults max_tokens to 4096 when not provided', () => {
      const result = converter.convertRequest({ model: 'gpt-4o', messages: [{ role: 'user', content: 'Hi' }] });
      expect(result.max_tokens).toBe(4096);
    });
  });

  describe('tools mapping', () => {
    it('converts function tools', () => {
      const result = converter.convertRequest({
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [{
          type: 'function',
          function: {
            name: 'get_weather', description: 'Get weather info',
            parameters: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] },
          },
        }],
      });

      expect(result.tools).toEqual([{
        name: 'get_weather', description: 'Get weather info',
        input_schema: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] },
      }]);
    });

    it('maps tool_choice "auto" to {type: "auto"}', () => {
      const result = converter.convertRequest({
        model: 'gpt-4o', messages: [{ role: 'user', content: 'Hi' }],
        tools: [{ type: 'function', function: { name: 'fn', parameters: { type: 'object' } } }],
        tool_choice: 'auto',
      });
      expect(result.tool_choice).toEqual({ type: 'auto' });
    });

    it('maps tool_choice "required" to {type: "any"}', () => {
      const result = converter.convertRequest({
        model: 'gpt-4o', messages: [{ role: 'user', content: 'Hi' }],
        tools: [{ type: 'function', function: { name: 'fn', parameters: { type: 'object' } } }],
        tool_choice: 'required',
      });
      expect(result.tool_choice).toEqual({ type: 'any' });
    });

    it('maps named tool_choice to {type: "tool", name}', () => {
      const result = converter.convertRequest({
        model: 'gpt-4o', messages: [{ role: 'user', content: 'Hi' }],
        tools: [{ type: 'function', function: { name: 'get_weather', parameters: { type: 'object' } } }],
        tool_choice: { type: 'function', function: { name: 'get_weather' } },
      });
      expect(result.tool_choice).toEqual({ type: 'tool', name: 'get_weather' });
    });
  });
});

// ─── convertResponse ───────────────────────────────────────────────

describe('ChatCompletionToMessagesConverter.convertResponse', () => {
  it('converts a basic text response', () => {
    const result = converter.convertResponse({
      id: 'chatcmpl-123', object: 'chat.completion', created: 1700000000, model: 'gpt-4o',
      choices: [{ index: 0, message: { role: 'assistant', content: 'Hello!', refusal: null }, finish_reason: 'stop', logprobs: null }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    });

    expect(result.id).toBe('chatcmpl-123');
    expect(result.content).toEqual([{ type: 'text', text: 'Hello!', citations: null }]);
    expect(result.stop_reason).toBe('end_turn');
    expect(result.usage.input_tokens).toBe(10);
    expect(result.usage.output_tokens).toBe(5);
  });

  it('converts a tool call response', () => {
    const result = converter.convertResponse({
      id: 'id', object: 'chat.completion', created: 0, model: 'gpt-4o',
      choices: [{
        index: 0,
        message: {
          role: 'assistant', content: null, refusal: null,
          tool_calls: [{ id: 'call_abc', type: 'function', function: { name: 'get_weather', arguments: '{"location":"SF"}' } }],
        },
        finish_reason: 'tool_calls', logprobs: null,
      }],
    });

    expect(result.stop_reason).toBe('tool_use');
    expect(result.content).toEqual([
      { type: 'tool_use', id: 'call_abc', name: 'get_weather', input: { location: 'SF' }, caller: { type: 'direct' } },
    ]);
  });

  it('maps finish_reason "length" to "max_tokens"', () => {
    const result = converter.convertResponse({
      id: 'id', object: 'chat.completion', created: 0, model: 'gpt-4o',
      choices: [{ index: 0, message: { role: 'assistant', content: 'x', refusal: null }, finish_reason: 'length', logprobs: null }],
    });
    expect(result.stop_reason).toBe('max_tokens');
  });
});

// ─── createStreamConverter ─────────────────────────────────────────

function makeChunk(overrides: {
  delta?: Partial<OpenAI.ChatCompletionChunk.Choice.Delta>;
  finish_reason?: OpenAI.ChatCompletionChunk.Choice['finish_reason'];
} = {}): OpenAI.ChatCompletionChunk {
  const { delta = {}, finish_reason = null } = overrides;
  return {
    id: 'chatcmpl-123', object: 'chat.completion.chunk', created: 1700000000, model: 'gpt-4o',
    choices: [{ index: 0, delta, finish_reason }],
  };
}

describe('ChatCompletionToMessagesConverter.createStreamConverter', () => {
  it('emits message_start + content_block_start on first chunk', () => {
    const convert = converter.createStreamConverter();
    const events = convert(makeChunk({ delta: { role: 'assistant' } }));

    expect(events.length).toBe(2);
    expect(events[0].type).toBe('message_start');
    expect(events[1].type).toBe('content_block_start');
  });

  it('emits text_delta for text chunks', () => {
    const convert = converter.createStreamConverter();
    convert(makeChunk({ delta: { role: 'assistant' } }));

    const events = convert(makeChunk({ delta: { content: 'Hello' } }));
    expect(events[0].type).toBe('content_block_delta');
    expect((events[0] as Anthropic.RawContentBlockDeltaEvent).delta.type).toBe('text_delta');
  });

  it('emits stop events on finish', () => {
    const convert = converter.createStreamConverter();
    convert(makeChunk({ delta: { role: 'assistant' } }));

    const events = convert(makeChunk({ finish_reason: 'stop' }));
    const types = events.map(e => e.type);
    expect(types).toContain('content_block_stop');
    expect(types).toContain('message_delta');
    expect(types).toContain('message_stop');
  });

  it('emits content_block_start for tool_use', () => {
    const convert = converter.createStreamConverter();
    convert(makeChunk({ delta: { role: 'assistant' } }));

    const events = convert(makeChunk({
      delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '' } }] },
    }));

    const starts = events.filter(e => e.type === 'content_block_start');
    expect(starts.length).toBe(1);
    expect((starts[0] as Anthropic.RawContentBlockStartEvent).content_block.type).toBe('tool_use');
  });

  it('emits input_json_delta for tool call arguments', () => {
    const convert = converter.createStreamConverter();
    convert(makeChunk({ delta: { role: 'assistant' } }));
    convert(makeChunk({
      delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'fn', arguments: '' } }] },
    }));

    const events = convert(makeChunk({
      delta: { tool_calls: [{ index: 0, function: { arguments: '{"loc' } }] },
    }));

    expect(events[0].type).toBe('content_block_delta');
    expect((events[0] as Anthropic.RawContentBlockDeltaEvent).delta.type).toBe('input_json_delta');
  });
});
