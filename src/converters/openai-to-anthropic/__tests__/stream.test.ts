import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';
import { createOpenAIToAnthropicStreamConverter } from '../stream';

function makeChunk(overrides: Partial<OpenAI.ChatCompletionChunk> & {
  delta?: Partial<OpenAI.ChatCompletionChunk.Choice.Delta>;
  finish_reason?: OpenAI.ChatCompletionChunk.Choice['finish_reason'];
} = {}): OpenAI.ChatCompletionChunk {
  const { delta = {}, finish_reason = null, ...rest } = overrides;
  return {
    id: 'chatcmpl-123',
    object: 'chat.completion.chunk',
    created: 1700000000,
    model: 'gpt-4o',
    choices: [
      {
        index: 0,
        delta,
        finish_reason,
      },
    ],
    ...rest,
  };
}

describe('createOpenAIToAnthropicStreamConverter', () => {
  describe('text streaming', () => {
    it('emits message_start + content_block_start on first chunk', () => {
      const convert = createOpenAIToAnthropicStreamConverter();

      const events = convert(makeChunk({ delta: { role: 'assistant' } }));

      expect(events.length).toBe(2);
      expect(events[0].type).toBe('message_start');
      expect(events[1].type).toBe('content_block_start');
      expect((events[1] as Anthropic.RawContentBlockStartEvent).index).toBe(0);
    });

    it('emits content_block_delta for text chunks', () => {
      const convert = createOpenAIToAnthropicStreamConverter();
      convert(makeChunk({ delta: { role: 'assistant' } }));

      const events = convert(makeChunk({ delta: { content: 'Hello' } }));

      expect(events.length).toBe(1);
      expect(events[0].type).toBe('content_block_delta');
      const delta = (events[0] as Anthropic.RawContentBlockDeltaEvent).delta;
      expect(delta.type).toBe('text_delta');
      expect((delta as Anthropic.TextDelta).text).toBe('Hello');
    });

    it('emits stop events on finish', () => {
      const convert = createOpenAIToAnthropicStreamConverter();
      convert(makeChunk({ delta: { role: 'assistant' } }));
      convert(makeChunk({ delta: { content: 'Hi' } }));

      const events = convert(makeChunk({ finish_reason: 'stop' }));

      const types = events.map(e => e.type);
      expect(types).toContain('content_block_stop');
      expect(types).toContain('message_delta');
      expect(types).toContain('message_stop');
    });
  });

  describe('tool call streaming', () => {
    it('emits content_block_start for tool_use when tool_call starts', () => {
      const convert = createOpenAIToAnthropicStreamConverter();
      convert(makeChunk({ delta: { role: 'assistant' } }));

      const events = convert(makeChunk({
        delta: {
          tool_calls: [{
            index: 0,
            id: 'call_1',
            type: 'function',
            function: { name: 'get_weather', arguments: '' },
          }],
        },
      }));

      // Should stop previous text block and start tool_use block
      const startEvents = events.filter(e => e.type === 'content_block_start');
      expect(startEvents.length).toBe(1);
      const startEvent = startEvents[0] as Anthropic.RawContentBlockStartEvent;
      expect(startEvent.content_block.type).toBe('tool_use');
      expect((startEvent.content_block as any).name).toBe('get_weather');
    });

    it('emits input_json_delta for tool call arguments', () => {
      const convert = createOpenAIToAnthropicStreamConverter();
      convert(makeChunk({ delta: { role: 'assistant' } }));
      convert(makeChunk({
        delta: {
          tool_calls: [{
            index: 0,
            id: 'call_1',
            type: 'function',
            function: { name: 'get_weather', arguments: '' },
          }],
        },
      }));

      const events = convert(makeChunk({
        delta: {
          tool_calls: [{
            index: 0,
            function: { arguments: '{"loc' },
          }],
        },
      }));

      expect(events.length).toBe(1);
      expect(events[0].type).toBe('content_block_delta');
      const delta = (events[0] as Anthropic.RawContentBlockDeltaEvent).delta;
      expect(delta.type).toBe('input_json_delta');
      expect((delta as Anthropic.InputJSONDelta).partial_json).toBe('{"loc');
    });
  });

  describe('text + tool call mixed', () => {
    it('handles text followed by tool calls', () => {
      const convert = createOpenAIToAnthropicStreamConverter();
      convert(makeChunk({ delta: { role: 'assistant' } }));
      convert(makeChunk({ delta: { content: 'Let me check.' } }));

      const events = convert(makeChunk({
        delta: {
          tool_calls: [{
            index: 0,
            id: 'call_1',
            type: 'function',
            function: { name: 'fn', arguments: '' },
          }],
        },
      }));

      // Should have content_block_stop for text, then content_block_start for tool
      const types = events.map(e => e.type);
      expect(types).toContain('content_block_stop');
      expect(types).toContain('content_block_start');
    });
  });
});
