import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';
import { createAnthropicToOpenAIStreamConverter } from '../stream';

const baseUsage: Anthropic.Usage = {
  input_tokens: 10,
  output_tokens: 0,
  cache_creation_input_tokens: null,
  cache_read_input_tokens: null,
  cache_creation: null,
  inference_geo: null,
  server_tool_use: null,
  service_tier: null,
};

const baseDeltaUsage: Anthropic.MessageDeltaUsage = {
  output_tokens: 5,
  input_tokens: null,
  cache_creation_input_tokens: null,
  cache_read_input_tokens: null,
  server_tool_use: null,
};

function messageStart(): Anthropic.RawMessageStartEvent {
  return {
    type: 'message_start',
    message: {
      id: 'msg_123',
      type: 'message',
      role: 'assistant',
      model: 'claude-sonnet-4-20250514',
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: baseUsage,
      container: null,
    },
  };
}

describe('createAnthropicToOpenAIStreamConverter', () => {
  describe('text streaming', () => {
    it('emits first chunk with role on message_start', () => {
      const convert = createAnthropicToOpenAIStreamConverter();

      const result = convert(messageStart());

      expect(result).not.toBeNull();
      expect(result!.id).toBe('msg_123');
      expect(result!.model).toBe('claude-sonnet-4-20250514');
      expect(result!.choices[0].delta.role).toBe('assistant');
    });

    it('returns null for content_block_start (text)', () => {
      const convert = createAnthropicToOpenAIStreamConverter();
      convert(messageStart());

      const result = convert({
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '', citations: null },
      });

      expect(result).toBeNull();
    });

    it('emits text content for content_block_delta (text_delta)', () => {
      const convert = createAnthropicToOpenAIStreamConverter();
      convert(messageStart());

      const result = convert({
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: 'Hello' },
      });

      expect(result).not.toBeNull();
      expect(result!.choices[0].delta.content).toBe('Hello');
    });

    it('emits finish_reason on message_delta', () => {
      const convert = createAnthropicToOpenAIStreamConverter();
      convert(messageStart());

      const result = convert({
        type: 'message_delta',
        delta: { stop_reason: 'end_turn', stop_sequence: null, container: null },
        usage: baseDeltaUsage,
      });

      expect(result).not.toBeNull();
      expect(result!.choices[0].finish_reason).toBe('stop');
    });

    it('returns null for message_stop', () => {
      const convert = createAnthropicToOpenAIStreamConverter();
      convert(messageStart());

      const result = convert({ type: 'message_stop' });
      expect(result).toBeNull();
    });
  });

  describe('tool call streaming', () => {
    it('emits tool_call start on content_block_start (tool_use)', () => {
      const convert = createAnthropicToOpenAIStreamConverter();
      convert(messageStart());

      const result = convert({
        type: 'content_block_start',
        index: 0,
        content_block: {
          type: 'tool_use',
          id: 'toolu_1',
          name: 'get_weather',
          input: {},
          caller: { type: 'direct' },
        },
      });

      expect(result).not.toBeNull();
      const tc = result!.choices[0].delta.tool_calls![0];
      expect(tc.index).toBe(0);
      expect(tc.id).toBe('toolu_1');
      expect(tc.type).toBe('function');
      expect(tc.function!.name).toBe('get_weather');
      expect(tc.function!.arguments).toBe('');
    });

    it('emits tool_call arguments on content_block_delta (input_json_delta)', () => {
      const convert = createAnthropicToOpenAIStreamConverter();
      convert(messageStart());
      convert({
        type: 'content_block_start',
        index: 0,
        content_block: {
          type: 'tool_use',
          id: 'toolu_1',
          name: 'get_weather',
          input: {},
          caller: { type: 'direct' },
        },
      });

      const result = convert({
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'input_json_delta', partial_json: '{"loc' },
      });

      expect(result).not.toBeNull();
      const tc = result!.choices[0].delta.tool_calls![0];
      expect(tc.index).toBe(0);
      expect(tc.function!.arguments).toBe('{"loc');
    });

    it('maps tool_use stop_reason to tool_calls finish_reason', () => {
      const convert = createAnthropicToOpenAIStreamConverter();
      convert(messageStart());

      const result = convert({
        type: 'message_delta',
        delta: { stop_reason: 'tool_use', stop_sequence: null, container: null },
        usage: baseDeltaUsage,
      });

      expect(result!.choices[0].finish_reason).toBe('tool_calls');
    });
  });

  describe('returns null for irrelevant events', () => {
    it('returns null for content_block_stop', () => {
      const convert = createAnthropicToOpenAIStreamConverter();
      convert(messageStart());

      const result = convert({ type: 'content_block_stop', index: 0 });
      expect(result).toBeNull();
    });
  });
});
