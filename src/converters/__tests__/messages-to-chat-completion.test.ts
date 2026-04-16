import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';
import { MessagesToChatCompletionConverter } from '../messages-to-chat-completion';

function makeMessage(overrides: Partial<Anthropic.Message> = {}): Anthropic.Message {
  return {
    id: 'msg_123', type: 'message', role: 'assistant', model: 'claude-sonnet-4-20250514',
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
      id: 'msg_123', type: 'message', role: 'assistant', model: 'claude-sonnet-4-20250514',
      content: [], stop_reason: null, stop_sequence: null, usage: baseUsage, container: null,
    },
  };
}

describe('MessagesToChatCompletionConverter', () => {
  const converter = new MessagesToChatCompletionConverter();

  // ===== convertRequest =====

  describe('convertRequest', () => {
    describe('basic text conversation', () => {
      it('converts system string to system message', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          system: 'You are helpful.',
          messages: [{ role: 'user', content: 'Hello' }],
        });

        expect(result.model).toBe('claude-sonnet-4-20250514');
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
    });

    describe('multi-turn conversation', () => {
      it('converts user and assistant turns', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [
            { role: 'user', content: 'Hello' },
            { role: 'assistant', content: 'Hi there!' },
            { role: 'user', content: 'How are you?' },
          ],
        });

        expect(result.messages).toEqual([
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there!' },
          { role: 'user', content: 'How are you?' },
        ]);
      });
    });

    describe('tool calling', () => {
      it('converts tool_use blocks to assistant tool_calls', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [
            { role: 'user', content: 'What is the weather?' },
            { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_123', name: 'get_weather', input: { location: 'SF' } }] },
            { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_123', content: '72°F' }] },
          ],
        });

        expect(result.messages[1]).toEqual({
          role: 'assistant',
          tool_calls: [{ id: 'toolu_123', type: 'function', function: { name: 'get_weather', arguments: '{"location":"SF"}' } }],
        });
        expect(result.messages[2]).toEqual({ role: 'tool', tool_call_id: 'toolu_123', content: '72°F' });
      });

      it('converts assistant with text + tool_use blocks', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [
            { role: 'user', content: 'Check weather' },
            { role: 'assistant', content: [{ type: 'text', text: 'Let me check.' }, { type: 'tool_use', id: 'toolu_1', name: 'get_weather', input: {} }] },
          ],
        });

        expect(result.messages[1]).toEqual({
          role: 'assistant', content: 'Let me check.',
          tool_calls: [{ id: 'toolu_1', type: 'function', function: { name: 'get_weather', arguments: '{}' } }],
        });
      });

      it('splits user message with multiple tool_results into separate tool messages', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [
            { role: 'user', content: 'Do two things' },
            { role: 'assistant', content: [{ type: 'tool_use', id: 'c1', name: 'fn1', input: {} }, { type: 'tool_use', id: 'c2', name: 'fn2', input: {} }] },
            { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'c1', content: 'result1' }, { type: 'tool_result', tool_use_id: 'c2', content: 'result2' }] },
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

        expect(content[0]).toEqual({ type: 'text', text: 'What is this?' });
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
          tools: [{ name: 'get_weather', description: 'Get weather info', input_schema: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] } }],
        });

        expect(result.tools).toEqual([{
          type: 'function',
          function: { name: 'get_weather', description: 'Get weather info', parameters: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] } },
        }]);
      });

      it('maps tool_choice {type: "auto"} to "auto"', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [{ role: 'user', content: 'Hi' }], tool_choice: { type: 'auto' },
        });
        expect(result.tool_choice).toBe('auto');
      });

      it('maps tool_choice {type: "any"} to "required"', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [{ role: 'user', content: 'Hi' }], tool_choice: { type: 'any' },
        });
        expect(result.tool_choice).toBe('required');
      });

      it('maps tool_choice {type: "none"} to "none"', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [{ role: 'user', content: 'Hi' }], tool_choice: { type: 'none' },
        });
        expect(result.tool_choice).toBe('none');
      });

      it('maps tool_choice {type: "tool", name} to named choice', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [{ role: 'user', content: 'Hi' }], tool_choice: { type: 'tool', name: 'get_weather' },
        });
        expect(result.tool_choice).toEqual({ type: 'function', function: { name: 'get_weather' } });
      });
    });

    describe('output_config mapping', () => {
      it('maps json_schema output_config to response_format', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [{ role: 'user', content: 'Hi' }],
          output_config: { format: { type: 'json_schema', schema: { type: 'object', properties: { answer: { type: 'string' } } } } },
        });
        expect(result.response_format).toEqual({
          type: 'json_schema',
          json_schema: { name: 'response', schema: { type: 'object', properties: { answer: { type: 'string' } } }, strict: true },
        });
      });

      it('maps output_config without format to text', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [{ role: 'user', content: 'Hi' }],
          output_config: {},
        });
        expect(result.response_format).toEqual({ type: 'text' });
      });
    });

    describe('thinking mapping', () => {
      it('maps thinking enabled to reasoning_effort', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 16000,
          messages: [{ role: 'user', content: 'Hi' }],
          thinking: { type: 'enabled', budget_tokens: 10000 },
        });
        expect((result as any).reasoning_effort).toBe('high');
      });

      it('maps thinking disabled to reasoning_effort none', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [{ role: 'user', content: 'Hi' }],
          thinking: { type: 'disabled' },
        });
        expect((result as any).reasoning_effort).toBe('none');
      });
    });

    describe('metadata mapping', () => {
      it('maps metadata.user_id to user', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [{ role: 'user', content: 'Hi' }],
          metadata: { user_id: 'user-123' },
        });
        expect(result.user).toBe('user-123');
      });
    });

    describe('parallel_tool_calls mapping', () => {
      it('maps disable_parallel_tool_use=true to parallel_tool_calls=false', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [{ role: 'user', content: 'Hi' }],
          tools: [{ name: 'fn', input_schema: { type: 'object' } }],
          tool_choice: { type: 'auto', disable_parallel_tool_use: true },
        });
        expect(result.tool_choice).toBe('auto');
        expect((result as any).parallel_tool_calls).toBe(false);
      });

      it('does not set parallel_tool_calls when disable_parallel_tool_use is not set', () => {
        const result = converter.convertRequest({
          model: 'claude-sonnet-4-20250514', max_tokens: 1024,
          messages: [{ role: 'user', content: 'Hi' }],
          tool_choice: { type: 'auto' },
        });
        expect(result.tool_choice).toBe('auto');
        expect((result as any).parallel_tool_calls).toBeUndefined();
      });
    });
  });

  // ===== convertResponse =====

  describe('convertResponse', () => {
    it('converts a basic text response', () => {
      const result = converter.convertResponse(makeMessage());

      expect(result.id).toBe('msg_123');
      expect(result.model).toBe('claude-sonnet-4-20250514');
      expect(result.object).toBe('chat.completion');
      expect(result.choices[0].message.content).toBe('Hello!');
      expect(result.choices[0].finish_reason).toBe('stop');
      expect(result.usage?.prompt_tokens).toBe(10);
      expect(result.usage?.completion_tokens).toBe(5);
      expect(result.usage?.total_tokens).toBe(15);
    });

    it('converts a tool_use response', () => {
      const result = converter.convertResponse(makeMessage({
        content: [{ type: 'tool_use', id: 'toolu_123', name: 'get_weather', input: { location: 'SF' }, caller: { type: 'direct' } }],
        stop_reason: 'tool_use',
      }));

      expect(result.choices[0].finish_reason).toBe('tool_calls');
      expect(result.choices[0].message.content).toBeNull();
      expect(result.choices[0].message.tool_calls).toEqual([
        { id: 'toolu_123', type: 'function', function: { name: 'get_weather', arguments: '{"location":"SF"}' } },
      ]);
    });

    it('converts mixed text + tool_use response', () => {
      const result = converter.convertResponse(makeMessage({
        content: [
          { type: 'text', text: 'Let me check.', citations: null },
          { type: 'tool_use', id: 'toolu_1', name: 'get_weather', input: {}, caller: { type: 'direct' } },
        ],
        stop_reason: 'tool_use',
      }));

      expect(result.choices[0].message.content).toBe('Let me check.');
      expect(result.choices[0].message.tool_calls).toEqual([
        { id: 'toolu_1', type: 'function', function: { name: 'get_weather', arguments: '{}' } },
      ]);
    });

    it('maps stop_reason "end_turn" to "stop"', () => {
      expect(converter.convertResponse(makeMessage({ stop_reason: 'end_turn' })).choices[0].finish_reason).toBe('stop');
    });

    it('maps stop_reason "max_tokens" to "length"', () => {
      expect(converter.convertResponse(makeMessage({ stop_reason: 'max_tokens' })).choices[0].finish_reason).toBe('length');
    });

    it('maps stop_reason "tool_use" to "tool_calls"', () => {
      expect(converter.convertResponse(makeMessage({ stop_reason: 'tool_use' })).choices[0].finish_reason).toBe('tool_calls');
    });

    it('joins multiple text blocks', () => {
      const result = converter.convertResponse(makeMessage({
        content: [{ type: 'text', text: 'Hello', citations: null }, { type: 'text', text: 'World', citations: null }],
      }));
      expect(result.choices[0].message.content).toBe('Hello\nWorld');
    });
  });

  // ===== convertStream =====

  describe('convertStream', () => {
    describe('text streaming', () => {
      it('emits first chunk with role on message_start', () => {
        const c = new MessagesToChatCompletionConverter();
        const result = c.convertStream(messageStart());

        expect(result).not.toBeNull();
        expect(result!.id).toBe('msg_123');
        expect(result!.model).toBe('claude-sonnet-4-20250514');
        expect(result!.choices[0].delta.role).toBe('assistant');
      });

      it('returns null for content_block_start (text)', () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
          type: 'content_block_start', index: 0,
          content_block: { type: 'text', text: '', citations: null },
        });

        expect(result).toBeNull();
      });

      it('emits text content for content_block_delta (text_delta)', () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
          type: 'content_block_delta', index: 0,
          delta: { type: 'text_delta', text: 'Hello' },
        });

        expect(result).not.toBeNull();
        expect(result!.choices[0].delta.content).toBe('Hello');
      });

      it('emits finish_reason on message_delta', () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
          type: 'message_delta',
          delta: { stop_reason: 'end_turn', stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });

        expect(result).not.toBeNull();
        expect(result!.choices[0].finish_reason).toBe('stop');
      });

      it('returns null for message_stop', () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());
        expect(c.convertStream({ type: 'message_stop' })).toBeNull();
      });
    });

    describe('tool call streaming', () => {
      it('emits tool_call start on content_block_start (tool_use)', () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
          type: 'content_block_start', index: 0,
          content_block: { type: 'tool_use', id: 'toolu_1', name: 'get_weather', input: {}, caller: { type: 'direct' } },
        });

        expect(result).not.toBeNull();
        const tc = result!.choices[0].delta.tool_calls![0];
        expect(tc.index).toBe(0);
        expect(tc.id).toBe('toolu_1');
        expect(tc.type).toBe('function');
        expect(tc.function!.name).toBe('get_weather');
      });

      it('emits tool_call arguments on content_block_delta (input_json_delta)', () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());
        c.convertStream({
          type: 'content_block_start', index: 0,
          content_block: { type: 'tool_use', id: 'toolu_1', name: 'get_weather', input: {}, caller: { type: 'direct' } },
        });

        const result = c.convertStream({
          type: 'content_block_delta', index: 0,
          delta: { type: 'input_json_delta', partial_json: '{"loc' },
        });

        expect(result).not.toBeNull();
        const tc = result!.choices[0].delta.tool_calls![0];
        expect(tc.index).toBe(0);
        expect(tc.function!.arguments).toBe('{"loc');
      });

      it('maps tool_use stop_reason to tool_calls finish_reason', () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
          type: 'message_delta',
          delta: { stop_reason: 'tool_use', stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });

        expect(result!.choices[0].finish_reason).toBe('tool_calls');
      });
    });

    describe('returns null for irrelevant events', () => {
      it('returns null for content_block_stop', () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());
        expect(c.convertStream({ type: 'content_block_stop', index: 0 })).toBeNull();
      });
    });
  });
});
