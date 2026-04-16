import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';
import { ChatCompletionToMessagesConverter } from '../chat-completion-to-messages';

describe('ChatCompletionToMessagesConverter', () => {
  const converter = new ChatCompletionToMessagesConverter();

  // ===== convertRequest =====

  describe('convertRequest', () => {
    describe('basic text conversation', () => {
      it('converts system + user messages', () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: 'gpt-4o',
          messages: [
            { role: 'system', content: 'You are helpful.' },
            { role: 'user', content: 'Hello' },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.model).toBe('gpt-4o');
        expect(result.system).toEqual([{ type: 'text', text: 'You are helpful.' }]);
        expect(result.messages).toEqual([
          { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
        ]);
      });

      it('converts developer role as system', () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: 'gpt-4o',
          messages: [
            { role: 'developer', content: 'Be concise.' },
            { role: 'user', content: 'Hi' },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.system).toEqual([{ type: 'text', text: 'Be concise.' }]);
        expect(result.messages).toEqual([
          { role: 'user', content: [{ type: 'text', text: 'Hi' }] },
        ]);
      });

      it('concatenates multiple system messages', () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: 'gpt-4o',
          messages: [
            { role: 'system', content: 'You are helpful.' },
            { role: 'system', content: 'Be concise.' },
            { role: 'user', content: 'Hi' },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.system).toEqual([
          { type: 'text', text: 'You are helpful.' },
          { type: 'text', text: 'Be concise.' },
        ]);
      });
    });

    describe('multi-turn conversation', () => {
      it('converts user and assistant turns', () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: 'gpt-4o',
          messages: [
            { role: 'user', content: 'Hello' },
            { role: 'assistant', content: 'Hi there!' },
            { role: 'user', content: 'How are you?' },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.messages).toEqual([
          { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
          { role: 'assistant', content: [{ type: 'text', text: 'Hi there!' }] },
          { role: 'user', content: [{ type: 'text', text: 'How are you?' }] },
        ]);
      });
    });

    describe('tool calling', () => {
      it('converts assistant tool_calls to tool_use blocks', () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: 'gpt-4o',
          messages: [
            { role: 'user', content: 'What is the weather?' },
            {
              role: 'assistant',
              tool_calls: [
                {
                  id: 'call_123',
                  type: 'function',
                  function: { name: 'get_weather', arguments: '{"location":"SF"}' },
                },
              ],
            },
            { role: 'tool', tool_call_id: 'call_123', content: '72°F' },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.messages).toEqual([
          { role: 'user', content: [{ type: 'text', text: 'What is the weather?' }] },
          {
            role: 'assistant',
            content: [
              { type: 'tool_use', id: 'call_123', name: 'get_weather', input: { location: 'SF' } },
            ],
          },
          {
            role: 'user',
            content: [
              { type: 'tool_result', tool_use_id: 'call_123', content: [{ type: 'text', text: '72°F' }] },
            ],
          },
        ]);
      });

      it('converts assistant with content + tool_calls', () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: 'gpt-4o',
          messages: [
            { role: 'user', content: 'Check weather' },
            {
              role: 'assistant',
              content: 'Let me check.',
              tool_calls: [
                { id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{}' } },
              ],
            },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.messages[1]).toEqual({
          role: 'assistant',
          content: [
            { type: 'text', text: 'Let me check.' },
            { type: 'tool_use', id: 'call_1', name: 'get_weather', input: {} },
          ],
        });
      });

      it('merges consecutive tool messages into one user message', () => {
        const input: OpenAI.ChatCompletionCreateParams = {
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
        };

        const result = converter.convertRequest(input);

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
        const input: OpenAI.ChatCompletionCreateParams = {
          model: 'gpt-4o',
          messages: [
            {
              role: 'user',
              content: [
                { type: 'text', text: 'What is this?' },
                { type: 'image_url', image_url: { url: 'https://example.com/img.png' } },
              ],
            },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.messages[0]).toEqual({
          role: 'user',
          content: [
            { type: 'text', text: 'What is this?' },
            { type: 'image', source: { type: 'url', url: 'https://example.com/img.png' } },
          ],
        });
      });

      it('converts image_url with base64 data URI', () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: 'gpt-4o',
          messages: [
            {
              role: 'user',
              content: [
                { type: 'image_url', image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' } },
              ],
            },
          ],
        };

        const result = converter.convertRequest(input);
        const content = result.messages[0].content as Anthropic.ImageBlockParam[];

        expect(content[0]).toEqual({
          type: 'image',
          source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgo=' },
        });
      });
    });

    describe('params mapping', () => {
      it('maps temperature, top_p, max_tokens, stop', () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: 'gpt-4o',
          messages: [{ role: 'user', content: 'Hi' }],
          temperature: 0.7, top_p: 0.9, max_tokens: 1000, stop: ['END', 'STOP'],
        };

        const result = converter.convertRequest(input);

        expect(result.temperature).toBe(0.7);
        expect(result.top_p).toBe(0.9);
        expect(result.max_tokens).toBe(1000);
        expect(result.stop_sequences).toEqual(['END', 'STOP']);
      });

      it('maps max_completion_tokens to max_tokens', () => {
        const result = converter.convertRequest({
          model: 'gpt-4o', messages: [{ role: 'user', content: 'Hi' }], max_completion_tokens: 500,
        });
        expect(result.max_tokens).toBe(500);
      });

      it('maps stop as string to stop_sequences array', () => {
        const result = converter.convertRequest({
          model: 'gpt-4o', messages: [{ role: 'user', content: 'Hi' }], stop: 'END',
        });
        expect(result.stop_sequences).toEqual(['END']);
      });

      it('defaults max_tokens to 4096 when not provided', () => {
        const result = converter.convertRequest({
          model: 'gpt-4o', messages: [{ role: 'user', content: 'Hi' }],
        });
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

      it('maps tool_choice "auto"', () => {
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

      it('maps tool_choice "none"', () => {
        const result = converter.convertRequest({
          model: 'gpt-4o', messages: [{ role: 'user', content: 'Hi' }], tool_choice: 'none',
        });
        expect(result.tool_choice).toEqual({ type: 'none' });
      });

      it('maps named tool_choice', () => {
        const result = converter.convertRequest({
          model: 'gpt-4o', messages: [{ role: 'user', content: 'Hi' }],
          tools: [{ type: 'function', function: { name: 'get_weather', parameters: { type: 'object' } } }],
          tool_choice: { type: 'function', function: { name: 'get_weather' } },
        });
        expect(result.tool_choice).toEqual({ type: 'tool', name: 'get_weather' });
      });
    });
  });

  // ===== convertResponse =====

  describe('convertResponse', () => {
    it('converts a basic text response', () => {
      const input: OpenAI.ChatCompletion = {
        id: 'chatcmpl-123', object: 'chat.completion', created: 1700000000, model: 'gpt-4o',
        choices: [{ index: 0, message: { role: 'assistant', content: 'Hello!', refusal: null }, finish_reason: 'stop', logprobs: null }],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      };

      const result = converter.convertResponse(input);

      expect(result.id).toBe('chatcmpl-123');
      expect(result.model).toBe('gpt-4o');
      expect(result.content).toEqual([{ type: 'text', text: 'Hello!', citations: null }]);
      expect(result.stop_reason).toBe('end_turn');
      expect(result.usage.input_tokens).toBe(10);
      expect(result.usage.output_tokens).toBe(5);
    });

    it('converts a tool call response', () => {
      const input: OpenAI.ChatCompletion = {
        id: 'chatcmpl-456', object: 'chat.completion', created: 1700000000, model: 'gpt-4o',
        choices: [{
          index: 0,
          message: {
            role: 'assistant', content: null, refusal: null,
            tool_calls: [{ id: 'call_abc', type: 'function', function: { name: 'get_weather', arguments: '{"location":"SF"}' } }],
          },
          finish_reason: 'tool_calls', logprobs: null,
        }],
      };

      const result = converter.convertResponse(input);

      expect(result.stop_reason).toBe('tool_use');
      expect(result.content).toEqual([
        { type: 'tool_use', id: 'call_abc', name: 'get_weather', input: { location: 'SF' }, caller: { type: 'direct' } },
      ]);
    });

    it('converts mixed content + tool calls', () => {
      const input: OpenAI.ChatCompletion = {
        id: 'chatcmpl-789', object: 'chat.completion', created: 1700000000, model: 'gpt-4o',
        choices: [{
          index: 0,
          message: {
            role: 'assistant', content: 'Let me check.', refusal: null,
            tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{}' } }],
          },
          finish_reason: 'tool_calls', logprobs: null,
        }],
      };

      const result = converter.convertResponse(input);

      expect(result.content).toEqual([
        { type: 'text', text: 'Let me check.', citations: null },
        { type: 'tool_use', id: 'call_1', name: 'get_weather', input: {}, caller: { type: 'direct' } },
      ]);
    });

    it('maps finish_reason "length" to "max_tokens"', () => {
      const input: OpenAI.ChatCompletion = {
        id: 'id', object: 'chat.completion', created: 0, model: 'gpt-4o',
        choices: [{ index: 0, message: { role: 'assistant', content: 'truncated', refusal: null }, finish_reason: 'length', logprobs: null }],
      };
      expect(converter.convertResponse(input).stop_reason).toBe('max_tokens');
    });

    it('maps finish_reason "content_filter" to "end_turn"', () => {
      const input: OpenAI.ChatCompletion = {
        id: 'id', object: 'chat.completion', created: 0, model: 'gpt-4o',
        choices: [{ index: 0, message: { role: 'assistant', content: '', refusal: null }, finish_reason: 'content_filter', logprobs: null }],
      };
      expect(converter.convertResponse(input).stop_reason).toBe('end_turn');
    });

    it('handles null content with no tool calls', () => {
      const input: OpenAI.ChatCompletion = {
        id: 'id', object: 'chat.completion', created: 0, model: 'gpt-4o',
        choices: [{ index: 0, message: { role: 'assistant', content: null, refusal: null }, finish_reason: 'stop', logprobs: null }],
      };
      expect(converter.convertResponse(input).content).toEqual([{ type: 'text', text: '', citations: null }]);
    });
  });

  // ===== convertStream =====

  describe('convertStream', () => {
    function makeChunk(overrides: Partial<OpenAI.ChatCompletionChunk> & {
      delta?: Partial<OpenAI.ChatCompletionChunk.Choice.Delta>;
      finish_reason?: OpenAI.ChatCompletionChunk.Choice['finish_reason'];
    } = {}): OpenAI.ChatCompletionChunk {
      const { delta = {}, finish_reason = null, ...rest } = overrides;
      return {
        id: 'chatcmpl-123', object: 'chat.completion.chunk', created: 1700000000, model: 'gpt-4o',
        choices: [{ index: 0, delta, finish_reason }],
        ...rest,
      };
    }

    describe('text streaming', () => {
      it('emits message_start + content_block_start on first chunk', () => {
        const c = new ChatCompletionToMessagesConverter();
        const events = c.convertStream(makeChunk({ delta: { role: 'assistant' } }));

        expect(events.length).toBe(2);
        expect(events[0].type).toBe('message_start');
        expect(events[1].type).toBe('content_block_start');
        expect((events[1] as Anthropic.RawContentBlockStartEvent).index).toBe(0);
      });

      it('emits content_block_delta for text chunks', () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStream(makeChunk({ delta: { role: 'assistant' } }));

        const events = c.convertStream(makeChunk({ delta: { content: 'Hello' } }));

        expect(events.length).toBe(1);
        expect(events[0].type).toBe('content_block_delta');
        const delta = (events[0] as Anthropic.RawContentBlockDeltaEvent).delta;
        expect(delta.type).toBe('text_delta');
        expect((delta as Anthropic.TextDelta).text).toBe('Hello');
      });

      it('emits stop events on finish', () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStream(makeChunk({ delta: { role: 'assistant' } }));
        c.convertStream(makeChunk({ delta: { content: 'Hi' } }));

        const events = c.convertStream(makeChunk({ finish_reason: 'stop' }));
        const types = events.map(e => e.type);

        expect(types).toContain('content_block_stop');
        expect(types).toContain('message_delta');
        expect(types).toContain('message_stop');
      });
    });

    describe('tool call streaming', () => {
      it('emits content_block_start for tool_use when tool_call starts', () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStream(makeChunk({ delta: { role: 'assistant' } }));

        const events = c.convertStream(makeChunk({
          delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '' } }] },
        }));

        const startEvents = events.filter(e => e.type === 'content_block_start');
        expect(startEvents.length).toBe(1);
        const startEvent = startEvents[0] as Anthropic.RawContentBlockStartEvent;
        expect(startEvent.content_block.type).toBe('tool_use');
        expect((startEvent.content_block as any).name).toBe('get_weather');
      });

      it('emits input_json_delta for tool call arguments', () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStream(makeChunk({ delta: { role: 'assistant' } }));
        c.convertStream(makeChunk({
          delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '' } }] },
        }));

        const events = c.convertStream(makeChunk({
          delta: { tool_calls: [{ index: 0, function: { arguments: '{"loc' } }] },
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
        const c = new ChatCompletionToMessagesConverter();
        c.convertStream(makeChunk({ delta: { role: 'assistant' } }));
        c.convertStream(makeChunk({ delta: { content: 'Let me check.' } }));

        const events = c.convertStream(makeChunk({
          delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'fn', arguments: '' } }] },
        }));

        const types = events.map(e => e.type);
        expect(types).toContain('content_block_stop');
        expect(types).toContain('content_block_start');
      });
    });
  });
});
