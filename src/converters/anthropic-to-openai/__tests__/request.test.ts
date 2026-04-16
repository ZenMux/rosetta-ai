import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';
import { convertAnthropicRequestToOpenAI } from '../request';

describe('convertAnthropicRequestToOpenAI', () => {
  describe('basic text conversation', () => {
    it('converts system string to system message', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        system: 'You are helpful.',
        messages: [{ role: 'user', content: 'Hello' }],
      };

      const result = convertAnthropicRequestToOpenAI(input);

      expect(result.model).toBe('claude-sonnet-4-20250514');
      expect(result.messages[0]).toEqual({ role: 'system', content: 'You are helpful.' });
      expect(result.messages[1]).toEqual({ role: 'user', content: 'Hello' });
    });

    it('converts system content blocks to system message', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        system: [
          { type: 'text', text: 'You are helpful.' },
          { type: 'text', text: 'Be concise.' },
        ],
        messages: [{ role: 'user', content: 'Hello' }],
      };

      const result = convertAnthropicRequestToOpenAI(input);

      expect(result.messages[0]).toEqual({
        role: 'system',
        content: 'You are helpful.\nBe concise.',
      });
    });

    it('converts user content blocks to string when text only', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [{ type: 'text', text: 'Hello' }],
          },
        ],
      };

      const result = convertAnthropicRequestToOpenAI(input);

      expect(result.messages[0]).toEqual({ role: 'user', content: 'Hello' });
    });
  });

  describe('multi-turn conversation', () => {
    it('converts user and assistant turns', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        messages: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there!' },
          { role: 'user', content: 'How are you?' },
        ],
      };

      const result = convertAnthropicRequestToOpenAI(input);

      expect(result.messages).toEqual([
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there!' },
        { role: 'user', content: 'How are you?' },
      ]);
    });
  });

  describe('tool calling', () => {
    it('converts tool_use blocks to assistant tool_calls', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        messages: [
          { role: 'user', content: 'What is the weather?' },
          {
            role: 'assistant',
            content: [
              {
                type: 'tool_use',
                id: 'toolu_123',
                name: 'get_weather',
                input: { location: 'SF' },
              },
            ],
          },
          {
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'toolu_123',
                content: '72°F',
              },
            ],
          },
        ],
      };

      const result = convertAnthropicRequestToOpenAI(input);

      expect(result.messages[1]).toEqual({
        role: 'assistant',
        tool_calls: [
          {
            id: 'toolu_123',
            type: 'function',
            function: {
              name: 'get_weather',
              arguments: '{"location":"SF"}',
            },
          },
        ],
      });

      expect(result.messages[2]).toEqual({
        role: 'tool',
        tool_call_id: 'toolu_123',
        content: '72°F',
      });
    });

    it('converts assistant with text + tool_use blocks', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        messages: [
          { role: 'user', content: 'Check weather' },
          {
            role: 'assistant',
            content: [
              { type: 'text', text: 'Let me check.' },
              {
                type: 'tool_use',
                id: 'toolu_1',
                name: 'get_weather',
                input: {},
              },
            ],
          },
        ],
      };

      const result = convertAnthropicRequestToOpenAI(input);

      expect(result.messages[1]).toEqual({
        role: 'assistant',
        content: 'Let me check.',
        tool_calls: [
          {
            id: 'toolu_1',
            type: 'function',
            function: { name: 'get_weather', arguments: '{}' },
          },
        ],
      });
    });

    it('splits user message with multiple tool_results into separate tool messages', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
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
      };

      const result = convertAnthropicRequestToOpenAI(input);

      expect(result.messages[2]).toEqual({
        role: 'tool',
        tool_call_id: 'c1',
        content: 'result1',
      });
      expect(result.messages[3]).toEqual({
        role: 'tool',
        tool_call_id: 'c2',
        content: 'result2',
      });
    });
  });

  describe('multimodal content', () => {
    it('converts image block with URL source', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'What is this?' },
              {
                type: 'image',
                source: { type: 'url', url: 'https://example.com/img.png' },
              },
            ],
          },
        ],
      };

      const result = convertAnthropicRequestToOpenAI(input);
      const msg = result.messages[0] as OpenAI.ChatCompletionUserMessageParam;
      const content = msg.content as OpenAI.ChatCompletionContentPart[];

      expect(content[0]).toEqual({ type: 'text', text: 'What is this?' });
      expect(content[1]).toEqual({
        type: 'image_url',
        image_url: { url: 'https://example.com/img.png' },
      });
    });

    it('converts image block with base64 source', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'image',
                source: {
                  type: 'base64',
                  media_type: 'image/png',
                  data: 'iVBORw0KGgo=',
                },
              },
            ],
          },
        ],
      };

      const result = convertAnthropicRequestToOpenAI(input);
      const msg = result.messages[0] as OpenAI.ChatCompletionUserMessageParam;
      const content = msg.content as OpenAI.ChatCompletionContentPart[];

      expect(content[0]).toEqual({
        type: 'image_url',
        image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' },
      });
    });
  });

  describe('params mapping', () => {
    it('maps temperature, top_p, max_tokens, stop_sequences', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1000,
        messages: [{ role: 'user', content: 'Hi' }],
        temperature: 0.7,
        top_p: 0.9,
        stop_sequences: ['END', 'STOP'],
      };

      const result = convertAnthropicRequestToOpenAI(input);

      expect(result.temperature).toBe(0.7);
      expect(result.top_p).toBe(0.9);
      expect(result.max_tokens).toBe(1000);
      expect(result.stop).toEqual(['END', 'STOP']);
    });
  });

  describe('tools mapping', () => {
    it('converts Anthropic tools to OpenAI function tools', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [
          {
            name: 'get_weather',
            description: 'Get weather info',
            input_schema: {
              type: 'object',
              properties: { location: { type: 'string' } },
              required: ['location'],
            },
          },
        ],
      };

      const result = convertAnthropicRequestToOpenAI(input);

      expect(result.tools).toEqual([
        {
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get weather info',
            parameters: {
              type: 'object',
              properties: { location: { type: 'string' } },
              required: ['location'],
            },
          },
        },
      ]);
    });

    it('maps tool_choice {type: "auto"} to "auto"', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tool_choice: { type: 'auto' },
      };

      const result = convertAnthropicRequestToOpenAI(input);
      expect(result.tool_choice).toBe('auto');
    });

    it('maps tool_choice {type: "any"} to "required"', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tool_choice: { type: 'any' },
      };

      const result = convertAnthropicRequestToOpenAI(input);
      expect(result.tool_choice).toBe('required');
    });

    it('maps tool_choice {type: "none"} to "none"', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tool_choice: { type: 'none' },
      };

      const result = convertAnthropicRequestToOpenAI(input);
      expect(result.tool_choice).toBe('none');
    });

    it('maps tool_choice {type: "tool", name} to named choice', () => {
      const input: Anthropic.MessageCreateParams = {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tool_choice: { type: 'tool', name: 'get_weather' },
      };

      const result = convertAnthropicRequestToOpenAI(input);
      expect(result.tool_choice).toEqual({
        type: 'function',
        function: { name: 'get_weather' },
      });
    });
  });
});
