import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';
import { convertOpenAIRequestToAnthropic } from '../request';

describe('convertOpenAIRequestToAnthropic', () => {
  describe('basic text conversation', () => {
    it('converts system + user messages', () => {
      const input: OpenAI.ChatCompletionCreateParams = {
        model: 'gpt-4o',
        messages: [
          { role: 'system', content: 'You are helpful.' },
          { role: 'user', content: 'Hello' },
        ],
      };

      const result = convertOpenAIRequestToAnthropic(input);

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

      const result = convertOpenAIRequestToAnthropic(input);

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

      const result = convertOpenAIRequestToAnthropic(input);

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

      const result = convertOpenAIRequestToAnthropic(input);

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
                function: {
                  name: 'get_weather',
                  arguments: '{"location":"SF"}',
                },
              },
            ],
          },
          {
            role: 'tool',
            tool_call_id: 'call_123',
            content: '72°F',
          },
        ],
      };

      const result = convertOpenAIRequestToAnthropic(input);

      expect(result.messages).toEqual([
        { role: 'user', content: [{ type: 'text', text: 'What is the weather?' }] },
        {
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'call_123',
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
              tool_use_id: 'call_123',
              content: [{ type: 'text', text: '72°F' }],
            },
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
              {
                id: 'call_1',
                type: 'function',
                function: { name: 'get_weather', arguments: '{}' },
              },
            ],
          },
        ],
      };

      const result = convertOpenAIRequestToAnthropic(input);

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

      const result = convertOpenAIRequestToAnthropic(input);

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

      const result = convertOpenAIRequestToAnthropic(input);

      expect(result.messages[0]).toEqual({
        role: 'user',
        content: [
          { type: 'text', text: 'What is this?' },
          {
            type: 'image',
            source: { type: 'url', url: 'https://example.com/img.png' },
          },
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
              {
                type: 'image_url',
                image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' },
              },
            ],
          },
        ],
      };

      const result = convertOpenAIRequestToAnthropic(input);
      const content = result.messages[0].content as Anthropic.ImageBlockParam[];

      expect(content[0]).toEqual({
        type: 'image',
        source: {
          type: 'base64',
          media_type: 'image/png',
          data: 'iVBORw0KGgo=',
        },
      });
    });
  });

  describe('params mapping', () => {
    it('maps temperature, top_p, max_tokens, stop', () => {
      const input: OpenAI.ChatCompletionCreateParams = {
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'Hi' }],
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 1000,
        stop: ['END', 'STOP'],
      };

      const result = convertOpenAIRequestToAnthropic(input);

      expect(result.temperature).toBe(0.7);
      expect(result.top_p).toBe(0.9);
      expect(result.max_tokens).toBe(1000);
      expect(result.stop_sequences).toEqual(['END', 'STOP']);
    });

    it('maps max_completion_tokens to max_tokens', () => {
      const input: OpenAI.ChatCompletionCreateParams = {
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'Hi' }],
        max_completion_tokens: 500,
      };

      const result = convertOpenAIRequestToAnthropic(input);

      expect(result.max_tokens).toBe(500);
    });

    it('maps stop as string to stop_sequences array', () => {
      const input: OpenAI.ChatCompletionCreateParams = {
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'Hi' }],
        stop: 'END',
      };

      const result = convertOpenAIRequestToAnthropic(input);

      expect(result.stop_sequences).toEqual(['END']);
    });

    it('defaults max_tokens to 4096 when not provided', () => {
      const input: OpenAI.ChatCompletionCreateParams = {
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'Hi' }],
      };

      const result = convertOpenAIRequestToAnthropic(input);

      expect(result.max_tokens).toBe(4096);
    });
  });

  describe('tools mapping', () => {
    it('converts function tools', () => {
      const input: OpenAI.ChatCompletionCreateParams = {
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_weather',
              description: 'Get weather info',
              parameters: {
                type: 'object',
                properties: {
                  location: { type: 'string' },
                },
                required: ['location'],
              },
            },
          },
        ],
      };

      const result = convertOpenAIRequestToAnthropic(input);

      expect(result.tools).toEqual([
        {
          name: 'get_weather',
          description: 'Get weather info',
          input_schema: {
            type: 'object',
            properties: {
              location: { type: 'string' },
            },
            required: ['location'],
          },
        },
      ]);
    });

    it('maps tool_choice "auto" to {type: "auto"}', () => {
      const input: OpenAI.ChatCompletionCreateParams = {
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [{ type: 'function', function: { name: 'fn', parameters: { type: 'object' } } }],
        tool_choice: 'auto',
      };

      const result = convertOpenAIRequestToAnthropic(input);
      expect(result.tool_choice).toEqual({ type: 'auto' });
    });

    it('maps tool_choice "required" to {type: "any"}', () => {
      const input: OpenAI.ChatCompletionCreateParams = {
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [{ type: 'function', function: { name: 'fn', parameters: { type: 'object' } } }],
        tool_choice: 'required',
      };

      const result = convertOpenAIRequestToAnthropic(input);
      expect(result.tool_choice).toEqual({ type: 'any' });
    });

    it('maps tool_choice "none" to {type: "none"}', () => {
      const input: OpenAI.ChatCompletionCreateParams = {
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'Hi' }],
        tool_choice: 'none',
      };

      const result = convertOpenAIRequestToAnthropic(input);
      expect(result.tool_choice).toEqual({ type: 'none' });
    });

    it('maps named tool_choice to {type: "tool", name}', () => {
      const input: OpenAI.ChatCompletionCreateParams = {
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [{ type: 'function', function: { name: 'get_weather', parameters: { type: 'object' } } }],
        tool_choice: { type: 'function', function: { name: 'get_weather' } },
      };

      const result = convertOpenAIRequestToAnthropic(input);
      expect(result.tool_choice).toEqual({ type: 'tool', name: 'get_weather' });
    });
  });
});
