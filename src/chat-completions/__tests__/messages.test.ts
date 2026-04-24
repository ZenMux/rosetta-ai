import type OpenAI from "openai";
import type Anthropic from "@anthropic-ai/sdk";
import { ChatCompletionToMessagesConverter } from "../messages";

function makeMessage(overrides: Partial<Anthropic.Message> = {}): Anthropic.Message {
  return {
    id: "msg_123",
    type: "message",
    role: "assistant",
    model: "claude-sonnet-4-20250514",
    content: [{ type: "text", text: "Hello!", citations: null }],
    stop_reason: "end_turn",
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
    type: "message_start",
    message: {
      id: "msg_123",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4-20250514",
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: baseUsage,
      container: null,
    },
  };
}

describe("ChatCompletionToMessagesConverter", () => {
  const converter = new ChatCompletionToMessagesConverter();

  // ===== convertRequest =====

  describe("convertRequest", () => {
    describe("basic text conversation", () => {
      it("converts system + user messages", () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are helpful." },
            { role: "user", content: "Hello" },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.model).toBe("gpt-4o");
        expect(result.system).toEqual([{ type: "text", text: "You are helpful." }]);
        expect(result.messages).toEqual([
          { role: "user", content: [{ type: "text", text: "Hello" }] },
        ]);
      });

      it("converts developer role as system", () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: "gpt-4o",
          messages: [
            { role: "developer", content: "Be concise." },
            { role: "user", content: "Hi" },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.system).toEqual([{ type: "text", text: "Be concise." }]);
        expect(result.messages).toEqual([
          { role: "user", content: [{ type: "text", text: "Hi" }] },
        ]);
      });

      it("concatenates multiple system messages", () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are helpful." },
            { role: "system", content: "Be concise." },
            { role: "user", content: "Hi" },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.system).toEqual([
          { type: "text", text: "You are helpful." },
          { type: "text", text: "Be concise." },
        ]);
      });
    });

    describe("multi-turn conversation", () => {
      it("converts user and assistant turns", () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "Hello" },
            { role: "assistant", content: "Hi there!" },
            { role: "user", content: "How are you?" },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.messages).toEqual([
          { role: "user", content: [{ type: "text", text: "Hello" }] },
          { role: "assistant", content: [{ type: "text", text: "Hi there!" }] },
          { role: "user", content: [{ type: "text", text: "How are you?" }] },
        ]);
      });
    });

    describe("tool calling", () => {
      it("converts assistant tool_calls to tool_use blocks", () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "What is the weather?" },
            {
              role: "assistant",
              tool_calls: [
                {
                  id: "call_123",
                  type: "function",
                  function: { name: "get_weather", arguments: '{"location":"SF"}' },
                },
              ],
            },
            { role: "tool", tool_call_id: "call_123", content: "72°F" },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.messages).toEqual([
          { role: "user", content: [{ type: "text", text: "What is the weather?" }] },
          {
            role: "assistant",
            content: [
              { type: "tool_use", id: "call_123", name: "get_weather", input: { location: "SF" } },
            ],
          },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "call_123",
                content: [{ type: "text", text: "72°F" }],
              },
            ],
          },
        ]);
      });

      it("converts assistant with content + tool_calls", () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "Check weather" },
            {
              role: "assistant",
              content: "Let me check.",
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: { name: "get_weather", arguments: "{}" },
                },
              ],
            },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: [
            { type: "text", text: "Let me check." },
            { type: "tool_use", id: "call_1", name: "get_weather", input: {} },
          ],
        });
      });

      it("merges consecutive tool messages into one user message", () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "Do two things" },
            {
              role: "assistant",
              tool_calls: [
                { id: "c1", type: "function", function: { name: "fn1", arguments: "{}" } },
                { id: "c2", type: "function", function: { name: "fn2", arguments: "{}" } },
              ],
            },
            { role: "tool", tool_call_id: "c1", content: "result1" },
            { role: "tool", tool_call_id: "c2", content: "result2" },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.messages[2]).toEqual({
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "c1",
              content: [{ type: "text", text: "result1" }],
            },
            {
              type: "tool_result",
              tool_use_id: "c2",
              content: [{ type: "text", text: "result2" }],
            },
          ],
        });
      });
    });

    describe("multimodal content", () => {
      it("converts image_url with regular URL", () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: "gpt-4o",
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "What is this?" },
                { type: "image_url", image_url: { url: "https://example.com/img.png" } },
              ],
            },
          ],
        };

        const result = converter.convertRequest(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: [
            { type: "text", text: "What is this?" },
            { type: "image", source: { type: "url", url: "https://example.com/img.png" } },
          ],
        });
      });

      it("converts image_url with base64 data URI", () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: "gpt-4o",
          messages: [
            {
              role: "user",
              content: [
                { type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } },
              ],
            },
          ],
        };

        const result = converter.convertRequest(input);
        const content = result.messages[0].content as Anthropic.ImageBlockParam[];

        expect(content[0]).toEqual({
          type: "image",
          source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" },
        });
      });
    });

    describe("params mapping", () => {
      it("maps temperature, top_p, max_tokens, stop", () => {
        const input: OpenAI.ChatCompletionCreateParams = {
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          temperature: 0.7,
          top_p: 0.9,
          max_tokens: 1000,
          stop: ["END", "STOP"],
        };

        const result = converter.convertRequest(input);

        expect(result.temperature).toBe(0.7);
        expect(result.top_p).toBe(0.9);
        expect(result.max_tokens).toBe(1000);
        expect(result.stop_sequences).toEqual(["END", "STOP"]);
      });

      it("maps max_completion_tokens to max_tokens", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          max_completion_tokens: 500,
        });
        expect(result.max_tokens).toBe(500);
      });

      it("maps stop as string to stop_sequences array", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          stop: "END",
        });
        expect(result.stop_sequences).toEqual(["END"]);
      });

      it("defaults max_tokens to 4096 when not provided", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
        });
        expect(result.max_tokens).toBe(4096);
      });
    });

    describe("tools mapping", () => {
      it("converts function tools", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          tools: [
            {
              type: "function",
              function: {
                name: "get_weather",
                description: "Get weather info",
                parameters: {
                  type: "object",
                  properties: { location: { type: "string" } },
                  required: ["location"],
                },
              },
            },
          ],
        });

        expect(result.tools).toEqual([
          {
            name: "get_weather",
            description: "Get weather info",
            input_schema: {
              type: "object",
              properties: { location: { type: "string" } },
              required: ["location"],
            },
          },
        ]);
      });

      it('maps tool_choice "auto"', () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          tools: [{ type: "function", function: { name: "fn", parameters: { type: "object" } } }],
          tool_choice: "auto",
        });
        expect(result.tool_choice).toEqual({ type: "auto" });
      });

      it('maps tool_choice "required" to {type: "any"}', () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          tools: [{ type: "function", function: { name: "fn", parameters: { type: "object" } } }],
          tool_choice: "required",
        });
        expect(result.tool_choice).toEqual({ type: "any" });
      });

      it('maps tool_choice "none"', () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          tool_choice: "none",
        });
        expect(result.tool_choice).toEqual({ type: "none" });
      });

      it("maps named tool_choice", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          tools: [
            { type: "function", function: { name: "get_weather", parameters: { type: "object" } } },
          ],
          tool_choice: { type: "function", function: { name: "get_weather" } },
        });
        expect(result.tool_choice).toEqual({ type: "tool", name: "get_weather" });
      });
    });

    describe("response_format mapping", () => {
      it("maps json_schema response_format to output_config", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          response_format: {
            type: "json_schema",
            json_schema: {
              name: "response",
              schema: { type: "object", properties: { answer: { type: "string" } } },
            },
          },
        });
        expect(result.output_config).toEqual({
          format: {
            type: "json_schema",
            schema: { type: "object", properties: { answer: { type: "string" } } },
          },
        });
      });

      it("maps json_object response_format to output_config", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          response_format: { type: "json_object" },
        });
        expect(result.output_config).toEqual({
          format: { type: "json_schema", schema: { type: "object" } },
        });
      });

      it("does not set output_config for text format", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          response_format: { type: "text" },
        });
        expect(result.output_config).toEqual({});
      });
    });

    describe("reasoning_effort mapping", () => {
      it('maps "high" reasoning_effort to thinking enabled', () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          reasoning_effort: "high",
        } as any);
        expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 10240 });
      });

      it('maps "low" reasoning_effort to thinking enabled with small budget', () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          reasoning_effort: "low",
        } as any);
        expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 2048 });
      });

      it('maps "none" reasoning_effort to thinking disabled', () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          reasoning_effort: "none",
        } as any);
        expect(result.thinking).toEqual({ type: "disabled" });
      });
    });

    describe("user mapping", () => {
      it("maps user to metadata.user_id", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          user: "user-123",
        });
        expect(result.metadata).toEqual({ user_id: "user-123" });
      });
    });

    describe("parallel_tool_calls mapping", () => {
      it("maps parallel_tool_calls=false to disable_parallel_tool_use=true", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          tools: [{ type: "function", function: { name: "fn", parameters: { type: "object" } } }],
          tool_choice: "auto",
          parallel_tool_calls: false,
        } as any);
        expect(result.tool_choice).toEqual({ type: "auto", disable_parallel_tool_use: true });
      });

      it("does not set disable_parallel_tool_use when parallel_tool_calls is true", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          tools: [{ type: "function", function: { name: "fn", parameters: { type: "object" } } }],
          tool_choice: "auto",
          parallel_tool_calls: true,
        } as any);
        expect(result.tool_choice).toEqual({ type: "auto" });
      });
    });

    describe("stream mapping", () => {
      it("passes stream=true through", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          stream: true,
        } as any);
        expect((result as any).stream).toBe(true);
      });

      it("does not set stream when not provided", () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
        });
        expect((result as any).stream).toBeUndefined();
      });
    });

    describe("service_tier mapping", () => {
      it('maps service_tier "auto" through', () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          service_tier: "auto",
        } as any);
        expect(result.service_tier).toBe("auto");
      });
    });

    describe("reasoning_effort mapping (additional)", () => {
      it('maps "medium" reasoning_effort to thinking enabled with budget 5120', () => {
        const result = converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          reasoning_effort: "medium",
        } as any);
        expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 5120 });
      });
    });
  });

  // ===== convertResponse (Messages -> CC, backward) =====

  describe("convertResponse", () => {
    it("converts a basic text response", () => {
      const result = converter.convertResponse(makeMessage());

      expect(result.id).toBe("msg_123");
      expect(result.model).toBe("claude-sonnet-4-20250514");
      expect(result.object).toBe("chat.completion");
      expect(result.choices[0].message.content).toBe("Hello!");
      expect(result.choices[0].finish_reason).toBe("stop");
      expect(result.usage?.prompt_tokens).toBe(10);
      expect(result.usage?.completion_tokens).toBe(5);
      expect(result.usage?.total_tokens).toBe(15);
    });

    it("converts a tool_use response", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            {
              type: "tool_use",
              id: "toolu_123",
              name: "get_weather",
              input: { location: "SF" },
              caller: { type: "direct" },
            },
          ],
          stop_reason: "tool_use",
        })
      );

      expect(result.choices[0].finish_reason).toBe("tool_calls");
      expect(result.choices[0].message.content).toBeNull();
      expect(result.choices[0].message.tool_calls).toEqual([
        {
          id: "toolu_123",
          type: "function",
          function: { name: "get_weather", arguments: '{"location":"SF"}' },
        },
      ]);
    });

    it("converts mixed text + tool_use response", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            { type: "text", text: "Let me check.", citations: null },
            {
              type: "tool_use",
              id: "toolu_1",
              name: "get_weather",
              input: {},
              caller: { type: "direct" },
            },
          ],
          stop_reason: "tool_use",
        })
      );

      expect(result.choices[0].message.content).toBe("Let me check.");
      expect(result.choices[0].message.tool_calls).toEqual([
        { id: "toolu_1", type: "function", function: { name: "get_weather", arguments: "{}" } },
      ]);
    });

    it('maps stop_reason "end_turn" to "stop"', () => {
      expect(
        converter.convertResponse(makeMessage({ stop_reason: "end_turn" })).choices[0].finish_reason
      ).toBe("stop");
    });

    it('maps stop_reason "max_tokens" to "length"', () => {
      expect(
        converter.convertResponse(makeMessage({ stop_reason: "max_tokens" })).choices[0]
          .finish_reason
      ).toBe("length");
    });

    it('maps stop_reason "tool_use" to "tool_calls"', () => {
      expect(
        converter.convertResponse(makeMessage({ stop_reason: "tool_use" })).choices[0].finish_reason
      ).toBe("tool_calls");
    });

    it('maps stop_reason "stop_sequence" to "stop"', () => {
      expect(
        converter.convertResponse(makeMessage({ stop_reason: "stop_sequence" })).choices[0]
          .finish_reason
      ).toBe("stop");
    });

    it("concatenates multiple text blocks without separator", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            { type: "text", text: "Hello ", citations: null },
            { type: "text", text: "world", citations: null },
          ],
        })
      );
      expect(result.choices[0].message.content).toBe("Hello world");
    });

    it('maps stop_reason "pause_turn" to "stop"', () => {
      expect(
        converter.convertResponse(makeMessage({ stop_reason: "pause_turn" })).choices[0]
          .finish_reason
      ).toBe("stop");
    });

    it('maps stop_reason "refusal" to "content_filter"', () => {
      expect(
        converter.convertResponse(makeMessage({ stop_reason: "refusal" })).choices[0].finish_reason
      ).toBe("content_filter");
    });

    it('maps stop_reason "model_context_window_exceeded" to "length"', () => {
      expect(
        converter.convertResponse(
          makeMessage({ stop_reason: "model_context_window_exceeded" as any })
        ).choices[0].finish_reason
      ).toBe("length");
    });

    it("extracts thinking blocks into reasoning and reasoning_details", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            { type: "thinking", thinking: "Let me think...", signature: "sig_abc" } as any,
            { type: "text", text: "The answer is 42.", citations: null },
          ],
        })
      );

      const msg = result.choices[0].message as any;
      expect(msg.content).toBe("The answer is 42.");
      expect(msg.reasoning).toBe("Let me think...");
      expect(msg.reasoning_details).toEqual([
        {
          type: "reasoning.text",
          text: "Let me think...",
          signature: "sig_abc",
          format: "anthropic-claude-v1",
          index: 0,
        },
      ]);
    });

    it("converts server_tool_use blocks to tool_calls", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            {
              type: "server_tool_use",
              id: "srv_1",
              name: "web_search",
              input: { query: "test" },
              caller: { type: "direct" },
            } as any,
          ],
        })
      );

      expect(result.choices[0].message.tool_calls).toEqual([
        {
          id: "srv_1",
          type: "function",
          function: { name: "web_search", arguments: '{"query":"test"}' },
        },
      ]);
    });

    it("extracts web_search_tool_result as url_citation annotations", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            { type: "text", text: "Based on my search:", citations: null },
            {
              type: "web_search_tool_result",
              tool_use_id: "srv_1",
              content: [
                {
                  type: "web_search_result",
                  url: "https://example.com",
                  title: "Example",
                  encrypted_content: "x",
                  page_age: null,
                },
              ],
              caller: { type: "direct" },
            } as any,
          ],
        })
      );

      const msg = result.choices[0].message;
      expect(msg.annotations).toEqual([
        {
          type: "url_citation",
          url_citation: {
            title: "Example",
            url: "https://example.com",
            start_index: 0,
            end_index: 0,
          },
        },
      ]);
    });

    it("includes extended usage fields (cache, web_search)", () => {
      const result = converter.convertResponse(
        makeMessage({
          usage: {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 20,
            cache_read_input_tokens: 30,
            cache_creation: { ephemeral_5m_input_tokens: 15, ephemeral_1h_input_tokens: 5 } as any,
            inference_geo: null,
            server_tool_use: { web_search_requests: 3 } as any,
            service_tier: null,
          },
        })
      );

      const details = result.usage!.prompt_tokens_details as any;
      expect(details.cached_tokens).toBe(30);
      expect(details.ephemeral_5m_input_tokens).toBe(15);
      expect(details.ephemeral_1h_input_tokens).toBe(5);
      expect(details.web_search).toBe(3);
      expect(details.cache_creation_input_tokens).toBe(20);
    });
  });

  // ===== convertStream (Messages -> CC, backward) =====

  describe("convertStream", () => {
    describe("text streaming", () => {
      it("emits first chunk with role on message_start", () => {
        const c = new ChatCompletionToMessagesConverter();
        const result = c.convertStreamEvent(messageStart());

        expect(result).not.toBeNull();
        expect(result!.id).toBe("msg_123");
        expect(result!.model).toBe("claude-sonnet-4-20250514");
        expect(result!.choices[0].delta.role).toBe("assistant");
      });

      it("returns null for content_block_start (text)", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());

        const result = c.convertStreamEvent({
          type: "content_block_start",
          index: 0,
          content_block: { type: "text", text: "", citations: null },
        });

        expect(result).toBeNull();
      });

      it("emits text content for content_block_delta (text_delta)", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());

        const result = c.convertStreamEvent({
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "Hello" },
        });

        expect(result).not.toBeNull();
        expect(result!.choices[0].delta.content).toBe("Hello");
      });

      it("emits finish_reason on message_delta", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());

        const result = c.convertStreamEvent({
          type: "message_delta",
          delta: { stop_reason: "end_turn", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });

        expect(result).not.toBeNull();
        expect(result!.choices[0].finish_reason).toBe("stop");
      });

      it("emits final usage chunk on message_stop", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());
        c.convertStreamEvent({
          type: "message_delta",
          delta: { stop_reason: "end_turn", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });

        const result = c.convertStreamEvent({ type: "message_stop" });

        expect(result).not.toBeNull();
        expect(result!.choices).toEqual([]);
        expect(result!.usage?.prompt_tokens).toBe(10);
        expect(result!.usage?.completion_tokens).toBe(5);
        expect(result!.usage?.total_tokens).toBe(15);
        expect(result!.usage?.completion_tokens_details).toEqual({ reasoning_tokens: 0 });
      });
    });

    describe("tool call streaming", () => {
      it("emits tool_call start on content_block_start (tool_use)", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());

        const result = c.convertStreamEvent({
          type: "content_block_start",
          index: 0,
          content_block: {
            type: "tool_use",
            id: "toolu_1",
            name: "get_weather",
            input: {},
            caller: { type: "direct" },
          },
        });

        expect(result).not.toBeNull();
        const tc = result!.choices[0].delta.tool_calls![0];
        expect(tc.index).toBe(0);
        expect(tc.id).toBe("toolu_1");
        expect(tc.type).toBe("function");
        expect(tc.function!.name).toBe("get_weather");
      });

      it("emits tool_call arguments on content_block_delta (input_json_delta)", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());
        c.convertStreamEvent({
          type: "content_block_start",
          index: 0,
          content_block: {
            type: "tool_use",
            id: "toolu_1",
            name: "get_weather",
            input: {},
            caller: { type: "direct" },
          },
        });

        const result = c.convertStreamEvent({
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: '{"loc' },
        });

        expect(result).not.toBeNull();
        const delta = result!.choices[0].delta;
        expect(delta.content).toBe("");
        const tc = delta.tool_calls![0];
        expect(tc.index).toBe(0);
        expect(tc.type).toBe("function");
        expect(tc.function!.arguments).toBe('{"loc');
      });

      it("maps tool_use stop_reason to tool_calls finish_reason", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());

        const result = c.convertStreamEvent({
          type: "message_delta",
          delta: { stop_reason: "tool_use", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });

        expect(result!.choices[0].finish_reason).toBe("tool_calls");
      });

      it("maps max_tokens stop_reason to length finish_reason in stream", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());

        const result = c.convertStreamEvent({
          type: "message_delta",
          delta: { stop_reason: "max_tokens", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });

        expect(result!.choices[0].finish_reason).toBe("length");
      });
    });

    describe("content_block_stop", () => {
      it("emits empty content chunk on content_block_stop", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());
        const result = c.convertStreamEvent({ type: "content_block_stop", index: 0 });
        expect(result).not.toBeNull();
        expect(result!.choices[0].delta.content).toBe("");
      });
    });

    describe("text in content_block_start", () => {
      it("emits content chunk when text block has initial non-empty text", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());
        const result = c.convertStreamEvent({
          type: "content_block_start",
          index: 0,
          content_block: { type: "text", text: "Hello", citations: null },
        });
        expect(result).not.toBeNull();
        expect(result!.choices[0].delta.content).toBe("Hello");
      });

      it("returns null when text block is empty", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());
        const result = c.convertStreamEvent({
          type: "content_block_start",
          index: 0,
          content_block: { type: "text", text: "", citations: null },
        });
        expect(result).toBeNull();
      });
    });

    describe("thinking streaming", () => {
      it("emits reasoning chunk for thinking_delta", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());

        const result = c.convertStreamEvent({
          type: "content_block_delta",
          index: 0,
          delta: { type: "thinking_delta", thinking: "Let me think..." } as any,
        });

        expect(result).not.toBeNull();
        const delta = result!.choices[0].delta as any;
        expect(delta.reasoning).toBe("Let me think...");
        expect(delta.reasoning_details).toEqual([
          {
            type: "reasoning.text",
            text: "Let me think...",
            format: "anthropic-claude-v1",
            index: 0,
          },
        ]);
      });

      it("emits reasoning_details with signature for signature_delta", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());

        const result = c.convertStreamEvent({
          type: "content_block_delta",
          index: 0,
          delta: { type: "signature_delta", signature: "sig_xyz" } as any,
        });

        expect(result).not.toBeNull();
        const delta = result!.choices[0].delta as any;
        expect(delta.reasoning).toBeNull();
        expect(delta.reasoning_details).toEqual([
          { type: "reasoning.text", signature: "sig_xyz", format: "anthropic-claude-v1", index: 0 },
        ]);
      });
    });

    describe("server_tool_use streaming", () => {
      it("treats server_tool_use same as tool_use in content_block_start", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());

        const result = c.convertStreamEvent({
          type: "content_block_start",
          index: 0,
          content_block: {
            type: "server_tool_use",
            id: "srv_1",
            name: "web_search",
            input: {},
            caller: { type: "direct" },
          } as any,
        });

        expect(result).not.toBeNull();
        const tc = result!.choices[0].delta.tool_calls![0];
        expect(tc.id).toBe("srv_1");
        expect(tc.function!.name).toBe("web_search");
      });
    });

    describe("web_search_tool_result streaming", () => {
      it("collects annotations and emits them on message_delta", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());

        // content_block_start for web_search_tool_result returns null
        const startResult = c.convertStreamEvent({
          type: "content_block_start",
          index: 0,
          content_block: {
            type: "web_search_tool_result",
            tool_use_id: "srv_1",
            content: [
              {
                type: "web_search_result",
                url: "https://example.com",
                title: "Example",
                encrypted_content: "x",
                page_age: null,
              },
            ],
            caller: { type: "direct" },
          } as any,
        });
        expect(startResult).toBeNull();

        const deltaResult = c.convertStreamEvent({
          type: "message_delta",
          delta: { stop_reason: "end_turn", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });

        expect((deltaResult!.choices[0].delta as any).annotations).toEqual([
          {
            type: "url_citation",
            url_citation: {
              title: "Example",
              url: "https://example.com",
              start_index: 0,
              end_index: 0,
            },
          },
        ]);
      });
    });

    describe("extended usage in message_stop", () => {
      it("includes cache and web_search counts from message_start + message_delta", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent({
          type: "message_start",
          message: {
            id: "msg_x",
            type: "message",
            role: "assistant",
            model: "claude-sonnet-4-20250514",
            content: [],
            stop_reason: null,
            stop_sequence: null,
            usage: {
              input_tokens: 100,
              output_tokens: 0,
              cache_read_input_tokens: 30,
              cache_creation_input_tokens: 20,
              cache_creation: {
                ephemeral_5m_input_tokens: 15,
                ephemeral_1h_input_tokens: 5,
              } as any,
              inference_geo: null,
              server_tool_use: null,
              service_tier: null,
            },
            container: null,
          },
        });
        c.convertStreamEvent({
          type: "message_delta",
          delta: { stop_reason: "end_turn", stop_sequence: null, container: null },
          usage: {
            output_tokens: 50,
            input_tokens: null,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            server_tool_use: { web_search_requests: 3 } as any,
          },
        });

        const result = c.convertStreamEvent({ type: "message_stop" });

        expect(result!.usage!.prompt_tokens).toBe(100);
        expect(result!.usage!.completion_tokens).toBe(50);
        expect(result!.usage!.total_tokens).toBe(150);
        const details = result!.usage!.prompt_tokens_details as any;
        expect(details.cached_tokens).toBe(30);
        expect(details.ephemeral_5m_input_tokens).toBe(15);
        expect(details.ephemeral_1h_input_tokens).toBe(5);
        expect(details.web_search).toBe(3);
        expect(details.cache_creation_input_tokens).toBe(20);
      });
    });

    describe("extended stop_reason mappings", () => {
      it("maps pause_turn to stop", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());
        const result = c.convertStreamEvent({
          type: "message_delta",
          delta: { stop_reason: "pause_turn", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });
        expect(result!.choices[0].finish_reason).toBe("stop");
      });

      it("maps refusal to content_filter", () => {
        const c = new ChatCompletionToMessagesConverter();
        c.convertStreamEvent(messageStart());
        const result = c.convertStreamEvent({
          type: "message_delta",
          delta: { stop_reason: "refusal", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });
        expect(result!.choices[0].finish_reason).toBe("content_filter");
      });
    });

    describe("convertStream (AsyncIterable)", () => {
      async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
        const result: T[] = [];
        for await (const item of iter) {
          result.push(item);
        }
        return result;
      }

      async function* toAsync<T>(items: T[]): AsyncIterable<T> {
        for (const item of items) {
          yield item;
        }
      }

      it("converts a full text stream", async () => {
        const events: Anthropic.RawMessageStreamEvent[] = [
          messageStart(),
          {
            type: "content_block_start",
            index: 0,
            content_block: { type: "text", text: "", citations: null },
          },
          { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } },
          { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: " world" } },
          { type: "content_block_stop", index: 0 },
          {
            type: "message_delta",
            delta: { stop_reason: "end_turn", stop_sequence: null, container: null },
            usage: baseDeltaUsage,
          },
          { type: "message_stop" },
        ];

        const c = new ChatCompletionToMessagesConverter();
        const chunks = await collect(c.convertStream(toAsync(events)));

        expect(chunks.length).toBeGreaterThan(0);
        expect(chunks[0].choices[0].delta.role).toBe("assistant");
        const textChunks = chunks.filter(
          ch => ch.choices[0]?.delta?.content && ch.choices[0].delta.content !== ""
        );
        expect(textChunks.length).toBe(2);
        const finishChunk = chunks.find(ch => ch.choices[0]?.finish_reason === "stop");
        expect(finishChunk).toBeDefined();
        const usageChunk = chunks.find(ch => ch.choices.length === 0 && ch.usage);
        expect(usageChunk).toBeDefined();
        expect(usageChunk!.usage!.completion_tokens).toBe(5);
      });

      it("filters out null events (content_block_start with empty text)", async () => {
        const events: Anthropic.RawMessageStreamEvent[] = [
          messageStart(),
          {
            type: "content_block_start",
            index: 0,
            content_block: { type: "text", text: "", citations: null },
          },
          { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hi" } },
          { type: "content_block_stop", index: 0 },
          {
            type: "message_delta",
            delta: { stop_reason: "end_turn", stop_sequence: null, container: null },
            usage: baseDeltaUsage,
          },
          { type: "message_stop" },
        ];

        const c = new ChatCompletionToMessagesConverter();
        const chunks = await collect(c.convertStream(toAsync(events)));

        // content_block_start with empty text returns null and should be filtered
        const allDefined = chunks.every(ch => ch != null);
        expect(allDefined).toBe(true);
      });
    });
  });
});
