import type OpenAI from "openai";
import type Anthropic from "@anthropic-ai/sdk";
import { MessagesToChatCompletionConverter } from "../chat-completions";

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

describe("MessagesToChatCompletionConverter", () => {
  const converter = new MessagesToChatCompletionConverter();

  // ===== convertRequest =====

  describe("convertRequest", () => {
    describe("basic text conversation", () => {
      it("converts system string to system message", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          system: "You are helpful.",
          messages: [{ role: "user", content: "Hello" }],
        });

        expect(result.model).toBe("claude-sonnet-4-20250514");
        expect(result.messages[0]).toEqual({ role: "system", content: "You are helpful." });
        expect(result.messages[1]).toEqual({ role: "user", content: "Hello" });
      });

      it("converts system content blocks to system message", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          system: [
            { type: "text", text: "You are helpful." },
            { type: "text", text: "Be concise." },
          ],
          messages: [{ role: "user", content: "Hello" }],
        });

        expect(result.messages[0]).toEqual({
          role: "system",
          content: "You are helpful.\nBe concise.",
        });
      });

      it("converts user content blocks to string when text only", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
        });

        expect(result.messages[0]).toEqual({ role: "user", content: "Hello" });
      });
    });

    describe("multi-turn conversation", () => {
      it("converts user and assistant turns", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "Hello" },
            { role: "assistant", content: "Hi there!" },
            { role: "user", content: "How are you?" },
          ],
        });

        expect(result.messages).toEqual([
          { role: "user", content: "Hello" },
          { role: "assistant", content: "Hi there!" },
          { role: "user", content: "How are you?" },
        ]);
      });
    });

    describe("tool calling", () => {
      it("converts tool_use blocks to assistant tool_calls", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "What is the weather?" },
            {
              role: "assistant",
              content: [
                {
                  type: "tool_use",
                  id: "toolu_123",
                  name: "get_weather",
                  input: { location: "SF" },
                },
              ],
            },
            {
              role: "user",
              content: [{ type: "tool_result", tool_use_id: "toolu_123", content: "72°F" }],
            },
          ],
        });

        expect(result.messages[1]).toEqual({
          role: "assistant",
          tool_calls: [
            {
              id: "toolu_123",
              type: "function",
              function: { name: "get_weather", arguments: '{"location":"SF"}' },
            },
          ],
        });
        expect(result.messages[2]).toEqual({
          role: "tool",
          tool_call_id: "toolu_123",
          content: "72°F",
        });
      });

      it("converts assistant with text + tool_use blocks", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "Check weather" },
            {
              role: "assistant",
              content: [
                { type: "text", text: "Let me check." },
                { type: "tool_use", id: "toolu_1", name: "get_weather", input: {} },
              ],
            },
          ],
        });

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: "Let me check.",
          tool_calls: [
            { id: "toolu_1", type: "function", function: { name: "get_weather", arguments: "{}" } },
          ],
        });
      });

      it("splits user message with multiple tool_results into separate tool messages", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "Do two things" },
            {
              role: "assistant",
              content: [
                { type: "tool_use", id: "c1", name: "fn1", input: {} },
                { type: "tool_use", id: "c2", name: "fn2", input: {} },
              ],
            },
            {
              role: "user",
              content: [
                { type: "tool_result", tool_use_id: "c1", content: "result1" },
                { type: "tool_result", tool_use_id: "c2", content: "result2" },
              ],
            },
          ],
        });

        expect(result.messages[2]).toEqual({
          role: "tool",
          tool_call_id: "c1",
          content: "result1",
        });
        expect(result.messages[3]).toEqual({
          role: "tool",
          tool_call_id: "c2",
          content: "result2",
        });
      });
    });

    describe("multimodal content", () => {
      it("converts image block with URL source", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "What is this?" },
                { type: "image", source: { type: "url", url: "https://example.com/img.png" } },
              ],
            },
          ],
        });

        const msg = result.messages[0] as OpenAI.ChatCompletionUserMessageParam;
        const content = msg.content as OpenAI.ChatCompletionContentPart[];

        expect(content[0]).toEqual({ type: "text", text: "What is this?" });
        expect(content[1]).toEqual({
          type: "image_url",
          image_url: { url: "https://example.com/img.png" },
        });
      });

      it("converts image block with base64 source", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "image",
                  source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" },
                },
              ],
            },
          ],
        });

        const msg = result.messages[0] as OpenAI.ChatCompletionUserMessageParam;
        const content = msg.content as OpenAI.ChatCompletionContentPart[];

        expect(content[0]).toEqual({
          type: "image_url",
          image_url: { url: "data:image/png;base64,iVBORw0KGgo=" },
        });
      });
    });

    describe("document content", () => {
      it("converts document block with base64 source to file content part", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "document",
                  source: { type: "base64", media_type: "application/pdf", data: "JVBER..." },
                  title: "report.pdf",
                },
              ],
            },
          ],
        });

        const msg = result.messages[0] as OpenAI.ChatCompletionUserMessageParam;
        const content = msg.content as OpenAI.ChatCompletionContentPart[];
        expect(content[0]).toEqual({
          type: "file",
          file: { file_data: "data:application/pdf;base64,JVBER...", filename: "report.pdf" },
        });
      });

      it("converts document block with plain text source to text part", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "document",
                  source: { type: "text", media_type: "text/plain", data: "Hello world" },
                },
              ],
            },
          ],
        });

        const msg = result.messages[0] as OpenAI.ChatCompletionUserMessageParam;
        const content = msg.content as OpenAI.ChatCompletionContentPart[];
        expect(content[0]).toEqual({ type: "text", text: "Hello world" });
      });
    });

    describe("assistant thinking blocks", () => {
      it("converts thinking blocks to reasoning and reasoning_details on assistant message", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 16000,
          messages: [
            { role: "user", content: "Solve this" },
            {
              role: "assistant",
              content: [
                { type: "thinking", thinking: "Let me reason...", signature: "sig_abc" },
                { type: "text", text: "The answer is 42." },
              ],
            },
            { role: "user", content: "Thanks" },
          ],
        });

        const msg = result.messages[1] as any;
        expect(msg.role).toBe("assistant");
        expect(msg.content).toBe("The answer is 42.");
        expect(msg.reasoning).toBe("Let me reason...");
        expect(msg.reasoning_details).toEqual([
          {
            type: "reasoning.text",
            text: "Let me reason...",
            signature: "sig_abc",
            format: "anthropic-claude-v1",
            index: 0,
          },
        ]);
      });
    });

    describe("web_search tool conversion", () => {
      it("converts web_search_20250305 tool to web_search_options", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Search for something" }],
          tools: [{ name: "web_search", type: "web_search_20250305", max_uses: 3 }],
        });

        expect(result.tools).toBeUndefined();
        expect((result as any).web_search_options).toBeDefined();
        expect((result as any).web_search_options.max_uses).toBe(3);
      });

      it("separates function tools from web_search tools", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          tools: [
            { name: "get_weather", input_schema: { type: "object" } },
            { name: "web_search", type: "web_search_20250305" },
          ],
        });

        expect(result.tools).toHaveLength(1);
        const tool = result.tools![0] as OpenAI.ChatCompletionFunctionTool;
        expect(tool.function.name).toBe("get_weather");
        expect((result as any).web_search_options).toBeDefined();
      });
    });

    describe("params mapping", () => {
      it("maps temperature, top_p, max_tokens, stop_sequences", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          messages: [{ role: "user", content: "Hi" }],
          temperature: 0.7,
          top_p: 0.9,
          stop_sequences: ["END", "STOP"],
        });

        expect(result.temperature).toBe(0.7);
        expect(result.top_p).toBe(0.9);
        expect(result.max_tokens).toBe(1000);
        expect(result.stop).toEqual(["END", "STOP"]);
      });
    });

    describe("tools mapping", () => {
      it("converts Anthropic tools to OpenAI function tools", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          tools: [
            {
              name: "get_weather",
              description: "Get weather info",
              input_schema: {
                type: "object",
                properties: { location: { type: "string" } },
                required: ["location"],
              },
            },
          ],
        });

        expect(result.tools).toEqual([
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
        ]);
      });

      it('maps tool_choice {type: "auto"} to "auto"', () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          tool_choice: { type: "auto" },
        });
        expect(result.tool_choice).toBe("auto");
      });

      it('maps tool_choice {type: "any"} to "required"', () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          tool_choice: { type: "any" },
        });
        expect(result.tool_choice).toBe("required");
      });

      it('maps tool_choice {type: "none"} to "none"', () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          tool_choice: { type: "none" },
        });
        expect(result.tool_choice).toBe("none");
      });

      it('maps tool_choice {type: "tool", name} to named choice', () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          tool_choice: { type: "tool", name: "get_weather" },
        });
        expect(result.tool_choice).toEqual({ type: "function", function: { name: "get_weather" } });
      });
    });

    describe("output_config mapping", () => {
      it("maps json_schema output_config to response_format", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          output_config: {
            format: {
              type: "json_schema",
              schema: { type: "object", properties: { answer: { type: "string" } } },
            },
          },
        });
        expect(result.response_format).toEqual({
          type: "json_schema",
          json_schema: {
            name: "response",
            schema: { type: "object", properties: { answer: { type: "string" } } },
            strict: true,
          },
        });
      });

      it("maps output_config without format to text", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          output_config: {},
        });
        expect(result.response_format).toEqual({ type: "text" });
      });
    });

    describe("thinking mapping", () => {
      it("maps thinking enabled to reasoning_effort", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 16000,
          messages: [{ role: "user", content: "Hi" }],
          thinking: { type: "enabled", budget_tokens: 10000 },
        });
        expect((result as any).reasoning_effort).toBe("high");
      });

      it("maps thinking disabled to reasoning_effort none", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          thinking: { type: "disabled" },
        });
        expect((result as any).reasoning_effort).toBe("none");
      });
    });

    describe("metadata mapping", () => {
      it("maps metadata.user_id to user", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          metadata: { user_id: "user-123" },
        });
        expect(result.user).toBe("user-123");
      });
    });

    describe("parallel_tool_calls mapping", () => {
      it("maps disable_parallel_tool_use=true to parallel_tool_calls=false", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          tools: [{ name: "fn", input_schema: { type: "object" } }],
          tool_choice: { type: "auto", disable_parallel_tool_use: true },
        });
        expect(result.tool_choice).toBe("auto");
        expect((result as any).parallel_tool_calls).toBe(false);
      });

      it("does not set parallel_tool_calls when disable_parallel_tool_use is not set", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          tool_choice: { type: "auto" },
        });
        expect(result.tool_choice).toBe("auto");
        expect((result as any).parallel_tool_calls).toBeUndefined();
      });
    });

    describe("stream mapping", () => {
      it("passes stream=true through", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          stream: true,
        } as any);
        expect((result as any).stream).toBe(true);
      });

      it("does not set stream when not provided", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
        });
        expect((result as any).stream).toBeUndefined();
      });
    });

    describe("service_tier mapping", () => {
      it('maps service_tier "auto" through', () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          service_tier: "auto",
        });
        expect((result as any).service_tier).toBe("auto");
      });

      it('maps service_tier "standard_only" to "default"', () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          service_tier: "standard_only",
        });
        expect((result as any).service_tier).toBe("default");
      });
    });

    describe("thinking mapping (additional)", () => {
      it('maps thinking adaptive to reasoning_effort "medium"', () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 16000,
          messages: [{ role: "user", content: "Hi" }],
          thinking: { type: "adaptive" },
        });
        expect((result as any).reasoning_effort).toBe("medium");
      });
    });
  });

  // ===== convertResponse =====

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

  // ===== convertStream =====

  describe("convertStream", () => {
    describe("text streaming", () => {
      it("emits first chunk with role on message_start", () => {
        const c = new MessagesToChatCompletionConverter();
        const result = c.convertStream(messageStart());

        expect(result).not.toBeNull();
        expect(result!.id).toBe("msg_123");
        expect(result!.model).toBe("claude-sonnet-4-20250514");
        expect(result!.choices[0].delta.role).toBe("assistant");
      });

      it("returns null for content_block_start (text)", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
          type: "content_block_start",
          index: 0,
          content_block: { type: "text", text: "", citations: null },
        });

        expect(result).toBeNull();
      });

      it("emits text content for content_block_delta (text_delta)", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "Hello" },
        });

        expect(result).not.toBeNull();
        expect(result!.choices[0].delta.content).toBe("Hello");
      });

      it("emits finish_reason on message_delta", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
          type: "message_delta",
          delta: { stop_reason: "end_turn", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });

        expect(result).not.toBeNull();
        expect(result!.choices[0].finish_reason).toBe("stop");
      });

      it("emits final usage chunk on message_stop", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());
        c.convertStream({
          type: "message_delta",
          delta: { stop_reason: "end_turn", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });

        const result = c.convertStream({ type: "message_stop" });

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
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
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
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());
        c.convertStream({
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

        const result = c.convertStream({
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
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
          type: "message_delta",
          delta: { stop_reason: "tool_use", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });

        expect(result!.choices[0].finish_reason).toBe("tool_calls");
      });

      it("maps max_tokens stop_reason to length finish_reason in stream", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
          type: "message_delta",
          delta: { stop_reason: "max_tokens", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });

        expect(result!.choices[0].finish_reason).toBe("length");
      });
    });

    describe("content_block_stop", () => {
      it("emits empty content chunk on content_block_stop", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());
        const result = c.convertStream({ type: "content_block_stop", index: 0 });
        expect(result).not.toBeNull();
        expect(result!.choices[0].delta.content).toBe("");
      });
    });

    describe("text in content_block_start", () => {
      it("emits content chunk when text block has initial non-empty text", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());
        const result = c.convertStream({
          type: "content_block_start",
          index: 0,
          content_block: { type: "text", text: "Hello", citations: null },
        });
        expect(result).not.toBeNull();
        expect(result!.choices[0].delta.content).toBe("Hello");
      });

      it("returns null when text block is empty", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());
        const result = c.convertStream({
          type: "content_block_start",
          index: 0,
          content_block: { type: "text", text: "", citations: null },
        });
        expect(result).toBeNull();
      });
    });

    describe("thinking streaming", () => {
      it("emits reasoning chunk for thinking_delta", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
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
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
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
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        const result = c.convertStream({
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
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());

        // content_block_start for web_search_tool_result returns null
        const startResult = c.convertStream({
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

        const deltaResult = c.convertStream({
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
        const c = new MessagesToChatCompletionConverter();
        c.convertStream({
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
        c.convertStream({
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

        const result = c.convertStream({ type: "message_stop" });

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
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());
        const result = c.convertStream({
          type: "message_delta",
          delta: { stop_reason: "pause_turn", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });
        expect(result!.choices[0].finish_reason).toBe("stop");
      });

      it("maps refusal to content_filter", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStream(messageStart());
        const result = c.convertStream({
          type: "message_delta",
          delta: { stop_reason: "refusal", stop_sequence: null, container: null },
          usage: baseDeltaUsage,
        });
        expect(result!.choices[0].finish_reason).toBe("content_filter");
      });
    });
  });
});
