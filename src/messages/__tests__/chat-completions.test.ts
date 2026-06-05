import type OpenAI from "openai";
import type Anthropic from "@anthropic-ai/sdk";
import { MessagesToChatCompletionConverter } from "../chat-completions";

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

      it("keeps text from search_result and tool_reference blocks in tool_result content", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "tool_result",
                  tool_use_id: "c1",
                  content: [
                    { type: "text", text: "Summary:" },
                    {
                      type: "search_result",
                      source: "https://example.com",
                      title: "Example",
                      content: [
                        { type: "text", text: "first hit" },
                        { type: "text", text: "second hit" },
                      ],
                    },
                    { type: "tool_reference", tool_name: "web_search" },
                  ],
                },
              ],
            },
          ],
        });

        expect(result.messages[0]).toEqual({
          role: "tool",
          tool_call_id: "c1",
          content: "Summary:\nfirst hit\nsecond hit\nweb_search",
        });
      });

      it("skips image blocks in tool_result content (unsupported by tool role)", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "tool_result",
                  tool_use_id: "c1",
                  content: [
                    { type: "text", text: "see image" },
                    {
                      type: "image",
                      source: { type: "base64", media_type: "image/png", data: "iVBOR..." },
                    },
                  ],
                },
              ],
            },
          ],
        });

        expect(result.messages[0]).toEqual({
          role: "tool",
          tool_call_id: "c1",
          content: "see image",
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

      it("unpacks document block with content source into nested text and image parts", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "document",
                  source: {
                    type: "content",
                    content: [
                      { type: "text", text: "Figure 1" },
                      {
                        type: "image",
                        source: { type: "base64", media_type: "image/png", data: "iVBOR..." },
                      },
                    ],
                  },
                },
              ],
            },
          ],
        });

        const msg = result.messages[0] as OpenAI.ChatCompletionUserMessageParam;
        const content = msg.content as OpenAI.ChatCompletionContentPart[];
        expect(content).toEqual([
          { type: "text", text: "Figure 1" },
          { type: "image_url", image_url: { url: "data:image/png;base64,iVBOR..." } },
        ]);
      });

      it("converts document block with string content source to text part", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "document",
                  source: { type: "content", content: "Inline text" },
                },
              ],
            },
          ],
        });

        const msg = result.messages[0] as OpenAI.ChatCompletionUserMessageParam;
        const content = msg.content as OpenAI.ChatCompletionContentPart[];
        expect(content[0]).toEqual({ type: "text", text: "Inline text" });
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
        expect(result.max_completion_tokens).toBe(1000);
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

      it("sets stream_options.include_usage when streaming", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          stream: true,
        } as any);
        expect((result as any).stream_options).toEqual({ include_usage: true });
      });

      it("does not set stream when not provided", () => {
        const result = converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
        });
        expect((result as any).stream).toBeUndefined();
        expect((result as any).stream_options).toBeUndefined();
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

  // ===== convertResponse (CC -> Messages, backward) =====

  describe("convertResponse", () => {
    it("converts a basic text response", () => {
      const input: OpenAI.ChatCompletion = {
        id: "chatcmpl-123",
        object: "chat.completion",
        created: 1700000000,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "Hello!", refusal: null },
            finish_reason: "stop",
            logprobs: null,
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      };

      const result = converter.convertResponse(input);

      expect(result.id).toBe("chatcmpl-123");
      expect(result.model).toBe("gpt-4o");
      expect(result.content).toEqual([{ type: "text", text: "Hello!", citations: null }]);
      expect(result.stop_reason).toBe("end_turn");
      expect(result.usage.input_tokens).toBe(10);
      expect(result.usage.output_tokens).toBe(5);
    });

    it("converts a tool call response", () => {
      const input: OpenAI.ChatCompletion = {
        id: "chatcmpl-456",
        object: "chat.completion",
        created: 1700000000,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              refusal: null,
              tool_calls: [
                {
                  id: "call_abc",
                  type: "function",
                  function: { name: "get_weather", arguments: '{"location":"SF"}' },
                },
              ],
            },
            finish_reason: "tool_calls",
            logprobs: null,
          },
        ],
      };

      const result = converter.convertResponse(input);

      expect(result.stop_reason).toBe("tool_use");
      expect(result.content).toEqual([
        {
          type: "tool_use",
          id: "call_abc",
          name: "get_weather",
          input: { location: "SF" },
          caller: { type: "direct" },
        },
      ]);
    });

    it("converts mixed content + tool calls", () => {
      const input: OpenAI.ChatCompletion = {
        id: "chatcmpl-789",
        object: "chat.completion",
        created: 1700000000,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "Let me check.",
              refusal: null,
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: { name: "get_weather", arguments: "{}" },
                },
              ],
            },
            finish_reason: "tool_calls",
            logprobs: null,
          },
        ],
      };

      const result = converter.convertResponse(input);

      expect(result.content).toEqual([
        { type: "text", text: "Let me check.", citations: null },
        {
          type: "tool_use",
          id: "call_1",
          name: "get_weather",
          input: {},
          caller: { type: "direct" },
        },
      ]);
    });

    it('maps finish_reason "length" to "max_tokens"', () => {
      const input: OpenAI.ChatCompletion = {
        id: "id",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "truncated", refusal: null },
            finish_reason: "length",
            logprobs: null,
          },
        ],
      };
      expect(converter.convertResponse(input).stop_reason).toBe("max_tokens");
    });

    it('maps finish_reason "content_filter" to "refusal"', () => {
      const input: OpenAI.ChatCompletion = {
        id: "id",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "", refusal: null },
            finish_reason: "content_filter",
            logprobs: null,
          },
        ],
      };
      expect(converter.convertResponse(input).stop_reason).toBe("refusal");
    });

    it("handles null content with no tool calls", () => {
      const input: OpenAI.ChatCompletion = {
        id: "id",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: null, refusal: null },
            finish_reason: "stop",
            logprobs: null,
          },
        ],
      };
      expect(converter.convertResponse(input).content).toEqual([
        { type: "text", text: "", citations: null },
      ]);
    });

    it("converts reasoning to thinking block", () => {
      const input: OpenAI.ChatCompletion = {
        id: "id",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "Answer is 42.",
              refusal: null,
              reasoning: "Let me think...",
            } as any,
            finish_reason: "stop",
            logprobs: null,
          },
        ],
      };
      const result = converter.convertResponse(input);
      expect(result.content[0]).toEqual({
        type: "thinking",
        thinking: "Let me think...",
        signature: "",
      });
      expect(result.content[1]).toEqual({ type: "text", text: "Answer is 42.", citations: null });
    });

    it("converts annotations to web_search_tool_result", () => {
      const input: OpenAI.ChatCompletion = {
        id: "id",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "Search result.",
              refusal: null,
              annotations: [
                {
                  type: "url_citation",
                  url_citation: {
                    title: "Example",
                    url: "https://example.com",
                    start_index: 0,
                    end_index: 10,
                  },
                },
              ],
            },
            finish_reason: "stop",
            logprobs: null,
          },
        ],
      };
      const result = converter.convertResponse(input);
      const wsBlock = result.content.find(b => b.type === "web_search_tool_result") as any;
      expect(wsBlock).toBeDefined();
      expect(wsBlock.content[0].title).toBe("Example");
      expect(wsBlock.content[0].url).toBe("https://example.com");
    });

    it("maps function_call finish_reason to tool_use", () => {
      const input: OpenAI.ChatCompletion = {
        id: "id",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: null, refusal: null },
            finish_reason: "function_call",
            logprobs: null,
          },
        ],
      };
      expect(converter.convertResponse(input).stop_reason).toBe("tool_use");
    });

    it("converts usage with cache details", () => {
      const input: OpenAI.ChatCompletion = {
        id: "id",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "Hi", refusal: null },
            finish_reason: "stop",
            logprobs: null,
          },
        ],
        usage: {
          prompt_tokens: 100,
          completion_tokens: 50,
          total_tokens: 150,
          prompt_tokens_details: { cached_tokens: 30 },
        },
      };
      const result = converter.convertResponse(input);
      expect(result.usage.input_tokens).toBe(100);
      expect(result.usage.output_tokens).toBe(50);
      expect(result.usage.cache_read_input_tokens).toBe(30);
    });

    it("enriches usage with OpenAI-style fields", () => {
      const input: OpenAI.ChatCompletion = {
        id: "id",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "Hi", refusal: null },
            finish_reason: "stop",
            logprobs: null,
          },
        ],
        usage: {
          prompt_tokens: 100,
          completion_tokens: 50,
          total_tokens: 150,
          prompt_tokens_details: { cached_tokens: 30, audio_tokens: 4 },
          completion_tokens_details: { reasoning_tokens: 12 },
        } as any,
      };
      const usage = converter.convertResponse(input).usage as any;
      expect(usage.prompt_tokens).toBe(100);
      expect(usage.completion_tokens).toBe(50);
      expect(usage.total_tokens).toBe(150);
      // original *_details fields are merged in, computed values take precedence
      expect(usage.completion_tokens_details).toEqual({ reasoning_tokens: 12 });
      expect(usage.prompt_tokens_details).toEqual({ cached_tokens: 30, audio_tokens: 4 });
      expect(usage.audio_input_tokens).toBe(4);
      expect(usage.service_tier).toBe("standard");
      expect(usage.cache_creation_input_tokens).toBe(0);
    });

    it("preserves extra fields from origin usage details", () => {
      const input: OpenAI.ChatCompletion = {
        id: "id",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "Hi", refusal: null },
            finish_reason: "stop",
            logprobs: null,
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
          prompt_tokens_details: { cached_tokens: 3, audio_tokens: 1 },
          completion_tokens_details: {
            reasoning_tokens: 2,
            accepted_prediction_tokens: 7,
          },
        } as any,
      };
      const usage = converter.convertResponse(input).usage as any;
      expect(usage.prompt_tokens_details).toEqual({ cached_tokens: 3, audio_tokens: 1 });
      expect(usage.completion_tokens_details).toEqual({
        reasoning_tokens: 2,
        accepted_prediction_tokens: 7,
      });
    });

    it("counts web_search requests in usage", () => {
      const input: OpenAI.ChatCompletion = {
        id: "id",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "Hi", refusal: null },
            finish_reason: "stop",
            logprobs: null,
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
          prompt_tokens_details: { cached_tokens: 0, web_search: 2 },
        } as any,
      };
      const usage = converter.convertResponse(input).usage as any;
      expect(usage.prompt_tokens_details.web_search).toBe(2);
      expect(usage.server_tool_use).toEqual({ web_fetch_requests: 0, web_search_requests: 2 });
    });
  });

  // ===== convertStream (CC -> Messages, backward) =====

  describe("convertStream", () => {
    function makeChunk(
      overrides: Partial<OpenAI.ChatCompletionChunk> & {
        delta?: Partial<OpenAI.ChatCompletionChunk.Choice.Delta>;
        finish_reason?: OpenAI.ChatCompletionChunk.Choice["finish_reason"];
      } = {}
    ): OpenAI.ChatCompletionChunk {
      const { delta = {}, finish_reason = null, ...rest } = overrides;
      return {
        id: "chatcmpl-123",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: "gpt-4o",
        choices: [{ index: 0, delta, finish_reason }],
        ...rest,
      };
    }

    describe("text streaming", () => {
      it("emits message_start on first chunk", () => {
        const c = new MessagesToChatCompletionConverter();
        const events = c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

        expect(events.length).toBe(1);
        expect(events[0].type).toBe("message_start");
      });

      it("emits content_block_start + content_block_delta for first text chunk", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

        const events = c.convertStreamChunk(makeChunk({ delta: { content: "Hello" } }));

        expect(events.length).toBe(2);
        expect(events[0].type).toBe("content_block_start");
        expect((events[0] as Anthropic.RawContentBlockStartEvent).content_block.type).toBe("text");
        expect(events[1].type).toBe("content_block_delta");
        const d = (events[1] as Anthropic.RawContentBlockDeltaEvent).delta;
        expect(d.type).toBe("text_delta");
        expect((d as Anthropic.TextDelta).text).toBe("Hello");
      });

      it("emits only content_block_delta for subsequent text chunks", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
        c.convertStreamChunk(makeChunk({ delta: { content: "Hello" } }));

        const events = c.convertStreamChunk(makeChunk({ delta: { content: " world" } }));

        expect(events.length).toBe(1);
        expect(events[0].type).toBe("content_block_delta");
        expect((events[0] as Anthropic.RawContentBlockDeltaEvent).delta.type).toBe("text_delta");
      });

      it("emits stop events on finish", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
        c.convertStreamChunk(makeChunk({ delta: { content: "Hi" } }));

        const events = c.convertStreamChunk(makeChunk({ finish_reason: "stop" }));
        const types = events.map(e => e.type);

        expect(types).toContain("content_block_stop");
        expect(types).toContain("message_delta");
        expect(types).toContain("message_stop");
      });

      it('maps finish_reason "tool_calls" to stop_reason "tool_use" in stream', () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

        const events = c.convertStreamChunk(makeChunk({ finish_reason: "tool_calls" }));
        const msgDelta = events.find(
          e => e.type === "message_delta"
        ) as Anthropic.RawMessageDeltaEvent;

        expect(msgDelta.delta.stop_reason).toBe("tool_use");
      });

      it('maps finish_reason "length" to stop_reason "max_tokens" in stream', () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

        const events = c.convertStreamChunk(makeChunk({ finish_reason: "length" }));
        const msgDelta = events.find(
          e => e.type === "message_delta"
        ) as Anthropic.RawMessageDeltaEvent;

        expect(msgDelta.delta.stop_reason).toBe("max_tokens");
      });
    });

    describe("tool call streaming", () => {
      it("emits content_block_start for tool_use when tool_call starts", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

        const events = c.convertStreamChunk(
          makeChunk({
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_1",
                  type: "function",
                  function: { name: "get_weather", arguments: "" },
                },
              ],
            },
          })
        );

        const startEvents = events.filter(e => e.type === "content_block_start");
        expect(startEvents.length).toBe(1);
        const startEvent = startEvents[0] as Anthropic.RawContentBlockStartEvent;
        expect(startEvent.content_block.type).toBe("tool_use");
        expect((startEvent.content_block as any).name).toBe("get_weather");
      });

      it("emits input_json_delta for tool call arguments", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
        c.convertStreamChunk(
          makeChunk({
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_1",
                  type: "function",
                  function: { name: "get_weather", arguments: "" },
                },
              ],
            },
          })
        );

        const events = c.convertStreamChunk(
          makeChunk({
            delta: { tool_calls: [{ index: 0, function: { arguments: '{"loc' } }] },
          })
        );

        expect(events.length).toBe(1);
        expect(events[0].type).toBe("content_block_delta");
        const delta = (events[0] as Anthropic.RawContentBlockDeltaEvent).delta;
        expect(delta.type).toBe("input_json_delta");
        expect((delta as Anthropic.InputJSONDelta).partial_json).toBe('{"loc');
      });
    });

    describe("text + tool call mixed", () => {
      it("handles text followed by tool calls", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
        c.convertStreamChunk(makeChunk({ delta: { content: "Let me check." } }));

        const events = c.convertStreamChunk(
          makeChunk({
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_1",
                  type: "function",
                  function: { name: "fn", arguments: "" },
                },
              ],
            },
          })
        );

        const types = events.map(e => e.type);
        expect(types).toContain("content_block_stop");
        expect(types).toContain("content_block_start");
      });
    });

    describe("reasoning streaming", () => {
      it("emits thinking block events for reasoning delta", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

        const events = c.convertStreamChunk(
          makeChunk({
            delta: { reasoning: "Let me think..." } as any,
          })
        );

        const types = events.map(e => e.type);
        expect(types).toContain("content_block_start");
        expect(types).toContain("content_block_delta");
        const startEvent = events.find(
          e => e.type === "content_block_start"
        ) as Anthropic.RawContentBlockStartEvent;
        expect(startEvent.content_block.type).toBe("thinking");
        const deltaEvent = events.find(
          e => e.type === "content_block_delta"
        ) as Anthropic.RawContentBlockDeltaEvent;
        expect(deltaEvent.delta.type).toBe("thinking_delta");
        expect((deltaEvent.delta as Anthropic.ThinkingDelta).thinking).toBe("Let me think...");
      });

      it("transitions from reasoning to text with content_block_stop", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
        c.convertStreamChunk(makeChunk({ delta: { reasoning: "thinking..." } as any }));

        const events = c.convertStreamChunk(makeChunk({ delta: { content: "Answer" } }));

        const types = events.map(e => e.type);
        expect(types[0]).toBe("content_block_stop");
        expect(types).toContain("content_block_start");
        expect(types).toContain("content_block_delta");
      });
    });

    describe("annotations streaming", () => {
      it("emits web_search_tool_result for annotations delta", () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
        c.convertStreamChunk(makeChunk({ delta: { content: "Results:" } }));

        const events = c.convertStreamChunk(
          makeChunk({
            delta: {
              annotations: [
                {
                  type: "url_citation",
                  url_citation: {
                    title: "Example",
                    url: "https://example.com",
                    start_index: 0,
                    end_index: 5,
                  },
                },
              ],
            } as any,
          })
        );

        const types = events.map(e => e.type);
        expect(types).toContain("content_block_stop");
        expect(types).toContain("content_block_start");
        const startEvent = events.find(
          e => e.type === "content_block_start"
        ) as Anthropic.RawContentBlockStartEvent;
        expect(startEvent.content_block.type).toBe("web_search_tool_result");
        expect((startEvent.content_block as any).content[0].title).toBe("Example");
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
        const chunks = [
          makeChunk({ delta: { role: "assistant" } }),
          makeChunk({ delta: { content: "Hello" } }),
          makeChunk({ delta: { content: " world" } }),
          makeChunk({ finish_reason: "stop" }),
        ];

        const c = new MessagesToChatCompletionConverter();
        const events = await collect(c.convertStream(toAsync(chunks)));
        const types = events.map(e => e.type);

        expect(types[0]).toBe("message_start");
        expect(types).toContain("content_block_start");
        expect(types.filter(t => t === "content_block_delta").length).toBe(2);
        expect(types).toContain("content_block_stop");
        expect(types).toContain("message_delta");
        expect(types).toContain("message_stop");
      });

      it("converts a tool call stream", async () => {
        const chunks = [
          makeChunk({ delta: { role: "assistant" } }),
          makeChunk({
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_1",
                  type: "function",
                  function: { name: "get_weather", arguments: "" },
                },
              ],
            },
          }),
          makeChunk({
            delta: { tool_calls: [{ index: 0, function: { arguments: '{"city":"SF"}' } }] },
          }),
          makeChunk({ finish_reason: "tool_calls" }),
        ];

        const c = new MessagesToChatCompletionConverter();
        const events = await collect(c.convertStream(toAsync(chunks)));
        const types = events.map(e => e.type);

        expect(types).toContain("message_start");
        expect(types).toContain("content_block_start");
        expect(types.filter(t => t === "content_block_delta").length).toBeGreaterThanOrEqual(1);
        expect(types).toContain("message_stop");
      });

      it("reports usage from the trailing usage-only chunk", async () => {
        const c = new MessagesToChatCompletionConverter();
        // Enable include_usage so terminal events defer to the usage chunk.
        c.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          stream: true,
        } as any);

        const chunks = [
          makeChunk({ delta: { role: "assistant" } }),
          makeChunk({ delta: { content: "Hello" } }),
          makeChunk({ finish_reason: "stop" }),
          // trailing usage-only chunk (empty choices) like OpenAI emits
          {
            id: "chatcmpl-123",
            object: "chat.completion.chunk",
            created: 1700000000,
            model: "gpt-4o",
            choices: [],
            usage: { prompt_tokens: 12, completion_tokens: 7, total_tokens: 19 },
          } as unknown as OpenAI.ChatCompletionChunk,
        ];

        const events = await collect(c.convertStream(toAsync(chunks)));
        const msgDelta = events.find(
          e => e.type === "message_delta"
        ) as Anthropic.RawMessageDeltaEvent;

        expect(msgDelta).toBeDefined();
        expect(msgDelta.usage.output_tokens).toBe(7);
        expect(msgDelta.usage.input_tokens).toBe(12);
        expect((msgDelta.usage as any).completion_tokens).toBe(7);
        expect((msgDelta.usage as any).prompt_tokens).toBe(12);
        expect((msgDelta.usage as any).total_tokens).toBe(19);
        expect(msgDelta.delta.stop_reason).toBe("end_turn");

        // message_delta/message_stop must appear exactly once, after the usage chunk
        const types = events.map(e => e.type);
        expect(types.filter(t => t === "message_delta").length).toBe(1);
        expect(types.filter(t => t === "message_stop").length).toBe(1);
        expect(types[types.length - 1]).toBe("message_stop");
      });

      it("flushes terminal events when usage chunk never arrives", async () => {
        const c = new MessagesToChatCompletionConverter();
        c.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          stream: true,
        } as any);

        const chunks = [
          makeChunk({ delta: { role: "assistant" } }),
          makeChunk({ delta: { content: "Hello" } }),
          makeChunk({ finish_reason: "stop" }),
          // no trailing usage chunk
        ];

        const events = await collect(c.convertStream(toAsync(chunks)));
        const types = events.map(e => e.type);

        expect(types.filter(t => t === "message_delta").length).toBe(1);
        expect(types.filter(t => t === "message_stop").length).toBe(1);
        const msgDelta = events.find(
          e => e.type === "message_delta"
        ) as Anthropic.RawMessageDeltaEvent;
        expect(msgDelta.usage.output_tokens).toBe(0);
        expect(msgDelta.delta.stop_reason).toBe("end_turn");
      });
    });
  });
});
