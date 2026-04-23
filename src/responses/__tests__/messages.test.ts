import type Anthropic from "@anthropic-ai/sdk";
import type OpenAI from "openai";
import { ResponsesToMessagesConverter } from "../messages";

describe("ResponsesToMessagesConverter", () => {
  const converter = new ResponsesToMessagesConverter();

  describe("convertRequest", () => {
    it("converts string input to user message", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hello",
      });

      expect(result.model).toBe("claude-sonnet-4-20250514");
      expect(result.messages).toEqual([
        { role: "user", content: [{ type: "text", text: "Hello" }] },
      ]);
    });

    it("converts instructions to system", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
        instructions: "Be helpful.",
      });

      expect(result.system).toBe("Be helpful.");
    });

    it("maps max_output_tokens to max_tokens", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
        max_output_tokens: 2000,
      });
      expect(result.max_tokens).toBe(2000);
    });

    it("uses default max_tokens when not specified", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
      });
      expect(result.max_tokens).toBe(4096);
    });

    it("maps temperature and top_p", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
        temperature: 0.7,
        top_p: 0.9,
      });
      expect(result.temperature).toBe(0.7);
      expect(result.top_p).toBe(0.9);
    });

    it("converts input array with user/assistant messages", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          { role: "user", content: "Hello" },
          { role: "assistant", content: "Hi there!" },
          { role: "user", content: "How are you?" },
        ],
      });

      expect(result.messages[0]).toEqual({
        role: "user",
        content: [{ type: "text", text: "Hello" }],
      });
      expect(result.messages[1]).toEqual({
        role: "assistant",
        content: [{ type: "text", text: "Hi there!" }],
      });
      expect(result.messages[2]).toEqual({
        role: "user",
        content: [{ type: "text", text: "How are you?" }],
      });
    });

    it("converts function_call and function_call_output", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          { role: "user", content: "What is the weather?" },
          {
            type: "function_call",
            id: "fc_1",
            call_id: "call_1",
            name: "get_weather",
            arguments: '{"city":"SF"}',
            status: "completed",
          },
          {
            type: "function_call_output",
            id: "fco_1",
            call_id: "call_1",
            output: "72°F",
          },
        ],
      });

      expect(result.messages[1]).toEqual({
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: "call_1",
            name: "get_weather",
            input: { city: "SF" },
          },
        ],
      });
      expect(result.messages[2]).toEqual({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "call_1",
            content: [{ type: "text", text: "72°F" }],
          },
        ],
      });
    });

    it("merges consecutive function_call items into one assistant message", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          { role: "user", content: "Do two things" },
          {
            type: "function_call",
            id: "fc_1",
            call_id: "call_1",
            name: "fn1",
            arguments: "{}",
            status: "completed",
          },
          {
            type: "function_call",
            id: "fc_2",
            call_id: "call_2",
            name: "fn2",
            arguments: "{}",
            status: "completed",
          },
          { type: "function_call_output", id: "fco_1", call_id: "call_1", output: "r1" },
          { type: "function_call_output", id: "fco_2", call_id: "call_2", output: "r2" },
        ],
      });

      expect(result.messages[1]).toEqual({
        role: "assistant",
        content: [
          { type: "tool_use", id: "call_1", name: "fn1", input: {} },
          { type: "tool_use", id: "call_2", name: "fn2", input: {} },
        ],
      });
      // tool results grouped in one user message
      expect(result.messages[2].role).toBe("user");
      const userContent = result.messages[2].content as any[];
      expect(userContent).toHaveLength(2);
      expect(userContent[0].type).toBe("tool_result");
      expect(userContent[1].type).toBe("tool_result");
    });

    it("converts function tools", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
        tools: [
          {
            type: "function",
            name: "get_weather",
            description: "Get weather",
            parameters: { type: "object", properties: { city: { type: "string" } } },
            strict: true,
          },
        ],
      });

      expect(result.tools![0]).toEqual({
        name: "get_weather",
        description: "Get weather",
        input_schema: { type: "object", properties: { city: { type: "string" } } },
      });
    });

    it("converts web_search tools", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
        tools: [{ type: "web_search_preview", search_context_size: "medium" }],
      } as any);

      expect(result.tools).toBeDefined();
      expect((result.tools![0] as any).type).toBe("web_search_20250305");
    });

    it("maps tool_choice", () => {
      expect(
        converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          input: "Hi",
          tool_choice: "auto",
        }).tool_choice
      ).toEqual({ type: "auto" });

      expect(
        converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          input: "Hi",
          tool_choice: "required",
        }).tool_choice
      ).toEqual({ type: "any" });

      expect(
        converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          input: "Hi",
          tool_choice: "none",
        }).tool_choice
      ).toEqual({ type: "none" });
    });

    it("maps parallel_tool_calls false to disable_parallel_tool_use", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
        tool_choice: "auto",
        parallel_tool_calls: false,
      } as any);

      expect(result.tool_choice).toEqual({ type: "auto", disable_parallel_tool_use: true });
    });

    it("maps reasoning.effort to thinking config", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
        reasoning: { effort: "high" },
      });

      expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 10240 });
    });

    it("maps text.format json_schema to output_config", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
        text: {
          format: {
            type: "json_schema",
            name: "answer",
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

    it("maps service_tier", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
        service_tier: "auto",
      } as any);
      expect(result.service_tier).toBe("auto");
    });

    it("extracts system messages from input to system field", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          { role: "system", content: "You are an assistant." } as any,
          { role: "user", content: "Hello" },
        ],
      });

      expect(result.system).toBe("You are an assistant.");
      expect(result.messages[0]).toEqual({
        role: "user",
        content: [{ type: "text", text: "Hello" }],
      });
    });

    it("converts input_image with data URI", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          {
            role: "user",
            type: "message",
            content: [
              { type: "input_text", text: "What is this?" },
              { type: "input_image", image_url: "data:image/png;base64,abc123" },
            ],
          },
        ],
      } as any);

      const userContent = result.messages[0].content as any[];
      expect(userContent[0]).toEqual({ type: "text", text: "What is this?" });
      expect(userContent[1]).toEqual({
        type: "image",
        source: { type: "base64", media_type: "image/png", data: "abc123" },
      });
    });

    it("converts input_image with URL", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          {
            role: "user",
            type: "message",
            content: [{ type: "input_image", image_url: "https://example.com/img.png" }],
          },
        ],
      } as any);

      const userContent = result.messages[0].content as any[];
      expect(userContent[0]).toEqual({
        type: "image",
        source: { type: "url", url: "https://example.com/img.png" },
      });
    });

    it("maps metadata.user_id to metadata", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
        metadata: { user_id: "user_456" },
      } as any);

      expect(result.metadata).toBeDefined();
      expect((result.metadata as any).user_id).toBe("user_456");
    });
  });

  // ===== convertResponse (Messages → Responses, backward) =====

  describe("convertResponse", () => {
    function makeMessage(overrides: Partial<Anthropic.Message> = {}): Anthropic.Message {
      return {
        id: "msg_123",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-20250514" as Anthropic.Model,
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

    it("converts a basic text response", () => {
      const result = converter.convertResponse(makeMessage());

      expect(result.id).toBe("msg_123");
      expect(result.model).toBe("claude-sonnet-4-20250514");
      expect(result.object).toBe("response");
      expect(result.status).toBe("completed");
      expect(result.usage?.input_tokens).toBe(10);
      expect(result.usage?.output_tokens).toBe(5);

      const msgOutput = result.output.find((o: any) => o.type === "message") as any;
      expect(msgOutput).toBeDefined();
      expect(msgOutput.content[0].type).toBe("output_text");
      expect(msgOutput.content[0].text).toBe("Hello!");
    });

    it("converts tool_use to function_call output items", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            {
              type: "tool_use",
              id: "tu_1",
              name: "get_weather",
              input: { city: "SF" },
              caller: { type: "direct" },
            },
          ],
          stop_reason: "tool_use",
        })
      );

      const fcOutput = result.output.find((o: any) => o.type === "function_call") as any;
      expect(fcOutput).toBeDefined();
      expect(fcOutput.name).toBe("get_weather");
      expect(fcOutput.arguments).toBe('{"city":"SF"}');
      expect(fcOutput.call_id).toBe("tu_1");
      expect(result.status).toBe("completed");
    });

    it("converts thinking blocks to reasoning output", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            { type: "thinking", thinking: "Let me think...", signature: "sig" },
            { type: "text", text: "42", citations: null },
          ],
        })
      );

      const reasoningOutput = result.output.find((o: any) => o.type === "reasoning") as any;
      expect(reasoningOutput).toBeDefined();
      expect(reasoningOutput.summary[0].text).toBe("Let me think...");

      const msgOutput = result.output.find((o: any) => o.type === "message") as any;
      expect(msgOutput.content[0].text).toBe("42");
    });

    it("maps max_tokens stop_reason to incomplete status", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [{ type: "text", text: "Partial", citations: null }],
          stop_reason: "max_tokens",
        })
      );

      expect(result.status).toBe("incomplete");
      expect(result.incomplete_details).toEqual({ reason: "max_output_tokens" });
    });

    it("counts web_search_tool_result for usage", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            {
              type: "web_search_tool_result",
              tool_use_id: "ws_1",
              content: [
                {
                  type: "web_search_result",
                  title: "Example",
                  url: "https://example.com",
                  encrypted_content: "",
                  page_age: null,
                },
              ],
              caller: { type: "direct" },
            } as any,
            { type: "text", text: "Result", citations: null },
          ],
        })
      );

      expect(result.status).toBe("completed");
      const msgOutput = result.output.find((o: any) => o.type === "message") as any;
      expect(msgOutput.content[0].text).toBe("Result");
    });

    it("converts usage with cache tokens", () => {
      const result = converter.convertResponse(
        makeMessage({
          usage: {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: 30,
            cache_creation: null,
            inference_geo: null,
            server_tool_use: null,
            service_tier: null,
          },
        })
      );

      expect(result.usage?.input_tokens).toBe(100);
      expect(result.usage?.output_tokens).toBe(50);
      expect(result.usage?.input_tokens_details?.cached_tokens).toBe(30);
    });
  });

  // ===== convertStreamEvent (Messages → Responses, backward) =====

  describe("convertStreamEvent", () => {
    it("emits response.created and response.in_progress on message_start", () => {
      const c = new ResponsesToMessagesConverter();
      const events = c.convertStreamEvent({
        type: "message_start",
        message: {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-4-20250514" as Anthropic.Model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: 10,
            output_tokens: 0,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            cache_creation: null,
            inference_geo: null,
            server_tool_use: null,
            service_tier: null,
          },
          container: null,
        },
      });

      const types = events.map(e => e.type);
      expect(types).toContain("response.created");
      expect(types).toContain("response.in_progress");
    });

    it("emits output_text.delta for text content", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent({
        type: "message_start",
        message: {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-4-20250514" as Anthropic.Model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: 10,
            output_tokens: 0,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            cache_creation: null,
            inference_geo: null,
            server_tool_use: null,
            service_tier: null,
          },
          container: null,
        },
      });

      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "", citations: null },
      });

      const events = c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "Hello" },
      });

      const textDelta = events.find(e => e.type === "response.output_text.delta") as any;
      expect(textDelta).toBeDefined();
      expect(textDelta.delta).toBe("Hello");
    });

    it("emits output_item.added for function_call on tool_use block", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent({
        type: "message_start",
        message: {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-4-20250514" as Anthropic.Model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: 10,
            output_tokens: 0,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            cache_creation: null,
            inference_geo: null,
            server_tool_use: null,
            service_tier: null,
          },
          container: null,
        },
      });

      const events = c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: {
          type: "tool_use",
          id: "tu_1",
          name: "get_weather",
          input: {},
          caller: { type: "direct" },
        },
      });

      const itemAdded = events.find(e => e.type === "response.output_item.added") as any;
      expect(itemAdded).toBeDefined();
      expect(itemAdded.item.type).toBe("function_call");
      expect(itemAdded.item.name).toBe("get_weather");
    });

    it("emits function_call_arguments.delta for input_json_delta", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent({
        type: "message_start",
        message: {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-4-20250514" as Anthropic.Model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: 10,
            output_tokens: 0,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            cache_creation: null,
            inference_geo: null,
            server_tool_use: null,
            service_tier: null,
          },
          container: null,
        },
      });
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: {
          type: "tool_use",
          id: "tu_1",
          name: "fn",
          input: {},
          caller: { type: "direct" },
        },
      });

      const events = c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: '{"x' },
      });

      const argsDelta = events.find(
        e => e.type === "response.function_call_arguments.delta"
      ) as any;
      expect(argsDelta).toBeDefined();
      expect(argsDelta.delta).toBe('{"x');
    });

    it("emits reasoning_summary_text.delta for thinking_delta", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent({
        type: "message_start",
        message: {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-4-20250514" as Anthropic.Model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: 10,
            output_tokens: 0,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            cache_creation: null,
            inference_geo: null,
            server_tool_use: null,
            service_tier: null,
          },
          container: null,
        },
      });
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "thinking", thinking: "", signature: "" },
      });

      const events = c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "Thinking..." } as any,
      });

      const reasoningDelta = events.find(
        e => e.type === "response.reasoning_summary_text.delta"
      ) as any;
      expect(reasoningDelta).toBeDefined();
      expect(reasoningDelta.delta).toBe("Thinking...");
    });

    it("emits response.completed on message_stop", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent({
        type: "message_start",
        message: {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-4-20250514" as Anthropic.Model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: 10,
            output_tokens: 0,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            cache_creation: null,
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
          output_tokens: 5,
          input_tokens: null,
          cache_creation_input_tokens: null,
          cache_read_input_tokens: null,
          server_tool_use: null,
        },
      });

      const events = c.convertStreamEvent({ type: "message_stop" });

      const completed = events.find(e => e.type === "response.completed") as any;
      expect(completed).toBeDefined();
      expect(completed.response.status).toBe("completed");
      expect(completed.response.usage.input_tokens).toBe(10);
      expect(completed.response.usage.output_tokens).toBe(5);
    });

    it("emits response.incomplete on max_tokens stop", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent({
        type: "message_start",
        message: {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-4-20250514" as Anthropic.Model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: 10,
            output_tokens: 0,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            cache_creation: null,
            inference_geo: null,
            server_tool_use: null,
            service_tier: null,
          },
          container: null,
        },
      });
      c.convertStreamEvent({
        type: "message_delta",
        delta: { stop_reason: "max_tokens", stop_sequence: null, container: null },
        usage: {
          output_tokens: 100,
          input_tokens: null,
          cache_creation_input_tokens: null,
          cache_read_input_tokens: null,
          server_tool_use: null,
        },
      });

      const events = c.convertStreamEvent({ type: "message_stop" });

      const incomplete = events.find(e => e.type === "response.incomplete") as any;
      expect(incomplete).toBeDefined();
    });
  });

  // ===== convertStream (AsyncIterable) =====

  describe("convertStream", () => {
    it("converts a full stream", async () => {
      const c = new ResponsesToMessagesConverter();

      async function* makeStream(): AsyncIterable<Anthropic.RawMessageStreamEvent> {
        yield {
          type: "message_start",
          message: {
            id: "msg_1",
            type: "message",
            role: "assistant",
            model: "claude-sonnet-4-20250514" as Anthropic.Model,
            content: [],
            stop_reason: null,
            stop_sequence: null,
            usage: {
              input_tokens: 10,
              output_tokens: 0,
              cache_creation_input_tokens: null,
              cache_read_input_tokens: null,
              cache_creation: null,
              inference_geo: null,
              server_tool_use: null,
              service_tier: null,
            },
            container: null,
          },
        };
        yield {
          type: "content_block_start",
          index: 0,
          content_block: { type: "text", text: "", citations: null },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "Hello" },
        };
        yield { type: "content_block_stop", index: 0 };
        yield {
          type: "message_delta",
          delta: { stop_reason: "end_turn", stop_sequence: null, container: null },
          usage: {
            output_tokens: 5,
            input_tokens: null,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            server_tool_use: null,
          },
        };
        yield { type: "message_stop" };
      }

      const events: OpenAI.Responses.ResponseStreamEvent[] = [];
      for await (const event of c.convertStream(makeStream())) {
        events.push(event);
      }

      const types = events.map(e => e.type);
      expect(types).toContain("response.created");
      expect(types).toContain("response.in_progress");
      expect(types).toContain("response.output_text.delta");
      expect(types).toContain("response.completed");

      const textDelta = events.find(e => e.type === "response.output_text.delta") as any;
      expect(textDelta.delta).toBe("Hello");
    });
  });

  describe("convertRequest - advanced input items", () => {
    it("converts assistant message input with output_text", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Hello" }],
          },
        ],
      } as any);

      expect(result.messages[0]).toEqual({
        role: "assistant",
        content: [{ type: "text", text: "Hello" }],
      });
    });

    it("converts system message input to system field", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          {
            type: "message",
            role: "system",
            content: [{ type: "input_text", text: "Be helpful." }],
          },
          { role: "user", content: "Hi" },
        ],
      } as any);

      expect(result.system).toBe("Be helpful.");
    });

    it("converts developer message input to system field", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          { role: "developer", content: "You are an assistant." } as any,
          { role: "user", content: "Hi" },
        ],
      });

      expect(result.system).toBe("You are an assistant.");
    });

    it("converts input_image with data URI to image block", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_image", image_url: "data:image/png;base64,abc123" }],
          },
        ],
      } as any);

      const content = result.messages[0].content as any[];
      expect(content[0].type).toBe("image");
      expect(content[0].source.type).toBe("base64");
      expect(content[0].source.data).toBe("abc123");
    });

    it("converts input_image with URL to image block", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_image", image_url: "https://example.com/img.png" }],
          },
        ],
      } as any);

      const content = result.messages[0].content as any[];
      expect(content[0].type).toBe("image");
      expect(content[0].source.type).toBe("url");
    });

    it("converts input_file with data URI to document block", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_file", file_data: "data:application/pdf;base64,pdfdata" }],
          },
        ],
      } as any);

      const content = result.messages[0].content as any[];
      expect(content[0].type).toBe("document");
    });

    it("converts input_file with URL to document block", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_file", file_data: "https://example.com/doc.pdf" }],
          },
        ],
      } as any);

      const content = result.messages[0].content as any[];
      expect(content[0].type).toBe("document");
      expect(content[0].source.type).toBe("url");
    });

    it("merges consecutive function_call_output into one user message", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          { role: "user", content: "Do things" },
          {
            type: "function_call",
            id: "fc_1",
            call_id: "c1",
            name: "fn1",
            arguments: "{}",
            status: "completed",
          },
          {
            type: "function_call",
            id: "fc_2",
            call_id: "c2",
            name: "fn2",
            arguments: "{}",
            status: "completed",
          },
          { type: "function_call_output", id: "fco_1", call_id: "c1", output: "r1" },
          { type: "function_call_output", id: "fco_2", call_id: "c2", output: "r2" },
        ],
      });

      const toolMsg = result.messages[2];
      expect(toolMsg.role).toBe("user");
      const content = toolMsg.content as any[];
      expect(content).toHaveLength(2);
      expect(content[0].type).toBe("tool_result");
      expect(content[1].type).toBe("tool_result");
    });

    it("converts service_tier", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
        service_tier: "auto",
      } as any);

      expect(result.service_tier).toBe("auto");
    });
  });

  describe("convertStreamEvent - advanced", () => {
    function makeMessageStart(): Anthropic.RawMessageStreamEvent {
      return {
        type: "message_start",
        message: {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-4-20250514" as Anthropic.Model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: 10,
            output_tokens: 0,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            cache_creation: null,
            inference_geo: null,
            server_tool_use: null,
            service_tier: null,
          },
          container: null,
        },
      };
    }

    it("emits tool_use content block on tool_use block start", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());

      const events = c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: {
          type: "tool_use",
          id: "tu_1",
          name: "get_weather",
          input: {},
          caller: { type: "direct" },
        },
      });

      const itemAdded = events.find(e => e.type === "response.output_item.added") as any;
      expect(itemAdded).toBeDefined();
      expect(itemAdded.item.type).toBe("function_call");
      expect(itemAdded.item.name).toBe("get_weather");
    });

    it("emits web_search_tool_result count on web_search block start", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());

      const events = c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: {
          type: "web_search_tool_result",
          tool_use_id: "ws_1",
          content: [],
          caller: { type: "direct" },
        },
      });

      // web_search_tool_result just increments counter, no Responses event
      expect(events.length).toBe(0);
    });

    it("handles input_json_delta for function arguments", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: {
          type: "tool_use",
          id: "tu_1",
          name: "fn",
          input: {},
          caller: { type: "direct" },
        },
      });

      const events = c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: '{"x' },
      });

      const argsDelta = events.find(
        e => e.type === "response.function_call_arguments.delta"
      ) as any;
      expect(argsDelta).toBeDefined();
      expect(argsDelta.delta).toBe('{"x');
    });

    it("tracks web search requests in message_delta usage", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());

      c.convertStreamEvent({
        type: "message_delta",
        delta: { stop_reason: "end_turn", stop_sequence: null, container: null },
        usage: {
          output_tokens: 5,
          input_tokens: null,
          cache_creation_input_tokens: null,
          cache_read_input_tokens: null,
          server_tool_use: { web_search_requests: 2 } as any,
        },
      });

      const events = c.convertStreamEvent({ type: "message_stop" });
      const completed = events.find(e => e.type === "response.completed") as any;
      expect(completed).toBeDefined();
    });
  });
});
