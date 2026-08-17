import type OpenAI from "openai";
import type Anthropic from "@anthropic-ai/sdk";
import { MessagesToResponsesConverter } from "../responses";

describe("MessagesToResponsesConverter", () => {
  const converter = new MessagesToResponsesConverter();

  describe("convertRequest", () => {
    it("converts system + user messages", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        system: "Be helpful.",
        messages: [{ role: "user", content: "Hello" }],
      });

      expect(result.model).toBe("claude-sonnet-4-20250514");
      expect(result.instructions).toBe("Be helpful.");
      expect(result.max_output_tokens).toBe(1024);
      const input = result.input as any[];
      expect(input[0]).toEqual({ role: "user", type: "message", content: "Hello" });
    });

    it("converts system content blocks to instructions", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        system: [
          { type: "text", text: "Be helpful." },
          { type: "text", text: "Be concise." },
        ],
        messages: [{ role: "user", content: "Hi" }],
      });

      expect(result.instructions).toBe("Be helpful.\nBe concise.");
    });

    it("converts user multimodal content", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is this?" },
              {
                type: "image",
                source: { type: "url", url: "https://example.com/img.png" },
              },
            ],
          },
        ],
      });

      const input = result.input as any[];
      expect(input[0].content[0]).toEqual({ type: "input_text", text: "What is this?" });
      expect(input[0].content[1]).toEqual({
        type: "input_image",
        image_url: "https://example.com/img.png",
      });
    });

    it("converts tool_result to function_call_output", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        messages: [
          {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "72F" }],
          },
        ],
      });

      const input = result.input as any[];
      expect(input[0]).toEqual({
        type: "function_call_output",
        call_id: "toolu_1",
        output: "72F",
      });
    });

    it("converts assistant tool_use to function_call", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        messages: [
          {
            role: "assistant",
            content: [
              { type: "tool_use", id: "toolu_1", name: "get_weather", input: { city: "SF" } },
            ],
          },
        ],
      });

      const input = result.input as any[];
      expect(input[0]).toEqual({
        type: "function_call",
        call_id: "toolu_1",
        name: "get_weather",
        arguments: '{"city":"SF"}',
      });
    });

    it("converts tools", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        tools: [
          {
            name: "get_weather",
            description: "Get weather",
            input_schema: { type: "object", properties: { city: { type: "string" } } },
          },
        ],
      });

      expect((result.tools as any[])[0].type).toBe("function");
      expect((result.tools as any[])[0].name).toBe("get_weather");
    });

    it("converts tool_choice", () => {
      expect(
        converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          tool_choice: { type: "auto" },
        }).tool_choice
      ).toBe("auto");

      expect(
        converter.convertRequest({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hi" }],
          tool_choice: { type: "any" },
        }).tool_choice
      ).toBe("required");
    });

    it("converts thinking to reasoning", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        max_tokens: 16000,
        messages: [{ role: "user", content: "Think" }],
        thinking: { type: "enabled", budget_tokens: 10000 },
      });

      expect((result.reasoning as any).effort).toBe("high");
    });

    it("converts output_config to text.format", () => {
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

      expect(result.text?.format).toEqual({
        type: "json_schema",
        name: "response",
        schema: { type: "object", properties: { answer: { type: "string" } } },
      });
    });

    it("maps temperature and top_p", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        temperature: 0.7,
        top_p: 0.9,
      });
      expect(result.temperature).toBe(0.7);
      expect(result.top_p).toBe(0.9);
    });

    it("maps disable_parallel_tool_use to parallel_tool_calls", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: { type: "auto", disable_parallel_tool_use: true },
      });

      expect((result as any).parallel_tool_calls).toBe(false);
    });

    it("maps metadata.user_id to metadata", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        metadata: { user_id: "user_123" },
      });

      expect((result as any).metadata).toEqual({ user_id: "user_123" });
    });
  });

  // ===== convertResponse (Responses → Messages, backward) =====

  describe("convertResponse", () => {
    function makeResponse(
      overrides: Partial<OpenAI.Responses.Response> = {}
    ): OpenAI.Responses.Response {
      return {
        id: "resp_123",
        object: "response",
        created_at: 1700000000,
        model: "claude-sonnet-4-20250514",
        output: [
          {
            type: "message",
            id: "msg_1",
            role: "assistant",
            status: "completed",
            content: [
              {
                type: "output_text",
                text: "Hello!",
                annotations: [],
                logprobs: null as any,
              },
            ],
          } as any,
        ],
        status: "completed",
        usage: {
          input_tokens: 10,
          output_tokens: 5,
          total_tokens: 15,
          input_tokens_details: { cached_tokens: 0 },
          output_tokens_details: { reasoning_tokens: 0 },
        },
        error: null,
        incomplete_details: null,
        instructions: null,
        metadata: {},
        temperature: null,
        top_p: null,
        max_output_tokens: null,
        previous_response_id: null,
        parallel_tool_calls: true,
        tool_choice: "auto",
        tools: [],
        text: { format: { type: "text" } },
        reasoning: null,
        truncation: null as any,
        user: undefined as any,
        ...overrides,
      } as OpenAI.Responses.Response;
    }

    it("converts a basic text response", () => {
      const result = converter.convertResponse(makeResponse());

      expect(result.id).toBe("resp_123");
      expect(result.model).toBe("claude-sonnet-4-20250514");
      expect(result.type).toBe("message");
      expect(result.role).toBe("assistant");
      expect(result.stop_reason).toBe("end_turn");
      expect(result.usage.input_tokens).toBe(10);
      expect(result.usage.output_tokens).toBe(5);

      const textBlock = result.content.find((b: any) => b.type === "text") as any;
      expect(textBlock).toBeDefined();
      expect(textBlock.text).toBe("Hello!");
    });

    it("converts function_call to tool_use", () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [
            {
              type: "function_call",
              id: "fc_1",
              call_id: "call_1",
              name: "get_weather",
              arguments: '{"city":"SF"}',
              status: "completed",
            } as any,
          ],
          status: "completed",
        })
      );

      const toolBlock = result.content.find((b: any) => b.type === "tool_use") as any;
      expect(toolBlock).toBeDefined();
      expect(toolBlock.name).toBe("get_weather");
      expect(toolBlock.id).toBe("call_1");
      expect(toolBlock.input).toEqual({ city: "SF" });
      expect(result.stop_reason).toBe("tool_use");
    });

    it("converts reasoning to thinking block", () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [
            {
              type: "reasoning",
              id: "r_1",
              summary: [{ type: "summary_text", text: "Let me think..." }],
            } as any,
            {
              type: "message",
              id: "msg_1",
              role: "assistant",
              status: "completed",
              content: [{ type: "output_text", text: "42", annotations: [], logprobs: null }],
            } as any,
          ],
        })
      );

      const thinkingBlock = result.content.find((b: any) => b.type === "thinking") as any;
      expect(thinkingBlock).toBeDefined();
      expect(thinkingBlock.thinking).toBe("Let me think...");

      const textBlock = result.content.find((b: any) => b.type === "text") as any;
      expect(textBlock.text).toBe("42");
    });

    it("maps incomplete status to max_tokens", () => {
      const result = converter.convertResponse(
        makeResponse({
          status: "incomplete",
          incomplete_details: { reason: "max_output_tokens" },
        })
      );
      expect(result.stop_reason).toBe("max_tokens");
    });

    it("separates cache read and write tokens from input_tokens", () => {
      const result = converter.convertResponse(
        makeResponse({
          usage: {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            input_tokens_details: { cached_tokens: 30, cache_write_tokens: 20 },
            output_tokens_details: { reasoning_tokens: 0 },
          } as any,
        })
      );

      expect(result.usage.input_tokens).toBe(50);
      expect(result.usage.output_tokens).toBe(50);
      expect(result.usage.cache_read_input_tokens).toBe(30);
      expect(result.usage.cache_creation_input_tokens).toBe(20);
      expect((result.usage as any).prompt_tokens).toBe(100);
      expect((result.usage as any).total_tokens).toBe(150);
    });

    it("counts only completed web search calls", () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [
            {
              type: "web_search_call",
              id: "ws_1",
              status: "completed",
              action: { type: "search", queries: ["q1"] },
            } as any,
            {
              type: "web_search_call",
              id: "ws_2",
              status: "in_progress",
              action: { type: "search", queries: ["q2"] },
            } as any,
          ],
        })
      );

      expect((result.usage as any).prompt_tokens_details?.web_search).toBe(1);
      expect((result.usage as any).server_tool_use?.web_search_requests).toBe(1);
    });

    it("converts empty output to empty text content", () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [],
        })
      );

      expect(result.content.length).toBe(1);
      expect(result.content[0].type).toBe("text");
      expect((result.content[0] as any).text).toBe("");
    });
  });

  // ===== convertStreamEvent (Responses → Messages, backward) =====

  describe("convertStreamEvent", () => {
    function responseCreated(): OpenAI.Responses.ResponseStreamEvent {
      return {
        type: "response.created",
        response: {
          id: "resp_123",
          model: "claude-sonnet-4-20250514",
          created_at: 1700000000,
          output: [],
          status: "in_progress",
        } as any,
        sequence_number: 0,
      };
    }

    it("emits message_start on response.created", () => {
      const c = new MessagesToResponsesConverter();
      const events = c.convertStreamEvent(responseCreated());

      expect(events.length).toBe(1);
      expect(events[0].type).toBe("message_start");
      const msg = (events[0] as Anthropic.RawMessageStartEvent).message;
      expect(msg.id).toBe("resp_123");
      expect(msg.model).toBe("claude-sonnet-4-20250514");
    });

    it("emits content_block_start for text message output item", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(responseCreated());

      const events = c.convertStreamEvent({
        type: "response.output_item.added",
        item: {
          type: "message",
          id: "msg_1",
          role: "assistant",
          status: "in_progress",
          content: [],
        },
        output_index: 0,
        sequence_number: 1,
      } as any);

      const startEvent = events.find(e => e.type === "content_block_start") as any;
      expect(startEvent).toBeDefined();
      expect(startEvent.content_block.type).toBe("text");
    });

    it("emits text_delta on response.output_text.delta", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(responseCreated());
      c.convertStreamEvent({
        type: "response.output_item.added",
        item: {
          type: "message",
          id: "msg_1",
          role: "assistant",
          status: "in_progress",
          content: [],
        },
        output_index: 0,
        sequence_number: 1,
      } as any);

      const events = c.convertStreamEvent({
        type: "response.output_text.delta",
        delta: "Hello",
        item_id: "msg_1",
        output_index: 0,
        content_index: 0,
        sequence_number: 2,
      } as any);

      const textDelta = events.find(e => e.type === "content_block_delta") as any;
      expect(textDelta).toBeDefined();
      expect(textDelta.delta.type).toBe("text_delta");
      expect(textDelta.delta.text).toBe("Hello");
    });

    it("emits content_block_start for function_call (tool_use)", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(responseCreated());

      const events = c.convertStreamEvent({
        type: "response.output_item.added",
        item: {
          type: "function_call",
          id: "fc_1",
          call_id: "toolu_1",
          name: "get_weather",
          arguments: "",
          status: "in_progress",
        },
        output_index: 0,
        sequence_number: 1,
      } as any);

      const startEvent = events.find(e => e.type === "content_block_start") as any;
      expect(startEvent).toBeDefined();
      expect(startEvent.content_block.type).toBe("tool_use");
      expect(startEvent.content_block.name).toBe("get_weather");
      expect(startEvent.content_block.id).toBe("toolu_1");
    });

    it("emits input_json_delta on response.function_call_arguments.delta", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(responseCreated());
      c.convertStreamEvent({
        type: "response.output_item.added",
        item: {
          type: "function_call",
          id: "fc_1",
          call_id: "toolu_1",
          name: "fn",
          arguments: "",
          status: "in_progress",
        },
        output_index: 0,
        sequence_number: 1,
      } as any);

      const events = c.convertStreamEvent({
        type: "response.function_call_arguments.delta",
        delta: '{"city',
        item_id: "fc_1",
        output_index: 0,
        sequence_number: 2,
      } as any);

      const argsDelta = events.find(e => e.type === "content_block_delta") as any;
      expect(argsDelta).toBeDefined();
      expect(argsDelta.delta.type).toBe("input_json_delta");
      expect(argsDelta.delta.partial_json).toBe('{"city');
    });

    it("emits thinking events for reasoning", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(responseCreated());
      c.convertStreamEvent({
        type: "response.output_item.added",
        item: { type: "reasoning", id: "rs_1", summary: [] },
        output_index: 0,
        sequence_number: 1,
      } as any);

      const events = c.convertStreamEvent({
        type: "response.reasoning_summary_text.delta",
        delta: "Hmm...",
        item_id: "rs_1",
        output_index: 0,
        summary_index: 0,
        sequence_number: 2,
      } as any);

      const thinkingDelta = events.find(e => e.type === "content_block_delta") as any;
      expect(thinkingDelta).toBeDefined();
      expect(thinkingDelta.delta.type).toBe("thinking_delta");
      expect(thinkingDelta.delta.thinking).toBe("Hmm...");
    });

    it("emits content_block_start for reasoning output item", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(responseCreated());

      const events = c.convertStreamEvent({
        type: "response.output_item.added",
        item: { type: "reasoning", id: "rs_1", summary: [] },
        output_index: 0,
        sequence_number: 1,
      } as any);

      const startEvent = events.find(e => e.type === "content_block_start") as any;
      expect(startEvent).toBeDefined();
      expect(startEvent.content_block.type).toBe("thinking");
    });

    it("emits message_delta + message_stop on response.completed", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(responseCreated());

      const events = c.convertStreamEvent({
        type: "response.completed",
        response: {
          id: "resp_123",
          status: "completed",
          output: [],
          usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
        } as any,
        sequence_number: 10,
      });

      const types = events.map(e => e.type);
      expect(types).toContain("message_delta");
      expect(types).toContain("message_stop");

      const msgDelta = events.find(
        e => e.type === "message_delta"
      ) as Anthropic.RawMessageDeltaEvent;
      expect(msgDelta.delta.stop_reason).toBe("end_turn");
    });

    it("emits max_tokens stop_reason on response.incomplete", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(responseCreated());

      const events = c.convertStreamEvent({
        type: "response.incomplete",
        response: {
          id: "resp_123",
          status: "incomplete",
          output: [],
          usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
        } as any,
        sequence_number: 10,
      });

      const msgDelta = events.find(
        e => e.type === "message_delta"
      ) as Anthropic.RawMessageDeltaEvent;
      expect(msgDelta.delta.stop_reason).toBe("max_tokens");
      expect(events.find(e => e.type === "message_stop")).toBeDefined();
    });

    it("emits end_turn on response.failed", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(responseCreated());

      const events = c.convertStreamEvent({
        type: "response.failed",
        response: { id: "resp_123", status: "failed", output: [] } as any,
        sequence_number: 10,
      });

      const msgDelta = events.find(
        e => e.type === "message_delta"
      ) as Anthropic.RawMessageDeltaEvent;
      expect(msgDelta.delta.stop_reason).toBe("end_turn");
    });

    it("includes usage in completed event", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(responseCreated());

      const events = c.convertStreamEvent({
        type: "response.completed",
        response: {
          id: "resp_123",
          status: "completed",
          output: [],
          usage: {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            input_tokens_details: { cached_tokens: 3, cache_write_tokens: 2 },
            output_tokens_details: { reasoning_tokens: 0 },
          },
        } as any,
        sequence_number: 10,
      });

      const msgDelta = events.find(
        e => e.type === "message_delta"
      ) as Anthropic.RawMessageDeltaEvent;
      expect(msgDelta.usage.input_tokens).toBe(5);
      expect(msgDelta.usage.output_tokens).toBe(5);
      expect(msgDelta.usage.cache_read_input_tokens).toBe(3);
      expect(msgDelta.usage.cache_creation_input_tokens).toBe(2);
    });

    it("emits tool_use stop_reason when function_call is in output", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(responseCreated());

      const events = c.convertStreamEvent({
        type: "response.completed",
        response: {
          id: "resp_123",
          status: "completed",
          output: [
            {
              type: "function_call",
              id: "fc_1",
              call_id: "toolu_1",
              name: "fn",
              arguments: "{}",
              status: "completed",
            },
          ],
          usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
        } as any,
        sequence_number: 10,
      });

      const msgDelta = events.find(
        e => e.type === "message_delta"
      ) as Anthropic.RawMessageDeltaEvent;
      expect(msgDelta.delta.stop_reason).toBe("tool_use");
    });

    it("counts web search when output_item.done completes a search", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(responseCreated());

      c.convertStreamEvent({
        type: "response.output_item.added",
        item: {
          type: "web_search_call",
          id: "ws_1",
          status: "completed",
          action: { type: "search", queries: ["q1"] },
        },
        output_index: 0,
        sequence_number: 1,
      } as any);

      c.convertStreamEvent({
        type: "response.output_item.done",
        item: {
          type: "web_search_call",
          id: "ws_1",
          status: "completed",
          action: { type: "search", queries: ["q1"] },
        },
        output_index: 0,
        sequence_number: 2,
      } as any);

      const events = c.convertStreamEvent({
        type: "response.completed",
        response: {
          id: "resp_123",
          status: "completed",
          output: [],
          usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
        } as any,
        sequence_number: 3,
      });

      const msgDelta = events.find(
        e => e.type === "message_delta"
      ) as Anthropic.RawMessageDeltaEvent;
      expect((msgDelta.usage as any).prompt_tokens_details?.web_search).toBe(1);
    });
  });
});
