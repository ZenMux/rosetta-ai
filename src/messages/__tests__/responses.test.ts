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
            content: [
              { type: "tool_result", tool_use_id: "toolu_1", content: "72F" },
            ],
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
  });

  describe("convertResponse", () => {
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
      expect(msgOutput.content[0].text).toBe("Hello!");
    });

    it("converts tool_use to function_call output items", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            {
              type: "tool_use",
              id: "toolu_1",
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
      expect(fcOutput.call_id).toBe("toolu_1");
      expect(fcOutput.arguments).toBe('{"city":"SF"}');
      expect(result.status).toBe("completed");
    });

    it("converts thinking to reasoning output item", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            { type: "thinking", thinking: "Let me think...", signature: "sig" } as any,
            { type: "text", text: "42", citations: null },
          ],
        })
      );

      const reasoning = result.output.find((o: any) => o.type === "reasoning") as any;
      expect(reasoning).toBeDefined();
      expect(reasoning.summary[0].text).toBe("Let me think...");
    });

    it("maps max_tokens to incomplete status", () => {
      const result = converter.convertResponse(
        makeMessage({ stop_reason: "max_tokens" })
      );
      expect(result.status).toBe("incomplete");
    });
  });

  describe("convertStreamEvent", () => {
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

    it("emits response.created + response.in_progress on message_start", () => {
      const c = new MessagesToResponsesConverter();
      const events = c.convertStreamEvent(messageStart());
      const types = events.map(e => e.type);

      expect(types).toContain("response.created");
      expect(types).toContain("response.in_progress");
    });

    it("emits output_item.added + content_part.added on text content_block_start", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(messageStart());

      const events = c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "", citations: null },
      });

      expect(events.find(e => e.type === "response.output_item.added")).toBeDefined();
      expect(events.find(e => e.type === "response.content_part.added")).toBeDefined();
    });

    it("emits output_text.delta on text_delta", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(messageStart());
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

    it("emits output_item.added for tool_use", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(messageStart());

      const events = c.convertStreamEvent({
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

      const itemAdded = events.find(e => e.type === "response.output_item.added") as any;
      expect(itemAdded).toBeDefined();
      expect(itemAdded.item.type).toBe("function_call");
      expect(itemAdded.item.name).toBe("get_weather");
    });

    it("emits function_call_arguments.delta on input_json_delta", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(messageStart());
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: {
          type: "tool_use",
          id: "toolu_1",
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

    it("emits reasoning_summary_text.delta on thinking_delta", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(messageStart());
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "thinking", thinking: "", signature: "" } as any,
      });

      const events = c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "Hmm..." } as any,
      });

      const reasoningDelta = events.find(
        e => e.type === "response.reasoning_summary_text.delta"
      ) as any;
      expect(reasoningDelta).toBeDefined();
      expect(reasoningDelta.delta).toBe("Hmm...");
    });

    it("emits response.completed on message_stop", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(messageStart());
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
    });

    it("emits response.incomplete on max_tokens stop", () => {
      const c = new MessagesToResponsesConverter();
      c.convertStreamEvent(messageStart());
      c.convertStreamEvent({
        type: "message_delta",
        delta: { stop_reason: "max_tokens", stop_sequence: null, container: null },
        usage: {
          output_tokens: 5,
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
});
