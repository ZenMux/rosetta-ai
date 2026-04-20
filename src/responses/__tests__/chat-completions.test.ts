import type OpenAI from "openai";
import { ResponsesToChatCompletionConverter } from "../chat-completions";

describe("ResponsesToChatCompletionConverter", () => {
  const converter = new ResponsesToChatCompletionConverter();

  describe("convertRequest", () => {
    it("converts string input to user message", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hello",
      });

      expect(result.model).toBe("gpt-4o");
      expect(result.messages).toEqual([{ role: "user", content: "Hello" }]);
    });

    it("converts instructions to system message", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hi",
        instructions: "Be helpful.",
      });

      expect(result.messages[0]).toEqual({ role: "system", content: "Be helpful." });
      expect(result.messages[1]).toEqual({ role: "user", content: "Hi" });
    });

    it("converts input array with easy messages", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: [
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

    it("converts function_call and function_call_output in input", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
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
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "get_weather", arguments: '{"city":"SF"}' },
          },
        ],
      });
      expect(result.messages[2]).toEqual({
        role: "tool",
        tool_call_id: "call_1",
        content: "72°F",
      });
    });

    it("maps max_output_tokens to max_completion_tokens", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hi",
        max_output_tokens: 1000,
      });
      expect(result.max_completion_tokens).toBe(1000);
    });

    it("maps temperature and top_p", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hi",
        temperature: 0.7,
        top_p: 0.9,
      });
      expect(result.temperature).toBe(0.7);
      expect(result.top_p).toBe(0.9);
    });

    it("converts function tools", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
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

      expect(result.tools).toEqual([
        {
          type: "function",
          function: {
            name: "get_weather",
            description: "Get weather",
            parameters: { type: "object", properties: { city: { type: "string" } } },
            strict: true,
          },
        },
      ]);
    });

    it("maps tool_choice", () => {
      expect(
        converter.convertRequest({ model: "gpt-4o", input: "Hi", tool_choice: "auto" }).tool_choice
      ).toBe("auto");
      expect(
        converter.convertRequest({ model: "gpt-4o", input: "Hi", tool_choice: "required" })
          .tool_choice
      ).toBe("required");
      expect(
        converter.convertRequest({ model: "gpt-4o", input: "Hi", tool_choice: "none" }).tool_choice
      ).toBe("none");
    });

    it("converts json_schema text format to response_format", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hi",
        text: {
          format: {
            type: "json_schema",
            name: "answer",
            schema: { type: "object", properties: { answer: { type: "string" } } },
          },
        },
      });

      expect(result.response_format).toEqual({
        type: "json_schema",
        json_schema: {
          name: "answer",
          schema: { type: "object", properties: { answer: { type: "string" } } },
          strict: undefined,
        },
      });
    });
  });

  describe("convertResponse", () => {
    function makeResponse(
      overrides: Partial<OpenAI.Responses.Response> = {}
    ): OpenAI.Responses.Response {
      return {
        id: "resp_123",
        object: "response",
        created_at: 1700000000,
        model: "gpt-4o",
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
      expect(result.model).toBe("gpt-4o");
      expect(result.object).toBe("chat.completion");
      expect(result.created).toBe(1700000000);
      expect(result.choices[0].message.content).toBe("Hello!");
      expect(result.choices[0].finish_reason).toBe("stop");
      expect(result.usage?.prompt_tokens).toBe(10);
      expect(result.usage?.completion_tokens).toBe(5);
    });

    it("converts function_call output items to tool_calls", () => {
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

      expect(result.choices[0].message.tool_calls).toEqual([
        {
          id: "call_1",
          type: "function",
          function: { name: "get_weather", arguments: '{"city":"SF"}' },
        },
      ]);
      expect(result.choices[0].finish_reason).toBe("stop");
    });

    it('maps incomplete status with max_output_tokens to "length"', () => {
      const result = converter.convertResponse(
        makeResponse({
          status: "incomplete",
          incomplete_details: { reason: "max_output_tokens" },
        })
      );
      expect(result.choices[0].finish_reason).toBe("length");
    });

    it('maps incomplete status with content_filter to "content_filter"', () => {
      const result = converter.convertResponse(
        makeResponse({
          status: "incomplete",
          incomplete_details: { reason: "content_filter" },
        })
      );
      expect(result.choices[0].finish_reason).toBe("content_filter");
    });

    it("extracts url_citation annotations", () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [
            {
              type: "message",
              id: "msg_1",
              role: "assistant",
              status: "completed",
              content: [
                {
                  type: "output_text",
                  text: "Result",
                  annotations: [
                    {
                      type: "url_citation",
                      title: "Example",
                      url: "https://example.com",
                      start_index: 0,
                      end_index: 6,
                    },
                  ],
                  logprobs: null as any,
                },
              ],
            } as any,
          ],
        })
      );

      expect(result.choices[0].message.annotations).toEqual([
        {
          type: "url_citation",
          url_citation: {
            title: "Example",
            url: "https://example.com",
            start_index: 0,
            end_index: 6,
          },
        },
      ]);
    });

    it("converts reasoning items", () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [
            {
              type: "reasoning",
              id: "r_1",
              summary: [{ type: "summary_text", text: "Thinking..." }],
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

      expect(result.choices[0].message.content).toBe("42");
      expect((result.choices[0].message as any).reasoning).toBe("Thinking...");
    });
  });

  describe("convertStreamEvent", () => {
    it("emits role chunk on response.created", () => {
      const c = new ResponsesToChatCompletionConverter();
      const result = c.convertStreamEvent({
        type: "response.created",
        response: { id: "resp_1", model: "gpt-4o", created_at: 100 } as any,
        sequence_number: 0,
      });

      expect(result).not.toBeNull();
      expect(result!.choices[0].delta.role).toBe("assistant");
      expect(result!.id).toBe("resp_1");
    });

    it("emits text delta on response.output_text.delta", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamEvent({
        type: "response.created",
        response: { id: "resp_1", model: "gpt-4o", created_at: 100 } as any,
        sequence_number: 0,
      });

      const result = c.convertStreamEvent({
        type: "response.output_text.delta",
        delta: "Hello",
        item_id: "msg_1",
        output_index: 0,
        content_index: 0,
        sequence_number: 1,
      } as any);

      expect(result!.choices[0].delta.content).toBe("Hello");
    });

    it("emits tool_call start on response.output_item.added (function_call)", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamEvent({
        type: "response.created",
        response: { id: "resp_1", model: "gpt-4o", created_at: 100 } as any,
        sequence_number: 0,
      });

      const result = c.convertStreamEvent({
        type: "response.output_item.added",
        item: {
          type: "function_call",
          id: "fc_1",
          call_id: "call_1",
          name: "get_weather",
          arguments: "",
          status: "in_progress",
        },
        output_index: 0,
        sequence_number: 1,
      } as any);

      expect(result).not.toBeNull();
      const tc = result!.choices[0].delta.tool_calls![0];
      expect(tc.id).toBe("call_1");
      expect(tc.function!.name).toBe("get_weather");
    });

    it("emits tool_call arguments delta", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamEvent({
        type: "response.created",
        response: { id: "resp_1", model: "gpt-4o", created_at: 100 } as any,
        sequence_number: 0,
      });
      c.convertStreamEvent({
        type: "response.output_item.added",
        item: {
          type: "function_call",
          id: "fc_1",
          call_id: "call_1",
          name: "fn",
          arguments: "",
          status: "in_progress",
        },
        output_index: 0,
        sequence_number: 1,
      } as any);

      const result = c.convertStreamEvent({
        type: "response.function_call_arguments.delta",
        delta: '{"city',
        item_id: "fc_1",
        output_index: 0,
        sequence_number: 2,
      } as any);

      expect(result!.choices[0].delta.tool_calls![0].function!.arguments).toBe('{"city');
    });

    it("emits finish_reason on response.completed", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamEvent({
        type: "response.created",
        response: { id: "resp_1", model: "gpt-4o", created_at: 100 } as any,
        sequence_number: 0,
      });

      const result = c.convertStreamEvent({
        type: "response.completed",
        response: {
          id: "resp_1",
          status: "completed",
          usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
        } as any,
        sequence_number: 10,
      });

      expect(result!.choices[0].finish_reason).toBe("stop");
      expect(result!.usage?.prompt_tokens).toBe(10);
    });

    it("returns null for unknown events", () => {
      const c = new ResponsesToChatCompletionConverter();
      const result = c.convertStreamEvent({
        type: "response.output_text.done",
      } as any);
      expect(result).toBeNull();
    });
  });
});
