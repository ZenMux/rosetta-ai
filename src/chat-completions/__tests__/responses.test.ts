import type OpenAI from "openai";
import { ChatCompletionToResponsesConverter } from "../responses";

describe("ChatCompletionToResponsesConverter", () => {
  const converter = new ChatCompletionToResponsesConverter();

  describe("convertRequest", () => {
    it("converts messages to input items", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [
          { role: "system", content: "Be helpful." },
          { role: "user", content: "Hello" },
        ],
      });

      expect(result.model).toBe("gpt-4o");
      expect((result.input as any[])[0]).toEqual({
        role: "system",
        type: "message",
        content: "Be helpful.",
      });
      expect((result.input as any[])[1]).toEqual({
        role: "user",
        type: "message",
        content: "Hello",
      });
    });

    it("converts assistant with tool_calls to function_call items", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [
          { role: "user", content: "Weather?" },
          {
            role: "assistant",
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: { name: "get_weather", arguments: '{"city":"SF"}' },
              },
            ],
          },
          { role: "tool", tool_call_id: "call_1", content: "72F" },
        ],
      });

      const input = result.input as any[];
      expect(input[1]).toEqual({
        type: "function_call",
        name: "get_weather",
        call_id: "call_1",
        arguments: '{"city":"SF"}',
      });
      expect(input[2]).toEqual({
        type: "function_call_output",
        call_id: "call_1",
        output: "72F",
      });
    });

    it("maps max_completion_tokens to max_output_tokens", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        max_completion_tokens: 1000,
      });
      expect(result.max_output_tokens).toBe(1000);
    });

    it("maps temperature and top_p", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        temperature: 0.7,
        top_p: 0.9,
      });
      expect(result.temperature).toBe(0.7);
      expect(result.top_p).toBe(0.9);
    });

    it("converts function tools", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              description: "Get weather",
              parameters: { type: "object" },
              strict: true,
            },
          },
        ],
      });

      expect((result.tools as any[])[0]).toEqual({
        type: "function",
        name: "get_weather",
        description: "Get weather",
        strict: true,
        parameters: { type: "object" },
      });
    });

    it("converts web_search_options to web_search tool", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Search" }],
        web_search_options: {},
      } as any);

      expect((result.tools as any[]).find((t: any) => t.type === "web_search")).toBeDefined();
    });

    it("maps response_format to text.format", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        response_format: {
          type: "json_schema",
          json_schema: { name: "answer", schema: { type: "object" } },
        },
      });

      expect(result.text?.format).toEqual({
        type: "json_schema",
        name: "answer",
        schema: { type: "object" },
      });
    });

    it("maps tool_choice", () => {
      expect(
        converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          tool_choice: "auto",
        }).tool_choice
      ).toBe("auto");

      expect(
        converter.convertRequest({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hi" }],
          tool_choice: "required",
        }).tool_choice
      ).toBe("required");
    });

    it("maps reasoning_effort to reasoning", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        reasoning_effort: "high",
      } as any);

      expect((result.reasoning as any).effort).toBe("high");
    });

    it("maps top_logprobs to include", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        top_logprobs: 5,
      } as any);

      expect(result.include).toContain("message.output_text.logprobs");
    });

    it("passes through parallel_tool_calls", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        parallel_tool_calls: false,
      } as any);

      expect((result as any).parallel_tool_calls).toBe(false);
    });

    it("converts user multimodal content", () => {
      const result = converter.convertRequest({
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
      });

      const input = result.input as any[];
      const content = input[0].content;
      expect(content[0]).toEqual({ type: "input_text", text: "What is this?" });
      expect(content[1]).toEqual({
        type: "input_image",
        image_url: "https://example.com/img.png",
        detail: "auto",
      });
    });
  });

  // ===== convertResponse (Responses → CC, backward) =====

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
      expect(result.choices[0].finish_reason).toBe("tool_calls");
    });

    it('maps incomplete status with max_output_tokens to "length"', () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [
            {
              type: "message",
              id: "msg_1",
              role: "assistant",
              status: "incomplete",
              content: [
                { type: "output_text", text: "Partial", annotations: [], logprobs: null as any },
              ],
            } as any,
          ],
          status: "incomplete",
          incomplete_details: { reason: "max_output_tokens" },
        })
      );
      expect(result.choices[0].finish_reason).toBe("length");
    });

    it('maps incomplete status with content_filter to "content_filter"', () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [
            {
              type: "message",
              id: "msg_1",
              role: "assistant",
              status: "incomplete",
              content: [{ type: "output_text", text: "", annotations: [], logprobs: null as any }],
            } as any,
          ],
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

  // ===== convertStreamEvent (Responses → CC, backward) =====

  describe("convertStreamEvent", () => {
    function first(
      result: OpenAI.ChatCompletionChunk | OpenAI.ChatCompletionChunk[] | null
    ): OpenAI.ChatCompletionChunk {
      if (Array.isArray(result)) return result[0];
      return result!;
    }

    function initStream(c: ChatCompletionToResponsesConverter) {
      c.convertStreamEvent({
        type: "response.created",
        response: { id: "resp_1", model: "gpt-4o", created_at: 100 } as any,
        sequence_number: 0,
      });
    }

    it("emits role chunk on response.created", () => {
      const c = new ChatCompletionToResponsesConverter();
      const result = first(
        c.convertStreamEvent({
          type: "response.created",
          response: { id: "resp_1", model: "gpt-4o", created_at: 100 } as any,
          sequence_number: 0,
        })
      );

      expect(result.choices[0].delta.role).toBe("assistant");
      expect(result.id).toBe("resp_1");
    });

    it("emits text delta with role on response.output_text.delta", () => {
      const c = new ChatCompletionToResponsesConverter();
      initStream(c);

      const result = first(
        c.convertStreamEvent({
          type: "response.output_text.delta",
          delta: "Hello",
          item_id: "msg_1",
          output_index: 0,
          content_index: 0,
          sequence_number: 1,
        } as any)
      );

      expect(result.choices[0].delta.role).toBe("assistant");
      expect(result.choices[0].delta.content).toBe("Hello");
    });

    it("emits tool_call start on response.output_item.added (function_call)", () => {
      const c = new ChatCompletionToResponsesConverter();
      initStream(c);

      const result = first(
        c.convertStreamEvent({
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
        } as any)
      );

      const tc = result.choices[0].delta.tool_calls![0];
      expect(tc.id).toBe("call_1");
      expect(tc.function!.name).toBe("get_weather");
    });

    it("emits tool_call arguments delta", () => {
      const c = new ChatCompletionToResponsesConverter();
      initStream(c);
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

      const result = first(
        c.convertStreamEvent({
          type: "response.function_call_arguments.delta",
          delta: '{"city',
          item_id: "fc_1",
          output_index: 0,
          sequence_number: 2,
        } as any)
      );

      expect(result.choices[0].delta.tool_calls![0].function!.arguments).toBe('{"city');
    });

    it("emits tool_calls finish_reason when tool calls present", () => {
      const c = new ChatCompletionToResponsesConverter();
      initStream(c);
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

      const result = first(
        c.convertStreamEvent({
          type: "response.completed",
          response: {
            id: "resp_1",
            status: "completed",
            usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
          } as any,
          sequence_number: 10,
        })
      );

      expect(result.choices[0].finish_reason).toBe("tool_calls");
    });

    it("emits finish_reason stop on response.completed without tool calls", () => {
      const c = new ChatCompletionToResponsesConverter();
      initStream(c);

      const result = first(
        c.convertStreamEvent({
          type: "response.completed",
          response: {
            id: "resp_1",
            status: "completed",
            usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
          } as any,
          sequence_number: 10,
        })
      );

      expect(result.choices[0].finish_reason).toBe("stop");
    });

    it("handles reasoning_summary_text.delta", () => {
      const c = new ChatCompletionToResponsesConverter();
      initStream(c);

      const result = first(
        c.convertStreamEvent({
          type: "response.reasoning_summary_text.delta",
          delta: "Thinking...",
          item_id: "rs_1",
          output_index: 0,
          summary_index: 0,
          sequence_number: 1,
        } as any)
      );

      expect((result.choices[0].delta as any).reasoning).toBe("Thinking...");
    });

    it("tracks web_search_call.completed count", () => {
      const c = new ChatCompletionToResponsesConverter();
      initStream(c);

      const r1 = c.convertStreamEvent({
        type: "response.web_search_call.completed",
      } as any);
      expect(r1).toBeNull();
    });

    it("collects annotations from output_text.annotation.added", () => {
      const c = new ChatCompletionToResponsesConverter();
      initStream(c);

      c.convertStreamEvent({
        type: "response.output_text.annotation.added",
        annotation: {
          type: "url_citation",
          url: "https://example.com",
          title: "Example",
          start_index: 0,
          end_index: 5,
        },
      } as any);

      const result = first(
        c.convertStreamEvent({
          type: "response.completed",
          response: { id: "resp_1", status: "completed" } as any,
          sequence_number: 10,
        })
      );

      expect((result.choices[0].delta as any).annotations).toEqual([
        {
          type: "url_citation",
          url_citation: {
            type: "url_citation",
            url: "https://example.com",
            title: "Example",
            start_index: 0,
            end_index: 5,
          },
        },
      ]);
    });

    it("returns null for unknown events", () => {
      const c = new ChatCompletionToResponsesConverter();
      const result = c.convertStreamEvent({
        type: "response.output_text.done",
      } as any);
      expect(result).toBeNull();
    });
  });
});
