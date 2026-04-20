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

  describe("convertResponse", () => {
    function makeCCResponse(
      overrides: Partial<OpenAI.ChatCompletion> = {}
    ): OpenAI.ChatCompletion {
      return {
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
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
        },
        ...overrides,
      };
    }

    it("converts a basic text response", () => {
      const result = converter.convertResponse(makeCCResponse());

      expect(result.id).toBe("chatcmpl-123");
      expect(result.model).toBe("gpt-4o");
      expect(result.object).toBe("response");
      expect(result.created_at).toBe(1700000000);
      expect(result.status).toBe("completed");
      expect(result.usage?.input_tokens).toBe(10);
      expect(result.usage?.output_tokens).toBe(5);

      const msgOutput = result.output.find((o: any) => o.type === "message") as any;
      expect(msgOutput).toBeDefined();
      expect(msgOutput.content[0].type).toBe("output_text");
      expect(msgOutput.content[0].text).toBe("Hello!");
    });

    it("converts tool_calls to function_call output items", () => {
      const result = converter.convertResponse(
        makeCCResponse({
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: null,
                refusal: null,
                tool_calls: [
                  {
                    id: "call_1",
                    type: "function",
                    function: { name: "get_weather", arguments: '{"city":"SF"}' },
                  },
                ],
              },
              finish_reason: "tool_calls",
              logprobs: null,
            },
          ],
        })
      );

      const fcOutput = result.output.find((o: any) => o.type === "function_call") as any;
      expect(fcOutput).toBeDefined();
      expect(fcOutput.name).toBe("get_weather");
      expect(fcOutput.arguments).toBe('{"city":"SF"}');
      expect(fcOutput.call_id).toBe("call_1");

      // Tool call choice has finish_reason "tool_calls"
      const choices = (result as any).__choices_debug;
      // Verify status reflects tool use completion
      expect(result.status).toBe("completed");
    });

    it("maps finish_reason length to incomplete status", () => {
      const result = converter.convertResponse(
        makeCCResponse({
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Partial", refusal: null },
              finish_reason: "length",
              logprobs: null,
            },
          ],
        })
      );

      expect(result.status).toBe("incomplete");
    });

    it("converts reasoning to reasoning output item", () => {
      const result = converter.convertResponse(
        makeCCResponse({
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: "42",
                refusal: null,
                reasoning: "Let me think...",
              } as any,
              finish_reason: "stop",
              logprobs: null,
            },
          ],
        })
      );

      const reasoningOutput = result.output.find((o: any) => o.type === "reasoning") as any;
      expect(reasoningOutput).toBeDefined();
      expect(reasoningOutput.summary[0].text).toBe("Let me think...");
    });

    it("converts annotations to output_text annotations", () => {
      const result = converter.convertResponse(
        makeCCResponse({
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: "Result",
                refusal: null,
                annotations: [
                  {
                    type: "url_citation",
                    url_citation: {
                      title: "Example",
                      url: "https://example.com",
                      start_index: 0,
                      end_index: 6,
                    },
                  },
                ],
              },
              finish_reason: "stop",
              logprobs: null,
            },
          ],
        })
      );

      const msgOutput = result.output.find((o: any) => o.type === "message") as any;
      expect(msgOutput.content[0].annotations[0].url).toBe("https://example.com");
    });

    it("converts refusal", () => {
      const result = converter.convertResponse(
        makeCCResponse({
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: null, refusal: "I cannot do that" },
              finish_reason: "stop",
              logprobs: null,
            },
          ],
        })
      );

      const msgOutput = result.output.find((o: any) => o.type === "message") as any;
      expect(msgOutput.content[0].type).toBe("refusal");
      expect(msgOutput.content[0].refusal).toBe("I cannot do that");
    });
  });

  describe("convertStreamChunk", () => {
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

    it("emits response.created and response.in_progress on first chunk", () => {
      const c = new ChatCompletionToResponsesConverter();
      const events = c.convertStreamChunk(
        makeChunk({ delta: { role: "assistant" } })
      );

      const types = events.map(e => e.type);
      expect(types).toContain("response.created");
      expect(types).toContain("response.in_progress");
    });

    it("emits output_text.delta for text content", () => {
      const c = new ChatCompletionToResponsesConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

      const events = c.convertStreamChunk(
        makeChunk({ delta: { content: "Hello" } })
      );

      const textDelta = events.find(e => e.type === "response.output_text.delta") as any;
      expect(textDelta).toBeDefined();
      expect(textDelta.delta).toBe("Hello");
    });

    it("emits output_item.added for function_call", () => {
      const c = new ChatCompletionToResponsesConverter();
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

      const itemAdded = events.find(e => e.type === "response.output_item.added") as any;
      expect(itemAdded).toBeDefined();
      expect(itemAdded.item.type).toBe("function_call");
      expect(itemAdded.item.name).toBe("get_weather");
    });

    it("emits function_call_arguments.delta for tool arguments", () => {
      const c = new ChatCompletionToResponsesConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(
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

      const events = c.convertStreamChunk(
        makeChunk({
          delta: { tool_calls: [{ index: 0, function: { arguments: '{"x' } }] },
        })
      );

      const argsDelta = events.find(
        e => e.type === "response.function_call_arguments.delta"
      ) as any;
      expect(argsDelta).toBeDefined();
      expect(argsDelta.delta).toBe('{"x');
    });

    it("emits response.completed on finish", () => {
      const c = new ChatCompletionToResponsesConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(makeChunk({ delta: { content: "Hi" } }));

      const events = c.convertStreamChunk(
        makeChunk({
          finish_reason: "stop",
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
        } as any)
      );

      const completed = events.find(e => e.type === "response.completed") as any;
      expect(completed).toBeDefined();
      expect(completed.response.status).toBe("completed");
    });

    it("emits response.incomplete on length finish", () => {
      const c = new ChatCompletionToResponsesConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

      const events = c.convertStreamChunk(makeChunk({ finish_reason: "length" }));

      const incomplete = events.find(e => e.type === "response.incomplete") as any;
      expect(incomplete).toBeDefined();
    });
  });
});
