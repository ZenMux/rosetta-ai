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

    it("converts easy messages with array content (input parts)", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: [
          { role: "developer", content: "You are helpful." },
          {
            role: "user",
            content: [{ type: "input_text", text: "hello" }],
          },
        ],
      });

      expect(result.messages).toEqual([
        { role: "developer", content: "You are helpful." },
        { role: "user", content: "hello" },
      ]);
    });

    it("converts easy messages with multiple input parts", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: [
          {
            role: "user",
            content: [
              { type: "input_text", text: "What is this?" },
              { type: "input_text", text: "Describe it." },
            ],
          },
        ],
      });

      expect(result.messages).toEqual([
        {
          role: "user",
          content: [
            { type: "text", text: "What is this?" },
            { type: "text", text: "Describe it." },
          ],
        },
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

    it("passes parallel_tool_calls through", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hi",
        parallel_tool_calls: false,
      } as any);
      expect((result as any).parallel_tool_calls).toBe(false);
    });

    it("passes prompt_cache_key and prompt_cache_retention through", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hi",
        prompt_cache_key: "key_123",
        prompt_cache_retention: "24h",
      } as any);
      expect((result as any).prompt_cache_key).toBe("key_123");
      expect((result as any).prompt_cache_retention).toBe("24h");
    });

    it("maps include logprobs to top_logprobs", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hi",
        include: ["message.output_text.logprobs"],
      } as any);
      expect((result as any).top_logprobs).toBe(20);
    });

    it("maps text.verbosity to verbosity", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hi",
        text: { format: { type: "text" }, verbosity: "low" },
      } as any);
      expect((result as any).verbosity).toBe("low");
    });

    it("converts web_search tools to web_search_options", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hi",
        tools: [{ type: "web_search_preview", search_context_size: "medium" }],
      } as any);
      expect(result.tools).toBeUndefined();
      expect((result as any).web_search_options).toBeDefined();
      expect((result as any).web_search_options.search_context_size).toBe("medium");
    });

    it("merges consecutive function_call items into one assistant message", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
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
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "fn1", arguments: "{}" } },
          { id: "call_2", type: "function", function: { name: "fn2", arguments: "{}" } },
        ],
      });
      expect(result.messages[2]).toEqual({ role: "tool", tool_call_id: "call_1", content: "r1" });
      expect(result.messages[3]).toEqual({ role: "tool", tool_call_id: "call_2", content: "r2" });
    });

    it("converts reasoning input items to assistant with reasoning_content", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: [
          { role: "user", content: "Think hard" },
          {
            type: "reasoning",
            id: "r_1",
            summary: [{ type: "summary_text", text: "Deep thought..." }],
          },
        ],
      } as any);

      const msg = result.messages[1] as any;
      expect(msg.role).toBe("assistant");
      expect(msg.reasoning_content).toBe("Deep thought...");
    });
  });

  // ===== convertResponse (CC → Responses, backward) =====

  describe("convertResponse", () => {
    function makeCCResponse(overrides: Partial<OpenAI.ChatCompletion> = {}): OpenAI.ChatCompletion {
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

  // ===== convertStreamChunk (CC → Responses, backward) =====

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
      const c = new ResponsesToChatCompletionConverter();
      const events = c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

      const types = events.map(e => e.type);
      expect(types).toContain("response.created");
      expect(types).toContain("response.in_progress");
    });

    it("emits output_text.delta for text content", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

      const events = c.convertStreamChunk(makeChunk({ delta: { content: "Hello" } }));

      const textDelta = events.find(e => e.type === "response.output_text.delta") as any;
      expect(textDelta).toBeDefined();
      expect(textDelta.delta).toBe("Hello");
    });

    it("emits output_item.added for function_call", () => {
      const c = new ResponsesToChatCompletionConverter();
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
      const c = new ResponsesToChatCompletionConverter();
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
      const c = new ResponsesToChatCompletionConverter();
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
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

      const events = c.convertStreamChunk(makeChunk({ finish_reason: "length" }));

      const incomplete = events.find(e => e.type === "response.incomplete") as any;
      expect(incomplete).toBeDefined();
    });

    it("emits reasoning events for reasoning_content delta", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

      const events = c.convertStreamChunk(
        makeChunk({ delta: { reasoning_content: "Thinking..." } as any })
      );

      const reasoningAdded = events.find(e => e.type === "response.output_item.added") as any;
      expect(reasoningAdded).toBeDefined();
      expect(reasoningAdded.item.type).toBe("reasoning");

      const reasoningDelta = events.find(
        e => e.type === "response.reasoning_summary_text.delta"
      ) as any;
      expect(reasoningDelta).toBeDefined();
      expect(reasoningDelta.delta).toBe("Thinking...");
    });

    it("emits tool call events", () => {
      const c = new ResponsesToChatCompletionConverter();
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
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(
        makeChunk({
          delta: {
            tool_calls: [
              { index: 0, id: "call_1", type: "function", function: { name: "fn", arguments: "" } },
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

    it("emits text content with message item", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

      const events = c.convertStreamChunk(makeChunk({ delta: { content: "Hello" } }));

      const itemAdded = events.find(e => e.type === "response.output_item.added") as any;
      expect(itemAdded).toBeDefined();
      expect(itemAdded.item.type).toBe("message");

      const textDelta = events.find(e => e.type === "response.output_text.delta") as any;
      expect(textDelta).toBeDefined();
      expect(textDelta.delta).toBe("Hello");
    });

    it("handles usage-only chunk without choices", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(makeChunk({ delta: { content: "Hi" } }));

      const usageChunk: OpenAI.ChatCompletionChunk = {
        id: "chatcmpl-123",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: "gpt-4o",
        choices: [],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      };

      const events = c.convertStreamChunk(usageChunk);
      const completed = events.find(e => e.type === "response.completed") as any;
      expect(completed).toBeDefined();
      expect(completed.response.usage.input_tokens).toBe(10);
    });

    it("emits finish with usage when finish_reason and usage in same chunk", () => {
      const c = new ResponsesToChatCompletionConverter();
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
    });

    it("emits finish without usage", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

      const events = c.convertStreamChunk(makeChunk({ finish_reason: "stop" }));

      const completed = events.find(e => e.type === "response.completed") as any;
      expect(completed).toBeDefined();
    });

    // ===== done / refusal / annotation / custom / failed events =====

    it("emits output_text.done, content_part.done and output_item.done on finish", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(makeChunk({ delta: { content: "Hi" } }));

      const events = c.convertStreamChunk(makeChunk({ finish_reason: "stop" }));

      const textDone = events.find(e => e.type === "response.output_text.done") as any;
      expect(textDone).toBeDefined();
      expect(textDone.text).toBe("Hi");

      const partDone = events.find(e => e.type === "response.content_part.done") as any;
      expect(partDone).toBeDefined();
      expect(partDone.part.type).toBe("output_text");
      expect(partDone.part.text).toBe("Hi");

      const itemDone = events.find(e => e.type === "response.output_item.done") as any;
      expect(itemDone).toBeDefined();
      expect(itemDone.item.type).toBe("message");
      expect(itemDone.item.content[0].text).toBe("Hi");
    });

    it("emits reasoning_summary_text.done, reasoning_summary_part.done and output_item.done when reasoning closes", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(makeChunk({ delta: { reasoning_content: "Thinking..." } as any }));
      // A text delta after reasoning closes the reasoning item.
      const events = c.convertStreamChunk(makeChunk({ delta: { content: "Hi" } }));

      const textDone = events.find(e => e.type === "response.reasoning_summary_text.done") as any;
      expect(textDone).toBeDefined();
      expect(textDone.text).toBe("Thinking...");

      const partDone = events.find(e => e.type === "response.reasoning_summary_part.done") as any;
      expect(partDone).toBeDefined();
      expect(partDone.part.text).toBe("Thinking...");

      const itemDone = events.find(e => e.type === "response.output_item.done") as any;
      expect(itemDone).toBeDefined();
      expect(itemDone.item.type).toBe("reasoning");
      expect(itemDone.item.summary[0].text).toBe("Thinking...");
    });

    it("emits function_call_arguments.done and output_item.done when a tool call finishes", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(
        makeChunk({
          delta: {
            tool_calls: [
              { index: 0, id: "call_1", type: "function", function: { name: "fn", arguments: "" } },
            ],
          },
        })
      );
      c.convertStreamChunk(
        makeChunk({ delta: { tool_calls: [{ index: 0, function: { arguments: '{"x":1}' } }] } })
      );

      const events = c.convertStreamChunk(makeChunk({ finish_reason: "tool_calls" }));

      const argsDone = events.find(e => e.type === "response.function_call_arguments.done") as any;
      expect(argsDone).toBeDefined();
      expect(argsDone.name).toBe("fn");
      expect(argsDone.arguments).toBe('{"x":1}');

      const itemDone = events.find(e => e.type === "response.output_item.done") as any;
      expect(itemDone).toBeDefined();
      expect(itemDone.item.type).toBe("function_call");
      expect(itemDone.item.arguments).toBe('{"x":1}');
    });

    it("emits refusal.delta and refusal.done", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(makeChunk({ delta: { refusal: "I can't" } as any }));

      const events = c.convertStreamChunk(makeChunk({ delta: { refusal: " help." } as any }));

      const refusalDelta = events.find(e => e.type === "response.refusal.delta") as any;
      expect(refusalDelta).toBeDefined();
      expect(refusalDelta.delta).toBe(" help.");

      // Closing via finish_reason emits refusal.done + content_part.done + output_item.done
      const finish = c.convertStreamChunk(makeChunk({ finish_reason: "stop" }));
      const refusalDone = finish.find(e => e.type === "response.refusal.done") as any;
      expect(refusalDone).toBeDefined();
      expect(refusalDone.refusal).toBe("I can't help.");
    });

    it("emits response.output_text.annotation.added for url_citation annotations", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(makeChunk({ delta: { content: "Hello" } }));

      const events = c.convertStreamChunk(
        makeChunk({
          delta: {
            annotations: [
              {
                type: "url_citation",
                url_citation: {
                  url: "https://example.com",
                  title: "Ex",
                  start_index: 0,
                  end_index: 5,
                },
              },
            ],
          } as any,
        })
      );

      const annAdded = events.find(e => e.type === "response.output_text.annotation.added") as any;
      expect(annAdded).toBeDefined();
      expect(annAdded.annotation.url).toBe("https://example.com");
      expect(annAdded.annotation_index).toBe(0);
    });

    it("emits custom_tool_call_input.delta and .done for custom tool calls", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(
        makeChunk({
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "call_1",
                type: "custom",
                custom: { name: "my_tool", input: "" },
              } as any,
            ],
          },
        })
      );

      const deltaEvents = c.convertStreamChunk(
        makeChunk({
          delta: {
            tool_calls: [{ index: 0, type: "custom", custom: { input: '{"a":1}' } } as any],
          },
        })
      );
      const inputDelta = deltaEvents.find(
        e => e.type === "response.custom_tool_call_input.delta"
      ) as any;
      expect(inputDelta).toBeDefined();
      expect(inputDelta.delta).toBe('{"a":1}');

      const finish = c.convertStreamChunk(makeChunk({ finish_reason: "tool_calls" }));
      const inputDone = finish.find(e => e.type === "response.custom_tool_call_input.done") as any;
      expect(inputDone).toBeDefined();
      expect(inputDone.input).toBe('{"a":1}');

      const itemDone = finish.find(e => e.type === "response.output_item.done") as any;
      expect(itemDone).toBeDefined();
      expect(itemDone.item.type).toBe("custom_tool_call");
    });

    it("emits response.failed on content_filter finish", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(makeChunk({ delta: { content: "Hi" } }));

      const events = c.convertStreamChunk(makeChunk({ finish_reason: "content_filter" }));

      const failed = events.find(e => e.type === "response.failed") as any;
      expect(failed).toBeDefined();
      expect(failed.response.status).toBe("failed");
    });

    it("carries accumulated output items on the terminal response", () => {
      const c = new ResponsesToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(makeChunk({ delta: { content: "Hello" } }));
      c.convertStreamChunk(makeChunk({ finish_reason: "stop" }));

      const lastEvents = c.convertStreamChunk(makeChunk({ finish_reason: "stop" }));
      const completed = lastEvents.find(e => e.type === "response.completed") as any;
      expect(completed).toBeDefined();
      expect(completed.response.output.length).toBeGreaterThanOrEqual(1);
      expect(completed.response.output[0].type).toBe("message");
    });
  });

  describe("convertRequest - advanced input items", () => {
    it("converts reasoning input items", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: [
          {
            type: "reasoning",
            id: "r_1",
            summary: [{ type: "summary_text", text: "Deep thought" }],
          },
        ],
      } as any);

      const msg = result.messages[0] as any;
      expect(msg.role).toBe("assistant");
      expect(msg.reasoning_content).toBe("Deep thought");
    });

    it("converts assistant message with output_text content", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Hello" }],
          },
        ],
      } as any);

      const msg = result.messages[0] as any;
      expect(msg.role).toBe("assistant");
      expect(msg.content).toBe("Hello");
    });

    it("converts input_image content", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_image", image_url: "https://example.com/img.png" }],
          },
        ],
      } as any);

      const content = result.messages[0] as any;
      expect(content.content[0].type).toBe("image_url");
    });

    it("converts input_file content", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_file", file_data: "data:application/pdf;base64,abc" }],
          },
        ],
      } as any);

      const content = result.messages[0] as any;
      expect(content.content[0].type).toBe("file");
    });

    it("converts named tool_choice", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hi",
        tool_choice: { type: "function", name: "get_weather" },
      } as any);

      expect(result.tool_choice).toEqual({ type: "function", function: { name: "get_weather" } });
    });

    it("maps json_object text format to response_format", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hi",
        text: { format: { type: "json_object" } },
      } as any);

      expect(result.response_format).toEqual({ type: "json_object" });
    });

    it("maps text format type to response_format", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        input: "Hi",
        text: { format: { type: "text" } },
      } as any);

      expect(result.response_format).toEqual({ type: "text" });
    });
  });
});
