import type OpenAI from "openai";
import { ChatCompletionToResponsesConverter } from "../responses";

describe("ChatCompletionToResponsesConverter", () => {
  let converter: ChatCompletionToResponsesConverter;

  beforeEach(() => {
    converter = new ChatCompletionToResponsesConverter();
  });

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

    it("uses deprecated max_tokens when max_completion_tokens is absent", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        max_tokens: 500,
      });

      expect(result.max_output_tokens).toBe(500);
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
        web_search_options: {
          search_context_size: "high",
          user_location: {
            type: "approximate",
            approximate: { city: "Paris", country: "FR", timezone: "Europe/Paris" },
          },
        },
      } as any);

      expect((result.tools as any[]).find((t: any) => t.type === "web_search")).toEqual({
        type: "web_search",
        search_context_size: "high",
        user_location: {
          type: "approximate",
          city: "Paris",
          country: "FR",
          timezone: "Europe/Paris",
        },
      });
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

    it("preserves structured output options and verbosity", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "answer",
            description: "A structured answer",
            schema: { type: "object" },
            strict: true,
          },
        },
        verbosity: "low",
      });

      expect(result.text).toEqual({
        format: {
          type: "json_schema",
          name: "answer",
          description: "A structured answer",
          schema: { type: "object" },
          strict: true,
        },
        verbosity: "low",
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

    it("preserves shared request controls", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        store: false,
        safety_identifier: "user_hash",
        user: "legacy_user",
        stream: true,
        stream_options: { include_usage: true, include_obfuscation: false },
      });

      expect(result.store).toBe(false);
      expect(result.safety_identifier).toBe("user_hash");
      expect(result.user).toBe("legacy_user");
      expect(result.stream_options).toEqual({ include_obfuscation: false });
    });

    it("preserves metadata, prompt-cache controls, and service tier", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        metadata: { test: "responses" },
        prompt_cache_key: "cache-key",
        prompt_cache_retention: "24h",
        service_tier: "priority",
      });

      expect(result.metadata).toEqual({ test: "responses" });
      expect(result.prompt_cache_key).toBe("cache-key");
      expect(result.prompt_cache_retention).toBe("24h");
      expect(result.service_tier).toBe("priority");
    });

    it("does not invent Responses mappings for Chat Completions-only controls", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        n: 2,
        frequency_penalty: 0.2,
        presence_penalty: 0.3,
        logit_bias: { "42": 1 },
        seed: 123,
        stop: ["END"],
        modalities: ["text"],
        prediction: { type: "content", content: "predicted" },
      });

      expect(result).not.toHaveProperty("n");
      expect(result).not.toHaveProperty("frequency_penalty");
      expect(result).not.toHaveProperty("presence_penalty");
      expect(result).not.toHaveProperty("logit_bias");
      expect(result).not.toHaveProperty("seed");
      expect(result).not.toHaveProperty("stop");
      expect(result).not.toHaveProperty("modalities");
      expect(result).not.toHaveProperty("prediction");
    });

    it("passes through parallel_tool_calls", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        parallel_tool_calls: false,
      } as any);

      expect((result as any).parallel_tool_calls).toBe(false);
    });

    it("preserves an explicit stream false value", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        stream: false,
      });

      expect((result as any).stream).toBe(false);
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

    it("omits user content parts that have no standard Responses mapping", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Transcribe this" },
              { type: "input_audio", input_audio: { data: "base64", format: "wav" } },
            ],
          },
        ],
      });

      expect((result.input as any[])[0].content).toEqual([
        { type: "input_text", text: "Transcribe this" },
      ]);
    });

    it("converts custom tools, custom calls, and custom tool outputs", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [
          { role: "user", content: "Run code" },
          {
            role: "assistant",
            content: "I will run it.",
            tool_calls: [
              {
                id: "call_custom",
                type: "custom",
                custom: { name: "code_exec", input: "print(1)" },
              },
            ],
          },
          { role: "tool", tool_call_id: "call_custom", content: "1" },
        ],
        tools: [
          {
            type: "custom",
            custom: {
              name: "code_exec",
              description: "Execute code",
              format: {
                type: "grammar",
                grammar: { syntax: "lark", definition: "start: /.+/" },
              },
            },
          },
        ],
        tool_choice: { type: "custom", custom: { name: "code_exec" } },
      });

      expect(result.tools).toEqual([
        {
          type: "custom",
          name: "code_exec",
          description: "Execute code",
          format: { type: "grammar", syntax: "lark", definition: "start: /.+/" },
        },
      ]);
      expect(result.tool_choice).toEqual({ type: "custom", name: "code_exec" });
      expect(result.input).toEqual([
        { role: "user", type: "message", content: "Run code" },
        {
          role: "assistant",
          type: "message",
          content: "I will run it.",
        },
        {
          type: "custom_tool_call",
          call_id: "call_custom",
          name: "code_exec",
          input: "print(1)",
        },
        { type: "custom_tool_call_output", call_id: "call_custom", output: "1" },
      ]);
    });

    it("converts allowed tool choices", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Use one tool" }],
        tool_choice: {
          type: "allowed_tools",
          allowed_tools: {
            mode: "required",
            tools: [
              { type: "function", function: { name: "lookup" } },
              { type: "custom", custom: { name: "shell" } },
            ],
          },
        },
      });

      expect(result.tool_choice).toEqual({
        type: "allowed_tools",
        mode: "required",
        tools: [
          { type: "function", name: "lookup" },
          { type: "custom", name: "shell" },
        ],
      });
    });

    it("converts deprecated function definitions when tools are absent", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Weather?" }],
        functions: [
          {
            name: "weather",
            description: "Get weather",
            parameters: { type: "object" },
          },
        ],
        function_call: { name: "weather" },
      });

      expect(result.tools).toEqual([
        {
          type: "function",
          name: "weather",
          description: "Get weather",
          strict: null,
          parameters: { type: "object" },
        },
      ]);
      expect(result.tool_choice).toEqual({ type: "function", name: "weather" });
    });

    it("converts deprecated function call history with matching synthetic call IDs", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [
          { role: "user", content: "Weather?" },
          {
            role: "assistant",
            content: null,
            function_call: { name: "weather", arguments: '{"city":"SF"}' },
          },
          { role: "function", name: "weather", content: "72F" },
        ],
        functions: [{ name: "weather" }],
      });

      expect(result.input).toEqual([
        { role: "user", type: "message", content: "Weather?" },
        {
          type: "function_call",
          name: "weather",
          call_id: "call_legacy_1",
          arguments: '{"city":"SF"}',
        },
        {
          type: "function_call_output",
          call_id: "call_legacy_1",
          output: "72F",
        },
      ]);
    });

    it("uses valid easy-input content for assistant history", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [
          {
            role: "assistant",
            content: [
              { type: "text", text: "First" },
              { type: "refusal", refusal: "Cannot continue" },
            ],
          },
          { role: "user", content: "Why?" },
        ],
      });

      expect((result.input as any[])[0]).toEqual({
        role: "assistant",
        type: "message",
        content: [
          { type: "output_text", text: "First" },
          { type: "refusal", refusal: "Cannot continue" },
        ],
      });
    });

    it("preserves a standalone assistant refusal as a refusal content part", () => {
      const result = converter.convertRequest({
        model: "gpt-4o",
        messages: [
          { role: "assistant", content: null, refusal: "Cannot continue" },
          { role: "user", content: "Why?" },
        ],
      });

      expect((result.input as any[])[0]).toEqual({
        role: "assistant",
        type: "message",
        content: [{ type: "refusal", refusal: "Cannot continue" }],
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

    it("omits usage when the Responses payload has no usage", () => {
      const result = converter.convertResponse(makeResponse({ usage: undefined as any }));

      expect(result.usage).toBeUndefined();
    });

    it("converts a basic text response", () => {
      const result = converter.convertResponse(makeResponse());

      expect(result.id).toBe("resp_123");
      expect(result.model).toBe("gpt-4o");
      expect(result.object).toBe("chat.completion");
      expect(result.created).toBe(1700000000);
      expect(result.choices[0].message.content).toBe("Hello!");
      expect(result.choices[0].message.annotations).toEqual([]);
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

    it("converts Responses function calls back to deprecated CC function_call mode", () => {
      converter.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Weather?" }],
        functions: [{ name: "get_weather" }],
        function_call: "auto",
      });

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
        })
      );

      expect(result.choices[0]).toMatchObject({
        finish_reason: "function_call",
        message: {
          function_call: { name: "get_weather", arguments: '{"city":"SF"}' },
        },
      });
      expect(result.choices[0].message.tool_calls).toBeUndefined();
    });

    it("keeps an incomplete status finish reason when a tool call is partial", () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [
            {
              type: "function_call",
              id: "fc_1",
              call_id: "call_1",
              name: "get_weather",
              arguments: '{"city":',
              status: "incomplete",
            } as any,
          ],
          status: "incomplete",
          incomplete_details: { reason: "max_output_tokens" },
        })
      );

      expect(result.choices[0].finish_reason).toBe("length");
      expect(result.choices[0].message.tool_calls).toHaveLength(1);
    });

    it("converts custom_tool_call output items to custom tool_calls", () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [
            {
              type: "custom_tool_call",
              id: "ctc_1",
              call_id: "call_1",
              name: "code_exec",
              input: "print(1)",
            } as any,
          ],
        })
      );

      expect(result.choices[0].message.tool_calls).toEqual([
        {
          id: "call_1",
          type: "custom",
          custom: { name: "code_exec", input: "print(1)" },
        },
      ]);
      expect(result.choices[0].finish_reason).toBe("tool_calls");
    });

    it("combines standard message parts, reasoning, and tool calls into one choice", () => {
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
                  text: "Hello ",
                  annotations: [],
                  logprobs: [{ token: "Hello", logprob: -0.1, bytes: [72], top_logprobs: [] }],
                },
                {
                  type: "output_text",
                  text: "world",
                  annotations: [
                    {
                      type: "url_citation",
                      title: "Example",
                      url: "https://example.com",
                      start_index: 0,
                      end_index: 5,
                    },
                  ],
                  logprobs: [],
                },
              ],
            } as any,
            {
              type: "reasoning",
              id: "r_1",
              summary: [{ type: "summary_text", text: "Thinking..." }],
            } as any,
            {
              type: "function_call",
              id: "fc_1",
              call_id: "call_1",
              name: "lookup",
              arguments: "{}",
              status: "completed",
            } as any,
          ],
        })
      );

      expect(result.choices).toHaveLength(1);
      expect(result.choices[0]).toMatchObject({
        index: 0,
        finish_reason: "tool_calls",
        message: {
          content: "Hello world",
          reasoning: "Thinking...",
          annotations: [
            {
              type: "url_citation",
              url_citation: {
                start_index: 6,
                end_index: 11,
              },
            },
          ],
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "lookup", arguments: "{}" },
            },
          ],
        },
      });
      expect(result.choices[0].logprobs?.content?.[0]).toEqual({
        token: "Hello",
        logprob: -0.1,
        bytes: [72],
        top_logprobs: [],
      });
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

    it("preserves every reasoning item and summary part in output order", () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [
            {
              type: "reasoning",
              id: "r_1",
              summary: [
                { type: "summary_text", text: "First " },
                { type: "summary_text", text: "step. " },
              ],
              encrypted_content: "encrypted-1",
            } as any,
            {
              type: "reasoning",
              id: "r_2",
              summary: [{ type: "summary_text", text: "Second step." }],
              encrypted_content: "encrypted-2",
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

      expect((result.choices[0].message as any).reasoning).toBe("First step. Second step.");
      expect((result.choices[0].message as any).reasoning_details).toEqual([
        {
          index: "0",
          format: "openai-responses-v1",
          type: "reasoning.summary",
          summary: "First ",
        },
        {
          index: "0",
          format: "openai-responses-v1",
          type: "reasoning.summary",
          summary: "step. ",
        },
        {
          id: "r_1",
          index: "0",
          format: "openai-responses-v1",
          type: "reasoning.encrypted",
          data: "encrypted-1",
        },
        {
          index: "0",
          format: "openai-responses-v1",
          type: "reasoning.summary",
          summary: "Second step.",
        },
        {
          id: "r_2",
          index: "0",
          format: "openai-responses-v1",
          type: "reasoning.encrypted",
          data: "encrypted-2",
        },
      ]);
    });

    it("returns a valid empty choice when Responses has no output items", () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [],
          status: "incomplete",
          incomplete_details: { reason: "content_filter" },
        })
      );

      expect(result.choices).toEqual([
        {
          index: 0,
          finish_reason: "content_filter",
          message: { role: "assistant", content: null, refusal: null },
          logprobs: null,
        },
      ]);
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
      expect(result.choices[0].logprobs).toBeNull();
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

    it("converts streaming text logprobs when requested", () => {
      const c = new ChatCompletionToResponsesConverter();
      c.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        stream: true,
        top_logprobs: 5,
      } as any);
      initStream(c);

      const result = first(
        c.convertStreamEvent({
          type: "response.output_text.delta",
          delta: "Hi",
          item_id: "msg_1",
          output_index: 0,
          content_index: 0,
          sequence_number: 1,
          logprobs: [
            {
              token: "Hi",
              logprob: -0.1,
              top_logprobs: [{ token: "Hello", logprob: -0.5 }],
            },
          ],
        } as any)
      );

      expect(result.choices[0].logprobs).toEqual({
        content: [
          {
            token: "Hi",
            logprob: -0.1,
            bytes: null,
            top_logprobs: [{ token: "Hello", logprob: -0.5, bytes: null }],
          },
        ],
        refusal: null,
      });
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

    it("streams deprecated function_call deltas and finish reason in legacy mode", () => {
      const c = new ChatCompletionToResponsesConverter();
      c.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Weather?" }],
        functions: [{ name: "weather" }],
        function_call: "auto",
        stream: true,
      });
      initStream(c);

      const start = first(
        c.convertStreamEvent({
          type: "response.output_item.added",
          item: {
            type: "function_call",
            id: "fc_1",
            call_id: "call_1",
            name: "weather",
            arguments: "",
            status: "in_progress",
          },
          output_index: 0,
          sequence_number: 1,
        } as any)
      );
      const delta = first(
        c.convertStreamEvent({
          type: "response.function_call_arguments.delta",
          delta: '{"city',
          item_id: "fc_1",
          output_index: 0,
          sequence_number: 2,
        } as any)
      );
      const done = first(
        c.convertStreamEvent({
          type: "response.completed",
          response: { id: "resp_1", status: "completed" } as any,
          sequence_number: 3,
        })
      );

      expect(start.choices[0].delta.function_call).toEqual({ name: "weather", arguments: "" });
      expect(delta.choices[0].delta.function_call).toEqual({ arguments: '{"city' });
      expect(done.choices[0].finish_reason).toBe("function_call");
    });

    it("emits custom tool call start and input delta", () => {
      const c = new ChatCompletionToResponsesConverter();
      initStream(c);

      const start = first(
        c.convertStreamEvent({
          type: "response.output_item.added",
          item: {
            type: "custom_tool_call",
            id: "ctc_1",
            call_id: "call_custom",
            name: "code_exec",
            input: "",
          },
          output_index: 0,
          sequence_number: 1,
        } as any)
      );
      const delta = first(
        c.convertStreamEvent({
          type: "response.custom_tool_call_input.delta",
          delta: "print(1)",
          item_id: "ctc_1",
          output_index: 0,
          sequence_number: 2,
        } as any)
      );

      expect((start.choices[0].delta.tool_calls?.[0] as any).custom).toEqual({
        name: "code_exec",
        input: "",
      });
      expect((delta.choices[0].delta.tool_calls?.[0] as any).custom).toEqual({
        input: "print(1)",
      });
      expect(delta.choices[0].delta.tool_calls?.[0].index).toBe(0);
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

    it("maps a streaming content-filtered incomplete response", () => {
      const c = new ChatCompletionToResponsesConverter();
      initStream(c);

      const result = first(
        c.convertStreamEvent({
          type: "response.incomplete",
          response: {
            id: "resp_1",
            status: "incomplete",
            incomplete_details: { reason: "content_filter" },
          } as any,
          sequence_number: 10,
        })
      );

      expect(result.choices[0].finish_reason).toBe("content_filter");
    });

    it("emits a trailing usage chunk when stream_options.include_usage is enabled", () => {
      const c = new ChatCompletionToResponsesConverter();
      c.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
        stream: true,
        stream_options: { include_usage: true },
      });
      initStream(c);

      const result = c.convertStreamEvent({
        type: "response.completed",
        response: {
          id: "resp_1",
          status: "completed",
          usage: {
            input_tokens: 10,
            input_tokens_details: { cached_tokens: 2 },
            output_tokens: 5,
            output_tokens_details: { reasoning_tokens: 1 },
            total_tokens: 15,
          },
        } as any,
        sequence_number: 10,
      });

      expect(Array.isArray(result)).toBe(true);
      expect((result as OpenAI.ChatCompletionChunk[])[0].usage).toBeNull();
      expect((result as OpenAI.ChatCompletionChunk[])[1]).toMatchObject({
        choices: [],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
        },
      });
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

    it("counts web search when output_item.done completes a search", () => {
      const c = new ChatCompletionToResponsesConverter();
      c.convertRequest({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Search" }],
        stream: true,
        stream_options: { include_usage: true },
      });
      initStream(c);

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

      const result = c.convertStreamEvent({
        type: "response.completed",
        response: {
          id: "resp_1",
          status: "completed",
          usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
        } as any,
        sequence_number: 10,
      });

      expect(c["streamState"].webSearchCount).toBe(1);
      expect(Array.isArray(result)).toBe(true);
      if (!Array.isArray(result)) throw new Error("expected finish and usage chunks");
      expect(result[0].choices[0].finish_reason).toBe("stop");
      expect((result[1].usage?.prompt_tokens_details as any)?.web_search).toBe(1);
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
            url: "https://example.com",
            title: "Example",
            start_index: 0,
            end_index: 5,
          },
        },
      ]);
    });

    it("ignores streaming annotations without a CC URL citation mapping", () => {
      const c = new ChatCompletionToResponsesConverter();
      initStream(c);

      c.convertStreamEvent({
        type: "response.output_text.annotation.added",
        annotation: { type: "file_citation", file_id: "file_1", filename: "a.txt", index: 0 },
      } as any);

      const result = first(
        c.convertStreamEvent({
          type: "response.completed",
          response: { id: "resp_1", status: "completed" } as any,
          sequence_number: 10,
        })
      );
      expect((result.choices[0].delta as any).annotations).toBeUndefined();
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
