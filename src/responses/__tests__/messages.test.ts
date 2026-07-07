import type Anthropic from "@anthropic-ai/sdk";
import { APIError } from "@anthropic-ai/sdk";
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

    it("coerces non-string function_call_output to string", () => {
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
            output: { temperature: 72, unit: "F" } as any,
          },
        ],
      });

      const toolResult = (result.messages[2].content as any[])[0];
      expect(toolResult.type).toBe("tool_result");
      expect(toolResult.content[0].type).toBe("text");
      expect(toolResult.content[0].text).toBe('{"temperature":72,"unit":"F"}');
    });

    it("treats null function_call_output as empty string", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          { role: "user", content: "What is the weather?" },
          {
            type: "function_call",
            id: "fc_1",
            call_id: "call_1",
            name: "get_weather",
            arguments: "{}",
            status: "completed",
          },
          {
            type: "function_call_output",
            id: "fco_1",
            call_id: "call_1",
            output: null as any,
          },
        ],
      });

      const toolResult = (result.messages[2].content as any[])[0];
      expect(toolResult.content[0].text).toBe("");
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

    it("merges assistant text and following function_calls into one assistant message", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          { role: "user", content: "Do two things" },
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "On it" }],
          },
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
      } as any);

      // The assistant text and both tool_use blocks must be in ONE assistant message,
      // not split across two consecutive assistant messages.
      const assistantMessages = result.messages.filter(m => m.role === "assistant");
      expect(assistantMessages).toHaveLength(1);
      const assistantContent = assistantMessages[0].content as any[];
      expect(assistantContent).toHaveLength(3);
      expect(assistantContent[0]).toEqual({ type: "text", text: "On it" });
      expect(assistantContent[1].type).toBe("tool_use");
      expect(assistantContent[1].name).toBe("fn1");
      expect(assistantContent[2].type).toBe("tool_use");
      expect(assistantContent[2].name).toBe("fn2");
    });

    it("flushes pending assistant text when a user message follows", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: [
          { role: "user", content: "Hi" },
          { type: "message", role: "assistant", content: [{ type: "output_text", text: "Hello" }] },
          { role: "user", content: "Bye" },
        ],
      } as any);

      expect(result.messages.map(m => m.role)).toEqual(["user", "assistant", "user"]);
      const assistant = result.messages[1];
      expect(assistant.content).toEqual([{ type: "text", text: "Hello" }]);
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
      expect((result.tools![0] as any).name).toBe("web_search");
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

    it("maps text.format json_object to a permissive object schema", () => {
      const result = converter.convertRequest({
        model: "claude-sonnet-4-20250514",
        input: "Hi",
        text: {
          format: { type: "json_object" },
        } as any,
      });

      expect(result.output_config).toEqual({
        format: {
          type: "json_schema",
          schema: { type: "object" },
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

    it("maps refusal stop_reason to failed status with error", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [{ type: "text", text: "I can't help with that.", citations: null }],
          stop_reason: "refusal",
        })
      );

      expect(result.status).toBe("failed");
      expect(result.error).toEqual({
        code: "content_filter",
        message: "Content refused by the model.",
      });
      expect(result.incomplete_details).toBeNull();
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

    it("converts server_tool_use web_search into web_search_call item", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            {
              type: "server_tool_use",
              id: "srv_1",
              name: "web_search",
              input: { query: "latest news" },
              caller: { type: "direct" },
            } as any,
            {
              type: "web_search_tool_result",
              tool_use_id: "srv_1",
              content: [],
              caller: { type: "direct" },
            } as any,
            { type: "text", text: "Here is the answer", citations: null },
          ],
        })
      );

      const wsCall = result.output.find((o: any) => o.type === "web_search_call") as any;
      expect(wsCall).toBeDefined();
      expect(wsCall.status).toBe("completed");
      expect(wsCall.action).toEqual({ type: "search", query: "latest news" });

      const fnCall = result.output.find((o: any) => o.type === "function_call") as any;
      expect(fnCall).toBeUndefined();
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

      // Responses input_tokens is inclusive of cached tokens; Anthropic reports 100 uncached
      // + 30 cache read, so the Responses total is 130 with cached_tokens as a 30-token subset.
      expect(result.usage?.input_tokens).toBe(130);
      expect(result.usage?.output_tokens).toBe(50);
      expect(result.usage?.total_tokens).toBe(180);
      expect(result.usage?.input_tokens_details?.cached_tokens).toBe(30);
    });

    it("folds cache creation tokens into input_tokens", () => {
      const result = converter.convertResponse(
        makeMessage({
          usage: {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 20,
            cache_read_input_tokens: 30,
            cache_creation: null,
            inference_geo: null,
            server_tool_use: null,
            service_tier: null,
          },
        })
      );

      // 100 uncached + 30 cache read + 20 cache creation = 150.
      expect(result.usage?.input_tokens).toBe(150);
      expect(result.usage?.total_tokens).toBe(200);
      expect(result.usage?.input_tokens_details?.cached_tokens).toBe(30);
    });

    it("preserves generation order for text → tool_use", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            { type: "text", text: "Let me check the weather", citations: null },
            {
              type: "tool_use",
              id: "tu_1",
              name: "get_weather",
              input: { city: "SF" },
              caller: { type: "direct" },
            } as any,
          ],
        })
      );

      expect(result.output.map((o: any) => o.type)).toEqual(["message", "function_call"]);
    });

    it("preserves generation order for thinking → text → tool_use", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            { type: "thinking", thinking: "Planning...", signature: "sig" },
            { type: "text", text: "Calling tool", citations: null },
            {
              type: "tool_use",
              id: "tu_1",
              name: "fn",
              input: {},
              caller: { type: "direct" },
            } as any,
          ],
        })
      );

      expect(result.output.map((o: any) => o.type)).toEqual([
        "reasoning",
        "message",
        "function_call",
      ]);
    });

    it("merges multiple text blocks into one message at first text position", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            { type: "text", text: "Hello ", citations: null },
            {
              type: "tool_use",
              id: "tu_1",
              name: "fn",
              input: {},
              caller: { type: "direct" },
            } as any,
            { type: "text", text: "world", citations: null },
          ],
        })
      );

      // Order: [message (at first text position), function_call]; second text merged in.
      expect(result.output.map((o: any) => o.type)).toEqual(["message", "function_call"]);
      const msg = result.output.find((o: any) => o.type === "message") as any;
      expect(msg.content[0].text).toBe("Hello world");
    });

    it("converts text citations to url_citation annotations", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            {
              type: "text",
              text: "See example.com for details.",
              citations: [
                {
                  type: "web_search_result",
                  start_index: 4,
                  end_index: 15,
                  url: "https://example.com",
                  title: "Example",
                  cited_text: "example.com",
                  encrypted_content: "",
                } as any,
              ],
            } as any,
          ],
        })
      );

      const msg = result.output.find((o: any) => o.type === "message") as any;
      expect(msg.content[0].annotations).toHaveLength(1);
      const ann = msg.content[0].annotations[0];
      expect(ann.type).toBe("url_citation");
      expect(ann.url).toBe("https://example.com");
      expect(ann.title).toBe("Example");
      expect(ann.start_index).toBe(4);
      expect(ann.end_index).toBe(15);
    });

    it("shifts citation indices when merging multiple text blocks", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            { type: "text", text: "Hello ", citations: null },
            {
              type: "text",
              text: "world",
              citations: [
                {
                  type: "web_search_result",
                  start_index: 0,
                  end_index: 5,
                  url: "https://example.com",
                  title: "World",
                } as any,
              ],
            } as any,
          ],
        })
      );

      const msg = result.output.find((o: any) => o.type === "message") as any;
      // "Hello " is 6 chars, so the second block's citation shifts by 6.
      expect(msg.content[0].annotations[0].start_index).toBe(6);
      expect(msg.content[0].annotations[0].end_index).toBe(11);
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

    it("emits response.failed and error on refusal stop", () => {
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
        delta: { stop_reason: "refusal", stop_sequence: null, container: null },
        usage: {
          output_tokens: 5,
          input_tokens: null,
          cache_creation_input_tokens: null,
          cache_read_input_tokens: null,
          server_tool_use: null,
        },
      });

      const events = c.convertStreamEvent({ type: "message_stop" });

      // Must NOT emit response.completed carrying a "failed" status.
      expect(events.find(e => e.type === "response.completed")).toBeUndefined();

      const failed = events.find(e => e.type === "response.failed") as any;
      expect(failed).toBeDefined();
      expect(failed.response.status).toBe("failed");
      expect(failed.response.error).toEqual({
        code: "content_filter",
        message: "Content refused by the model.",
      });

      const error = events.find(e => e.type === "error") as any;
      expect(error).toBeDefined();
      expect(error.code).toBe("content_filter");
      expect(error.param).toBeNull();
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

    it("emits web_search_call events for server_tool_use web_search", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());

      const startEvents = c.convertStreamEvent({
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

      const itemAdded = startEvents.find(e => e.type === "response.output_item.added") as any;
      expect(itemAdded).toBeDefined();
      expect(itemAdded.item.type).toBe("web_search_call");
      expect(itemAdded.item.status).toBe("in_progress");

      const inProgress = startEvents.find(
        e => e.type === "response.web_search_call.in_progress"
      ) as any;
      expect(inProgress).toBeDefined();
      expect(inProgress.item_id).toBe(itemAdded.item.id);

      // searching fires once at content_block_start, not on every delta.
      const searching = startEvents.find(
        e => e.type === "response.web_search_call.searching"
      ) as any;
      expect(searching).toBeDefined();
      expect(searching.item_id).toBe(itemAdded.item.id);

      const searchEvents = c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: '{"query":"news"}' },
      });
      // No additional searching events on deltas.
      expect(
        searchEvents.filter((e: any) => e.type === "response.web_search_call.searching")
      ).toHaveLength(0);

      c.convertStreamEvent({ type: "content_block_stop", index: 0 });

      const doneEvents = c.convertStreamEvent({
        type: "content_block_start",
        index: 1,
        content_block: {
          type: "web_search_tool_result",
          tool_use_id: "srv_1",
          content: [],
          caller: { type: "direct" },
        },
      });
      const completed = doneEvents.find(
        e => e.type === "response.web_search_call.completed"
      ) as any;
      expect(completed).toBeDefined();
      expect(completed.item_id).toBe(itemAdded.item.id);

      const itemDone = doneEvents.find(e => e.type === "response.output_item.done") as any;
      expect(itemDone).toBeDefined();
      expect(itemDone.item.type).toBe("web_search_call");
      expect(itemDone.item.status).toBe("completed");
      expect(itemDone.item.action.query).toBe("news");
    });

    it("accumulates web_search query across multiple input_json_delta chunks", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());
      c.convertStreamEvent({
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
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: '{"query":"latest ' },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: 'news today"}' },
      });
      c.convertStreamEvent({ type: "content_block_stop", index: 0 });

      const doneEvents = c.convertStreamEvent({
        type: "content_block_start",
        index: 1,
        content_block: {
          type: "web_search_tool_result",
          tool_use_id: "srv_1",
          content: [],
          caller: { type: "direct" },
        },
      });
      const itemDone = doneEvents.find(e => e.type === "response.output_item.done") as any;
      expect(itemDone.item.action.query).toBe("latest news today");
    });

    it("emits searching exactly once regardless of delta count", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());
      const startEvents = c.convertStreamEvent({
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
      const startSearching = startEvents.filter(
        (e: any) => e.type === "response.web_search_call.searching"
      );
      expect(startSearching).toHaveLength(1);

      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: '{"query":"a"' },
      });
      const delta1 = c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: "}" },
      });
      expect(
        delta1.filter((e: any) => e.type === "response.web_search_call.searching")
      ).toHaveLength(0);
    });

    it("backfills web_search_call done at message_stop when result never arrives", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());
      c.convertStreamEvent({
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
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: '{"query":"partial' },
      });
      // No content_block_stop, no web_search_tool_result — stream truncated.

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
      const completed = events.find(e => e.type === "response.web_search_call.completed") as any;
      expect(completed).toBeDefined();

      const itemDone = events.find(e => e.type === "response.output_item.done") as any;
      expect(itemDone).toBeDefined();
      expect(itemDone.item.type).toBe("web_search_call");
      // Partial JSON `{"query":"partial` should still yield "partial" via regex fallback.
      expect(itemDone.item.action.query).toBe("partial");

      const incomplete = events.find(e => e.type === "response.incomplete") as any;
      expect(incomplete).toBeDefined();
      // The backfilled item should also appear in the final response.output.
      const wsInOutput = incomplete.response.output.find((o: any) => o.type === "web_search_call");
      expect(wsInOutput).toBeDefined();
      expect(wsInOutput.action.query).toBe("partial");
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

    it("marks response completed when max_tokens hits after a complete tool_use", () => {
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
        } as any,
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: '{"x":1}' },
      });
      c.convertStreamEvent({ type: "content_block_stop", index: 0 });

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
      const completed = events.find(e => e.type === "response.completed") as any;
      const incomplete = events.find(e => e.type === "response.incomplete") as any;
      expect(completed).toBeDefined();
      expect(incomplete).toBeUndefined();
      expect(completed.response.status).toBe("completed");
    });

    it("marks response incomplete when max_tokens hits mid tool_use (no content_block_stop)", () => {
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
        } as any,
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: '{"x":' },
      });
      // No content_block_stop — block truncated by max_tokens.

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
      expect(incomplete.response.status).toBe("incomplete");
    });

    it("assigns distinct output_index for text and tool_use without reasoning", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());

      const textStart = c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "", citations: null },
      });
      const textItemAdded = textStart.find(e => e.type === "response.output_item.added") as any;

      const textDelta = c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "thinking..." },
      });
      const textDeltaEvent = textDelta.find(e => e.type === "response.output_text.delta") as any;

      c.convertStreamEvent({ type: "content_block_stop", index: 0 });

      const toolStart = c.convertStreamEvent({
        type: "content_block_start",
        index: 1,
        content_block: {
          type: "tool_use",
          id: "tu_1",
          name: "fn",
          input: {},
          caller: { type: "direct" },
        } as any,
      });
      const toolItemAdded = toolStart.find(e => e.type === "response.output_item.added") as any;

      const toolDelta = c.convertStreamEvent({
        type: "content_block_delta",
        index: 1,
        delta: { type: "input_json_delta", partial_json: "{}" },
      });
      const toolArgsDelta = toolDelta.find(
        e => e.type === "response.function_call_arguments.delta"
      ) as any;

      // Text and tool_use must not share the same output_index.
      expect(textItemAdded.output_index).toBe(0);
      expect(toolItemAdded.output_index).toBe(1);
      // Deltas reuse the indices assigned at item start.
      expect(textDeltaEvent.output_index).toBe(textItemAdded.output_index);
      expect(toolArgsDelta.output_index).toBe(toolItemAdded.output_index);
    });

    it("assigns sequential output_index for reasoning → text → tool_use", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());

      const reasoningStart = c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "thinking", thinking: "", signature: "" },
      });
      const reasoningAdded = reasoningStart.find(
        e => e.type === "response.output_item.added"
      ) as any;
      c.convertStreamEvent({ type: "content_block_stop", index: 0 });

      const textStart = c.convertStreamEvent({
        type: "content_block_start",
        index: 1,
        content_block: { type: "text", text: "", citations: null },
      });
      const textAdded = textStart.find(e => e.type === "response.output_item.added") as any;
      c.convertStreamEvent({ type: "content_block_stop", index: 1 });

      const toolStart = c.convertStreamEvent({
        type: "content_block_start",
        index: 2,
        content_block: {
          type: "tool_use",
          id: "tu_1",
          name: "fn",
          input: {},
          caller: { type: "direct" },
        } as any,
      });
      const toolAdded = toolStart.find(e => e.type === "response.output_item.added") as any;

      expect(reasoningAdded.output_index).toBe(0);
      expect(textAdded.output_index).toBe(1);
      expect(toolAdded.output_index).toBe(2);
    });

    it("emits done events for reasoning on content_block_stop", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "thinking", thinking: "", signature: "" },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "Hello" },
      });
      const stopEvents = c.convertStreamEvent({
        type: "content_block_stop",
        index: 0,
      });

      const textDone = stopEvents.find(
        e => e.type === "response.reasoning_summary_text.done"
      ) as any;
      expect(textDone).toBeDefined();
      expect(textDone.text).toBe("Hello");

      const partDone = stopEvents.find(
        e => e.type === "response.reasoning_summary_part.done"
      ) as any;
      expect(partDone).toBeDefined();

      const itemDone = stopEvents.find(e => e.type === "response.output_item.done") as any;
      expect(itemDone).toBeDefined();
      expect(itemDone.item.type).toBe("reasoning");
      expect(itemDone.item.summary[0].text).toBe("Hello");
    });

    it("resets reasoning text accumulator between thinking blocks", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());

      // First thinking block
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "thinking", thinking: "", signature: "" },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "Hello" },
      });
      const stop1 = c.convertStreamEvent({
        type: "content_block_stop",
        index: 0,
      });
      const itemDone1 = stop1.find(e => e.type === "response.output_item.done") as any;
      const firstItemId = itemDone1.item.id;
      expect(itemDone1.item.summary[0].text).toBe("Hello");

      // Second thinking block — must not inherit the first block's text
      c.convertStreamEvent({
        type: "content_block_start",
        index: 1,
        content_block: { type: "thinking", thinking: "", signature: "" },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 1,
        delta: { type: "thinking_delta", thinking: " World" },
      });
      const stop2 = c.convertStreamEvent({
        type: "content_block_stop",
        index: 1,
      });

      const textDone2 = stop2.find(e => e.type === "response.reasoning_summary_text.done") as any;
      expect(textDone2.text).toBe(" World");

      const itemDone2 = stop2.find(e => e.type === "response.output_item.done") as any;
      expect(itemDone2.item.id).not.toBe(firstItemId);
      expect(itemDone2.item.summary[0].text).toBe(" World");
    });

    it("emits done events for function_call on content_block_stop", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: {
          type: "tool_use",
          id: "tu_1",
          name: "get_weather",
          input: {},
          caller: { type: "direct" },
        } as any,
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: '{"city":"SF"}' },
      });
      const stopEvents = c.convertStreamEvent({
        type: "content_block_stop",
        index: 0,
      });

      const argsDone = stopEvents.find(
        e => e.type === "response.function_call_arguments.done"
      ) as any;
      expect(argsDone).toBeDefined();
      expect(argsDone.arguments).toBe('{"city":"SF"}');

      const itemDone = stopEvents.find(e => e.type === "response.output_item.done") as any;
      expect(itemDone).toBeDefined();
      expect(itemDone.item.type).toBe("function_call");
      expect(itemDone.item.status).toBe("completed");
      expect(itemDone.item.arguments).toBe('{"city":"SF"}');
      expect(itemDone.item.call_id).toBe("tu_1");
      expect(itemDone.item.name).toBe("get_weather");
    });

    it("emits done events for message on message_stop", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "", citations: null },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "Hello " },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "world" },
      });
      c.convertStreamEvent({ type: "content_block_stop", index: 0 });
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
      const textDone = events.find(e => e.type === "response.output_text.done") as any;
      expect(textDone).toBeDefined();
      expect(textDone.text).toBe("Hello world");

      const partDone = events.find(e => e.type === "response.content_part.done") as any;
      expect(partDone).toBeDefined();
      expect(partDone.part.text).toBe("Hello world");

      const itemDone = events.find(e => e.type === "response.output_item.done") as any;
      expect(itemDone).toBeDefined();
      expect(itemDone.item.type).toBe("message");
      expect(itemDone.item.status).toBe("completed");
      expect(itemDone.item.content[0].text).toBe("Hello world");

      // Done events must fire before response.completed.
      const completedIdx = events.findIndex(e => e.type === "response.completed");
      const itemDoneIdx = events.findIndex(e => e.type === "response.output_item.done");
      expect(itemDoneIdx).toBeLessThan(completedIdx);
    });

    it("emits citations_delta as annotations on message done events", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "", citations: null },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "See example.com" },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: {
          type: "citations_delta",
          citation: {
            type: "web_search_result",
            start_index: 4,
            end_index: 15,
            url: "https://example.com",
            title: "Example",
          },
        } as any,
      });
      c.convertStreamEvent({ type: "content_block_stop", index: 0 });
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
      const partDone = events.find(e => e.type === "response.content_part.done") as any;
      expect(partDone.part.annotations).toHaveLength(1);
      expect(partDone.part.annotations[0]).toEqual({
        type: "url_citation",
        start_index: 4,
        end_index: 15,
        url: "https://example.com",
        title: "Example",
      });

      const itemDone = events.find(e => e.type === "response.output_item.done") as any;
      expect(itemDone.item.content[0].annotations).toHaveLength(1);
      expect(itemDone.item.content[0].annotations[0].url).toBe("https://example.com");
    });

    it("shifts citation indices across multiple streamed text blocks", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());
      // Block 1: "Hello " (6 chars)
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "", citations: null },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "Hello " },
      });
      c.convertStreamEvent({ type: "content_block_stop", index: 0 });
      // Block 2: "world" with citation at start_index=0 relative to "world"
      c.convertStreamEvent({
        type: "content_block_start",
        index: 1,
        content_block: { type: "text", text: "", citations: null },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 1,
        delta: { type: "text_delta", text: "world" },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 1,
        delta: {
          type: "citations_delta",
          citation: {
            type: "web_search_result",
            start_index: 0,
            end_index: 5,
            url: "https://example.com",
            title: "World",
          },
        } as any,
      });
      c.convertStreamEvent({ type: "content_block_stop", index: 1 });
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
      const itemDone = events.find(e => e.type === "response.output_item.done") as any;
      // Citation shifts by 6 (length of "Hello ").
      expect(itemDone.item.content[0].annotations[0].start_index).toBe(6);
      expect(itemDone.item.content[0].annotations[0].end_index).toBe(11);
      expect(itemDone.item.content[0].text).toBe("Hello world");
    });

    it("populates response.output on response.completed with all items", () => {
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());

      // reasoning block
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "thinking", thinking: "", signature: "" },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "Hmm" },
      });
      c.convertStreamEvent({ type: "content_block_stop", index: 0 });

      // tool_use block
      c.convertStreamEvent({
        type: "content_block_start",
        index: 1,
        content_block: {
          type: "tool_use",
          id: "tu_1",
          name: "fn",
          input: {},
          caller: { type: "direct" },
        } as any,
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 1,
        delta: { type: "input_json_delta", partial_json: '{"x":1}' },
      });
      c.convertStreamEvent({ type: "content_block_stop", index: 1 });

      // text block
      c.convertStreamEvent({
        type: "content_block_start",
        index: 2,
        content_block: { type: "text", text: "", citations: null },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 2,
        delta: { type: "text_delta", text: "Answer" },
      });
      c.convertStreamEvent({ type: "content_block_stop", index: 2 });

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

      const output = completed.response.output as any[];
      expect(output).toHaveLength(3);
      expect(output.map((o: any) => o.type)).toEqual(["reasoning", "function_call", "message"]);
      expect(output[0].summary[0].text).toBe("Hmm");
      expect(output[1].arguments).toBe('{"x":1}');
      expect(output[1].status).toBe("completed");
      expect(output[2].content[0].text).toBe("Answer");
    });

    it("orders response.output by output_index, not completion order", () => {
      // reasoning(0) → text(1) → tool_use(2), but the text block stays open until
      // message_stop while tool_use completes first. Completion order is
      // [reasoning, tool_use, text]; output must be sorted back to [reasoning, text, tool_use].
      const c = new ResponsesToMessagesConverter();
      c.convertStreamEvent(makeMessageStart());

      // reasoning block (index 0) — opens and closes
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "thinking", thinking: "", signature: "" },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "Plan" },
      });
      c.convertStreamEvent({ type: "content_block_stop", index: 0 });

      // text block (index 1) — opens, receives text, but does NOT stop before tool_use
      c.convertStreamEvent({
        type: "content_block_start",
        index: 1,
        content_block: { type: "text", text: "", citations: null },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 1,
        delta: { type: "text_delta", text: "Answer" },
      });

      // tool_use block (index 2) — opens and closes, finalizing before the text block
      c.convertStreamEvent({
        type: "content_block_start",
        index: 2,
        content_block: {
          type: "tool_use",
          id: "tu_1",
          name: "fn",
          input: {},
          caller: { type: "direct" },
        } as any,
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 2,
        delta: { type: "input_json_delta", partial_json: '{"x":1}' },
      });
      c.convertStreamEvent({ type: "content_block_stop", index: 2 });

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
      const output = completed.response.output as any[];

      expect(output.map((o: any) => o.type)).toEqual(["reasoning", "message", "function_call"]);
      expect(output[1].content[0].text).toBe("Answer");
      expect(output[2].arguments).toBe('{"x":1}');
    });

    it("includes truncated tool_call in response.output with incomplete status", () => {
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
        } as any,
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: '{"x":' },
      });
      // No content_block_stop — truncated by max_tokens.

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

      const output = incomplete.response.output as any[];
      expect(output).toHaveLength(1);
      expect(output[0].type).toBe("function_call");
      expect(output[0].status).toBe("incomplete");
      expect(output[0].arguments).toBe('{"x":');
    });
  });

  describe("convertStream - error handling", () => {
    it("emits response.failed and error events when upstream throws APIError", async () => {
      const c = new ResponsesToMessagesConverter();
      async function* failingStream(): AsyncIterable<Anthropic.RawMessageStreamEvent> {
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
        } as Anthropic.RawMessageStreamEvent;
        throw new APIError(529, { message: "overloaded" }, "overloaded", undefined as any);
      }

      const events: any[] = [];
      for await (const e of c.convertStream(failingStream())) {
        events.push(e);
      }

      const failed = events.find(e => e.type === "response.failed") as any;
      expect(failed).toBeDefined();
      expect(failed.response.status).toBe("failed");
      expect(failed.response.error).toEqual({
        code: "server_error",
        message: "529 overloaded",
      });

      const error = events.find(e => e.type === "error") as any;
      expect(error).toBeDefined();
      expect(error.code).toBe("server_error");
      expect(error.message).toBe("529 overloaded");
      expect(error.param).toBeNull();
    });

    it("maps 429 APIError to rate_limit_exceeded", async () => {
      const c = new ResponsesToMessagesConverter();
      async function* failingStream(): AsyncIterable<Anthropic.RawMessageStreamEvent> {
        throw new APIError(429, { message: "rate limited" }, "rate limited", undefined as any);
      }

      const events: any[] = [];
      for await (const e of c.convertStream(failingStream())) {
        events.push(e);
      }

      const failed = events.find(e => e.type === "response.failed") as any;
      expect(failed.response.error.code).toBe("rate_limit_exceeded");

      const error = events.find(e => e.type === "error") as any;
      expect(error.code).toBe("rate_limit_exceeded");
    });

    it("re-throws non-APIError exceptions", async () => {
      const c = new ResponsesToMessagesConverter();
      async function* failingStream(): AsyncIterable<Anthropic.RawMessageStreamEvent> {
        throw new TypeError("programming bug");
      }

      let threw = false;
      let caught: unknown = null;
      try {
        for await (const _ of c.convertStream(failingStream())) {
          // drain
        }
      } catch (err) {
        threw = true;
        caught = err;
      }
      expect(threw).toBe(true);
      expect(caught).toBeInstanceOf(TypeError);
    });
  });
});
