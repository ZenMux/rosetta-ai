import type OpenAI from "openai";
import type {
  GenerateContentParameters,
  GenerateContentResponse,
  Candidate,
  Content,
} from "@google/genai";
import { GeminiToResponsesConverter } from "../responses";

describe("GeminiToResponsesConverter", () => {
  const converter = new GeminiToResponsesConverter();

  describe("convertRequest", () => {
    it("converts basic user content to input", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
      });

      expect(result.model).toBe("gemini-2.0-flash");
      const input = result.input as any[];
      expect(input[0]).toEqual({
        role: "user",
        type: "message",
        content: [{ type: "input_text", text: "Hello" }],
      });
    });

    it("converts string contents to input", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: "Hello" as any,
      });

      const input = result.input as any[];
      expect(input[0]).toEqual({
        role: "user",
        type: "message",
        content: [{ type: "input_text", text: "Hello" }],
      });
    });

    it("converts systemInstruction to instructions", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          systemInstruction: { parts: [{ text: "Be helpful." }] },
        },
      });

      expect(result.instructions).toBe("Be helpful.");
    });

    it("maps maxOutputTokens to max_output_tokens", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: { maxOutputTokens: 2000 },
      });
      expect(result.max_output_tokens).toBe(2000);
    });

    it("maps temperature and topP", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: { temperature: 0.7, topP: 0.9 },
      });
      expect(result.temperature).toBe(0.7);
      expect(result.top_p).toBe(0.9);
    });

    it("converts model content to assistant message", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [
          { role: "user", parts: [{ text: "Hello" }] },
          { role: "model", parts: [{ text: "Hi there!" }] },
          { role: "user", parts: [{ text: "How are you?" }] },
        ],
      });

      const input = result.input as any[];
      expect(input[1]).toEqual({
        role: "assistant",
        type: "message",
        content: [{ type: "output_text", text: "Hi there!" }],
      });
    });

    it("converts functionCall parts to function_call input items", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [
          { role: "user", parts: [{ text: "Weather?" }] },
          {
            role: "model",
            parts: [
              {
                functionCall: {
                  id: "call_1",
                  name: "get_weather",
                  args: { city: "SF" },
                },
              },
            ],
          },
          {
            role: "user",
            parts: [
              {
                functionResponse: {
                  id: "call_1",
                  name: "get_weather",
                  response: { output: "72F" },
                },
              },
            ],
          },
        ],
      });

      const input = result.input as any[];
      expect(input[1]).toEqual({
        type: "function_call",
        call_id: "call_1",
        name: "get_weather",
        arguments: '{"city":"SF"}',
        status: "completed",
      });
      expect(input[2]).toEqual({
        type: "function_call_output",
        call_id: "call_1",
        output: '{"output":"72F"}',
      });
    });

    it("converts functionDeclarations to function tools", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          tools: [
            {
              functionDeclarations: [
                {
                  name: "get_weather",
                  description: "Get weather",
                  parametersJsonSchema: {
                    type: "object",
                    properties: { city: { type: "string" } },
                  },
                },
              ],
            },
          ],
        },
      });

      const tools = result.tools as any[];
      expect(tools[0]).toEqual({
        type: "function",
        name: "get_weather",
        description: "Get weather",
        parameters: { type: "object", properties: { city: { type: "string" } } },
      });
    });

    it("converts googleSearch to web_search_preview tool", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Search" }] }],
        config: {
          tools: [{ googleSearch: {} }],
        },
      });

      const tools = result.tools as any[];
      expect(tools.some((t: any) => t.type === "web_search_preview")).toBe(true);
    });

    it("maps toolConfig AUTO to auto", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          toolConfig: { functionCallingConfig: { mode: "AUTO" as any } },
        },
      });
      expect(result.tool_choice).toBe("auto");
    });

    it("maps toolConfig ANY to required", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          toolConfig: { functionCallingConfig: { mode: "ANY" as any } },
        },
      });
      expect(result.tool_choice).toBe("required");
    });

    it("maps toolConfig ANY with allowedFunctionNames to named function", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          toolConfig: {
            functionCallingConfig: {
              mode: "ANY" as any,
              allowedFunctionNames: ["get_weather"],
            },
          },
        },
      });
      expect(result.tool_choice).toEqual({
        type: "function",
        name: "get_weather",
      });
    });

    it("maps thinkingConfig to reasoning", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          thinkingConfig: { includeThoughts: true, thinkingBudget: 10240 },
        },
      });
      expect(result.reasoning).toEqual({ effort: "high" });
    });

    it("maps json response config to text.format", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          responseMimeType: "application/json",
          responseJsonSchema: { type: "object" },
        },
      });
      expect(result.text?.format).toEqual({
        type: "json_schema",
        name: "response",
        schema: { type: "object" },
      });
    });

    it("converts inlineData to input_image with data URI", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [
          {
            role: "user",
            parts: [
              { text: "What is this?" },
              { inlineData: { mimeType: "image/png", data: "abc123" } },
            ],
          },
        ],
      });

      const input = result.input as any[];
      const content = input[0].content;
      expect(content[1]).toEqual({
        type: "input_image",
        image_url: "data:image/png;base64,abc123",
      });
    });
  });

  // ===== convertResponse (Responses → Gemini, backward) =====

  describe("convertResponse", () => {
    function makeResponse(
      overrides: Partial<OpenAI.Responses.Response> = {}
    ): OpenAI.Responses.Response {
      return {
        id: "resp_123",
        object: "response",
        created_at: 1700000000,
        model: "gemini-2.0-flash",
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

      expect(result.responseId).toBe("resp_123");
      expect(result.modelVersion).toBe("gemini-2.0-flash");

      const parts = result.candidates![0].content!.parts!;
      expect(parts[0].text).toBe("Hello!");
      expect(result.candidates![0].finishReason).toBe("STOP");
    });

    it("converts function_call output to functionCall parts", () => {
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

      const parts = result.candidates![0].content!.parts!;
      expect(parts[0].functionCall).toEqual({
        id: "call_1",
        name: "get_weather",
        args: { city: "SF" },
      });
    });

    it("converts reasoning output to thought parts", () => {
      const result = converter.convertResponse(
        makeResponse({
          output: [
            {
              type: "reasoning",
              id: "rs_1",
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

      const parts = result.candidates![0].content!.parts!;
      expect(parts[0].thought).toBe(true);
      expect(parts[0].text).toBe("Thinking...");
      expect(parts[1].text).toBe("42");
    });

    it("maps incomplete status to MAX_TOKENS", () => {
      const result = converter.convertResponse(
        makeResponse({
          status: "incomplete",
        })
      );

      expect(result.candidates![0].finishReason).toBe("MAX_TOKENS");
    });

    it("maps failed status to SAFETY", () => {
      const result = converter.convertResponse(
        makeResponse({
          status: "failed",
        })
      );

      expect(result.candidates![0].finishReason).toBe("SAFETY");
    });

    it("converts usage", () => {
      const result = converter.convertResponse(
        makeResponse({
          usage: {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            input_tokens_details: { cached_tokens: 30 },
            output_tokens_details: { reasoning_tokens: 20 },
          },
        })
      );

      const usage = result.usageMetadata as any;
      expect(usage.promptTokenCount).toBe(100);
      expect(usage.candidatesTokenCount).toBe(50);
      expect(usage.cachedContentTokenCount).toBe(30);
      expect(usage.thoughtsTokenCount).toBe(20);
    });
  });

  // ===== convertStreamEvent (Responses → Gemini, backward) =====

  describe("convertStreamEvent", () => {
    it("emits chunk on response.created", () => {
      const c = new GeminiToResponsesConverter();
      const chunk = c.convertStreamEvent({
        type: "response.created",
        response: { id: "resp_1", model: "gemini-2.0-flash" } as any,
        sequence_number: 0,
      });

      expect(chunk).not.toBeNull();
      expect(chunk!.responseId).toBe("resp_1");
    });

    it("accumulates text on output_text.delta", () => {
      const c = new GeminiToResponsesConverter();
      c.convertStreamEvent({
        type: "response.created",
        response: { id: "resp_1", model: "gemini-2.0-flash" } as any,
        sequence_number: 0,
      });

      const chunk1 = c.convertStreamEvent({
        type: "response.output_text.delta",
        delta: "Hello",
        item_id: "msg_1",
        output_index: 0,
        content_index: 0,
        sequence_number: 1,
      } as any);

      expect(chunk1!.candidates![0].content!.parts![0].text).toBe("Hello");

      const chunk2 = c.convertStreamEvent({
        type: "response.output_text.delta",
        delta: " world",
        item_id: "msg_1",
        output_index: 0,
        content_index: 0,
        sequence_number: 2,
      } as any);

      expect(chunk2!.candidates![0].content!.parts![0].text).toBe("Hello world");
    });

    it("accumulates reasoning on reasoning_summary_text.delta", () => {
      const c = new GeminiToResponsesConverter();
      c.convertStreamEvent({
        type: "response.created",
        response: { id: "resp_1", model: "gemini-2.0-flash" } as any,
        sequence_number: 0,
      });

      const chunk = c.convertStreamEvent({
        type: "response.reasoning_summary_text.delta",
        delta: "Thinking...",
        item_id: "rs_1",
        output_index: 0,
        summary_index: 0,
        sequence_number: 1,
      } as any);

      const parts = chunk!.candidates![0].content!.parts!;
      expect(parts[0].thought).toBe(true);
      expect(parts[0].text).toBe("Thinking...");
    });

    it("adds function_call on output_item.added", () => {
      const c = new GeminiToResponsesConverter();
      c.convertStreamEvent({
        type: "response.created",
        response: { id: "resp_1", model: "gemini-2.0-flash" } as any,
        sequence_number: 0,
      });

      const chunk = c.convertStreamEvent({
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

      const parts = chunk!.candidates![0].content!.parts!;
      expect(parts[0].functionCall!.name).toBe("get_weather");
    });

    it("emits finishReason on response.completed", () => {
      const c = new GeminiToResponsesConverter();
      c.convertStreamEvent({
        type: "response.created",
        response: { id: "resp_1", model: "gemini-2.0-flash" } as any,
        sequence_number: 0,
      });

      const chunk = c.convertStreamEvent({
        type: "response.completed",
        response: {
          id: "resp_1",
          status: "completed",
          usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
        } as any,
        sequence_number: 10,
      });

      expect(chunk!.candidates![0].finishReason).toBe("STOP");
      expect((chunk!.usageMetadata as any).promptTokenCount).toBe(10);
    });

    it("emits MAX_TOKENS on response.incomplete", () => {
      const c = new GeminiToResponsesConverter();
      c.convertStreamEvent({
        type: "response.created",
        response: { id: "resp_1", model: "gemini-2.0-flash" } as any,
        sequence_number: 0,
      });

      const chunk = c.convertStreamEvent({
        type: "response.incomplete",
        response: { id: "resp_1", status: "incomplete" } as any,
        sequence_number: 10,
      });

      expect(chunk!.candidates![0].finishReason).toBe("MAX_TOKENS");
    });

    it("returns null for unknown events", () => {
      const c = new GeminiToResponsesConverter();
      const result = c.convertStreamEvent({
        type: "response.output_text.done",
      } as any);
      expect(result).toBeNull();
    });
  });

  // ===== convertStream (AsyncIterable) =====

  describe("convertStream", () => {
    it("converts a full stream", async () => {
      const c = new GeminiToResponsesConverter();

      async function* makeStream(): AsyncIterable<OpenAI.Responses.ResponseStreamEvent> {
        yield {
          type: "response.created",
          response: { id: "resp_1", model: "gemini-2.0-flash" } as any,
          sequence_number: 0,
        };
        yield {
          type: "response.output_text.delta",
          delta: "Hello",
          item_id: "msg_1",
          output_index: 0,
          content_index: 0,
          sequence_number: 1,
        } as any;
        yield {
          type: "response.output_text.delta",
          delta: " world",
          item_id: "msg_1",
          output_index: 0,
          content_index: 0,
          sequence_number: 2,
        } as any;
        yield {
          type: "response.completed",
          response: {
            id: "resp_1",
            status: "completed",
            usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
          } as any,
          sequence_number: 3,
        };
      }

      const chunks: GenerateContentResponse[] = [];
      for await (const chunk of c.convertStream(makeStream())) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThanOrEqual(3);

      const lastChunk = chunks[chunks.length - 1];
      expect(lastChunk.candidates![0].finishReason).toBe("STOP");
      expect(lastChunk.candidates![0].content!.parts!.some(p => p.text === "Hello world")).toBe(
        true
      );
    });
  });
});
