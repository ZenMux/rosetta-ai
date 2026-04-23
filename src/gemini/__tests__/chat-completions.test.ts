import type OpenAI from "openai";
import type { GenerateContentResponse, Candidate } from "@google/genai";
import { GeminiToChatCompletionConverter } from "../chat-completions";

describe("GeminiToChatCompletionConverter", () => {
  const converter = new GeminiToChatCompletionConverter();

  describe("convertRequest", () => {
    it("converts basic user content to message", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
      });

      expect(result.model).toBe("gemini-2.0-flash");
      expect(result.messages).toEqual([{ role: "user", content: "Hello" }]);
    });

    it("converts string contents", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: "Hello" as any,
      });

      expect(result.messages).toEqual([{ role: "user", content: "Hello" }]);
    });

    it("converts systemInstruction to system message", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          systemInstruction: { parts: [{ text: "Be helpful." }] },
        },
      });

      expect(result.messages[0]).toEqual({
        role: "system",
        content: "Be helpful.",
      });
    });

    it("maps maxOutputTokens to max_completion_tokens", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: { maxOutputTokens: 2000 },
      });
      expect(result.max_completion_tokens).toBe(2000);
    });

    it("maps temperature, topP, stopSequences", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          temperature: 0.7,
          topP: 0.9,
          stopSequences: ["END"],
        },
      });
      expect(result.temperature).toBe(0.7);
      expect(result.top_p).toBe(0.9);
      expect(result.stop).toEqual(["END"]);
    });

    it("converts model content to assistant message", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [
          { role: "user", parts: [{ text: "Hello" }] },
          { role: "model", parts: [{ text: "Hi there!" }] },
        ],
      });

      expect(result.messages[1]).toEqual({
        role: "assistant",
        content: "Hi there!",
      });
    });

    it("converts functionCall parts to tool_calls", () => {
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

      const assistantMsg = result.messages[1] as any;
      expect(assistantMsg.tool_calls[0]).toEqual({
        id: "call_1",
        type: "function",
        function: { name: "get_weather", arguments: '{"city":"SF"}' },
      });

      expect(result.messages[2]).toEqual({
        role: "tool",
        tool_call_id: "call_1",
        content: '{"output":"72F"}',
      });
    });

    it("converts functionDeclarations to tools", () => {
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

      expect(result.tools![0]).toEqual({
        type: "function",
        function: {
          name: "get_weather",
          description: "Get weather",
          parameters: { type: "object", properties: { city: { type: "string" } } },
        },
      });
    });

    it("converts googleSearch to web_search_options", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Search" }] }],
        config: { tools: [{ googleSearch: {} }] },
      });

      expect((result as any).web_search_options).toBeDefined();
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
        function: { name: "get_weather" },
      });
    });

    it("maps json response config to response_format", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          responseMimeType: "application/json",
          responseJsonSchema: { type: "object" },
        },
      });
      expect(result.response_format).toEqual({
        type: "json_schema",
        json_schema: { name: "response", schema: { type: "object" } },
      });
    });

    it("maps thinkingConfig to reasoning_effort", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          thinkingConfig: { includeThoughts: true, thinkingBudget: 10240 },
        },
      });
      expect(result.reasoning_effort).toBe("high");
    });

    it("converts inlineData to image_url", () => {
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

      const content = result.messages[0] as any;
      expect(content.content[1]).toEqual({
        type: "image_url",
        image_url: { url: "data:image/png;base64,abc123" },
      });
    });

    it("maps seed, frequencyPenalty, presencePenalty", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          seed: 42,
          frequencyPenalty: 0.5,
          presencePenalty: 0.3,
        },
      });
      expect((result as any).seed).toBe(42);
      expect(result.frequency_penalty).toBe(0.5);
      expect(result.presence_penalty).toBe(0.3);
    });
  });

  // ===== convertResponse (CC → Gemini, backward) =====

  describe("convertResponse", () => {
    function makeCCResponse(overrides: Partial<OpenAI.ChatCompletion> = {}): OpenAI.ChatCompletion {
      return {
        id: "chatcmpl-123",
        object: "chat.completion",
        created: 1700000000,
        model: "gemini-2.0-flash",
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

      expect(result.responseId).toBe("chatcmpl-123");
      expect(result.modelVersion).toBe("gemini-2.0-flash");

      const parts = result.candidates![0].content!.parts!;
      expect(parts[0].text).toBe("Hello!");
      expect(result.candidates![0].finishReason).toBe("STOP");
    });

    it("converts tool_calls to functionCall parts", () => {
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

      const parts = result.candidates![0].content!.parts!;
      expect(parts[0].functionCall).toEqual({
        id: "call_1",
        name: "get_weather",
        args: { city: "SF" },
      });
    });

    it("converts reasoning to thought parts", () => {
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

      const parts = result.candidates![0].content!.parts!;
      expect(parts[0].thought).toBe(true);
      expect(parts[0].text).toBe("Let me think...");
      expect(parts[1].text).toBe("42");
    });

    it("maps length finish_reason to MAX_TOKENS", () => {
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

      expect(result.candidates![0].finishReason).toBe("MAX_TOKENS");
    });

    it("maps content_filter to SAFETY", () => {
      const result = converter.convertResponse(
        makeCCResponse({
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "", refusal: null },
              finish_reason: "content_filter",
              logprobs: null,
            },
          ],
        })
      );

      expect(result.candidates![0].finishReason).toBe("SAFETY");
    });

    it("converts usage", () => {
      const result = converter.convertResponse(
        makeCCResponse({
          usage: {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: { cached_tokens: 30 },
            completion_tokens_details: { reasoning_tokens: 20 },
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

  // ===== convertStreamChunk (CC → Gemini, backward) =====

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
        model: "gemini-2.0-flash",
        choices: [{ index: 0, delta, finish_reason }],
        ...rest,
      };
    }

    it("emits initial chunk on first call", () => {
      const c = new GeminiToChatCompletionConverter();
      const result = c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

      expect(result).not.toBeNull();
      expect(result!.candidates![0].content!.parts).toEqual([]);
    });

    it("accumulates text content", () => {
      const c = new GeminiToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

      const r1 = c.convertStreamChunk(makeChunk({ delta: { content: "Hello" } }));
      expect(r1!.candidates![0].content!.parts!.some(p => p.text === "Hello")).toBe(true);

      const r2 = c.convertStreamChunk(makeChunk({ delta: { content: " world" } }));
      expect(r2!.candidates![0].content!.parts!.some(p => p.text === "Hello world")).toBe(true);
    });

    it("handles tool call deltas", () => {
      const c = new GeminiToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

      const result = c.convertStreamChunk(
        makeChunk({
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "call_1",
                type: "function",
                function: { name: "get_weather", arguments: '{"city":"SF"}' },
              },
            ],
          },
        })
      );

      const parts = result!.candidates![0].content!.parts!;
      expect(parts.some(p => p.functionCall?.name === "get_weather")).toBe(true);
    });

    it("handles reasoning deltas", () => {
      const c = new GeminiToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));

      const result = c.convertStreamChunk(
        makeChunk({
          delta: { content: "", reasoning: "Thinking..." } as any,
        })
      );

      const parts = result!.candidates![0].content!.parts!;
      expect(parts.some(p => p.thought === true && p.text === "Thinking...")).toBe(true);
    });

    it("emits finishReason on finish", () => {
      const c = new GeminiToChatCompletionConverter();
      c.convertStreamChunk(makeChunk({ delta: { role: "assistant" } }));
      c.convertStreamChunk(makeChunk({ delta: { content: "Hi" } }));

      const result = c.convertStreamChunk(
        makeChunk({
          finish_reason: "stop",
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
        } as any)
      );

      expect(result!.candidates![0].finishReason).toBe("STOP");
      expect((result!.usageMetadata as any).promptTokenCount).toBe(10);
    });
  });

  // ===== convertStream (AsyncIterable) =====

  describe("convertStream", () => {
    it("converts a full stream", async () => {
      const c = new GeminiToChatCompletionConverter();

      async function* makeStream(): AsyncIterable<OpenAI.ChatCompletionChunk> {
        yield {
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
          created: 1700000000,
          model: "gemini-2.0-flash",
          choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
        };
        yield {
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
          created: 1700000000,
          model: "gemini-2.0-flash",
          choices: [{ index: 0, delta: { content: "Hello" }, finish_reason: null }],
        };
        yield {
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
          created: 1700000000,
          model: "gemini-2.0-flash",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
        } as OpenAI.ChatCompletionChunk;
      }

      const chunks: GenerateContentResponse[] = [];
      for await (const chunk of c.convertStream(makeStream())) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThanOrEqual(2);

      const last = chunks[chunks.length - 1];
      expect(last.candidates![0].finishReason).toBe("STOP");
    });
  });
});
