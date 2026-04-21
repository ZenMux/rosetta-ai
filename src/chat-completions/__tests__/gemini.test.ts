import type OpenAI from "openai";
import type { GenerateContentResponse, Candidate } from "@google/genai";
import { ChatCompletionToGeminiConverter } from "../gemini";

describe("ChatCompletionToGeminiConverter", () => {
  const converter = new ChatCompletionToGeminiConverter();

  describe("convertRequest", () => {
    it("converts basic user message", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hello" }],
      });

      expect(result.model).toBe("gemini-2.0-flash");
      expect(result.contents as any[]).toEqual([{ role: "user", parts: [{ text: "Hello" }] }]);
    });

    it("converts system message to systemInstruction", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [
          { role: "system", content: "Be helpful." },
          { role: "user", content: "Hi" },
        ],
      });

      expect(result.config?.systemInstruction).toEqual({
        parts: [{ text: "Be helpful." }],
      });
      expect(result.contents as any[]).toEqual([{ role: "user", parts: [{ text: "Hi" }] }]);
    });

    it("converts developer message to systemInstruction", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [
          { role: "developer", content: "You are an assistant." },
          { role: "user", content: "Hi" },
        ],
      });

      expect(result.config?.systemInstruction).toEqual({
        parts: [{ text: "You are an assistant." }],
      });
    });

    it("converts assistant message to model role", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [
          { role: "user", content: "Hello" },
          { role: "assistant", content: "Hi there!" },
          { role: "user", content: "How are you?" },
        ],
      });

      expect((result.contents as any[])[0]).toEqual({
        role: "user",
        parts: [{ text: "Hello" }],
      });
      expect((result.contents as any[])[1]).toEqual({
        role: "model",
        parts: [{ text: "Hi there!" }],
      });
      expect((result.contents as any[])[2]).toEqual({
        role: "user",
        parts: [{ text: "How are you?" }],
      });
    });

    it("converts assistant tool_calls to functionCall parts", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
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

      const modelParts = (result.contents as any[])[1].parts!;
      expect(modelParts[0].functionCall).toEqual({
        id: "call_1",
        name: "get_weather",
        args: { city: "SF" },
      });

      const toolParts = (result.contents as any[])[2].parts!;
      expect(toolParts[0].functionResponse).toBeDefined();
      expect(toolParts[0].functionResponse!.id).toBe("call_1");
      expect(toolParts[0].functionResponse!.response).toEqual({
        output: "72F",
      });
    });

    it("merges consecutive tool responses into one user content", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [
          { role: "user", content: "Do two things" },
          {
            role: "assistant",
            tool_calls: [
              { id: "c1", type: "function", function: { name: "fn1", arguments: "{}" } },
              { id: "c2", type: "function", function: { name: "fn2", arguments: "{}" } },
            ],
          },
          { role: "tool", tool_call_id: "c1", content: "r1" },
          { role: "tool", tool_call_id: "c2", content: "r2" },
        ],
      });

      // Tool responses should be merged into one user content
      expect((result.contents as any[])[2].role).toBe("user");
      expect((result.contents as any[])[2].parts).toHaveLength(2);
      expect((result.contents as any[])[2].parts![0].functionResponse).toBeDefined();
      expect((result.contents as any[])[2].parts![1].functionResponse).toBeDefined();
    });

    it("maps max_completion_tokens to maxOutputTokens", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        max_completion_tokens: 1000,
      });
      expect(result.config?.maxOutputTokens).toBe(1000);
    });

    it("maps temperature and top_p", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        temperature: 0.7,
        top_p: 0.9,
      });
      expect(result.config?.temperature).toBe(0.7);
      expect(result.config?.topP).toBe(0.9);
    });

    it("maps stop to stopSequences", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        stop: ["END", "STOP"],
      });
      expect(result.config?.stopSequences).toEqual(["END", "STOP"]);
    });

    it("maps stop string to stopSequences array", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        stop: "END",
      });
      expect(result.config?.stopSequences).toEqual(["END"]);
    });

    it("converts function tools to functionDeclarations", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              description: "Get weather",
              parameters: { type: "object", properties: { city: { type: "string" } } },
            },
          },
        ],
      });

      const tools = result.config?.tools as any[];
      expect(tools[0].functionDeclarations[0]).toEqual({
        name: "get_weather",
        description: "Get weather",
        parametersJsonSchema: { type: "object", properties: { city: { type: "string" } } },
      });
    });

    it("maps tool_choice auto", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: "auto",
      });
      expect(result.config?.toolConfig?.functionCallingConfig?.mode).toBe("AUTO");
    });

    it("maps tool_choice required to ANY", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: "required",
      });
      expect(result.config?.toolConfig?.functionCallingConfig?.mode).toBe("ANY");
    });

    it("maps tool_choice none", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: "none",
      });
      expect(result.config?.toolConfig?.functionCallingConfig?.mode).toBe("NONE");
    });

    it("maps named tool_choice to ANY with allowedFunctionNames", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: { type: "function", function: { name: "get_weather" } },
      });
      const fcc = result.config?.toolConfig?.functionCallingConfig;
      expect(fcc?.mode).toBe("ANY");
      expect(fcc?.allowedFunctionNames).toEqual(["get_weather"]);
    });

    it("converts json_schema response_format", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "answer",
            schema: { type: "object", properties: { answer: { type: "string" } } },
          },
        },
      });

      expect(result.config?.responseMimeType).toBe("application/json");
      expect(result.config?.responseJsonSchema).toEqual({
        type: "object",
        properties: { answer: { type: "string" } },
      });
    });

    it("converts json_object response_format", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        response_format: { type: "json_object" },
      });

      expect(result.config?.responseMimeType).toBe("application/json");
    });

    it("maps reasoning_effort to thinkingConfig", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        reasoning_effort: "high",
      } as any);

      expect(result.config?.thinkingConfig).toEqual({
        includeThoughts: true,
        thinkingBudget: 10240,
      });
    });

    it("maps reasoning_effort none to disabled", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        reasoning_effort: "none",
      } as any);

      expect(result.config?.thinkingConfig).toEqual({
        thinkingBudget: 0,
      });
    });

    it("converts image_url with base64 data URI to inlineData", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is this?" },
              {
                type: "image_url",
                image_url: { url: "data:image/png;base64,abc123" },
              },
            ],
          },
        ],
      });

      const parts = (result.contents as any[])[0].parts!;
      expect(parts[0]).toEqual({ text: "What is this?" });
      expect(parts[1]).toEqual({
        inlineData: { mimeType: "image/png", data: "abc123" },
      });
    });

    it("converts image_url with URL to fileData", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [
          {
            role: "user",
            content: [
              {
                type: "image_url",
                image_url: { url: "https://example.com/img.png" },
              },
            ],
          },
        ],
      });

      const parts = (result.contents as any[])[0].parts!;
      expect(parts[0]).toEqual({
        fileData: { fileUri: "https://example.com/img.png", mimeType: "image/*" },
      });
    });

    it("maps seed, frequency_penalty, presence_penalty", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        seed: 42,
        frequency_penalty: 0.5,
        presence_penalty: 0.3,
      } as any);

      expect(result.config?.seed).toBe(42);
      expect(result.config?.frequencyPenalty).toBe(0.5);
      expect(result.config?.presencePenalty).toBe(0.3);
    });

    it("maps logprobs and top_logprobs", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        logprobs: true,
        top_logprobs: 5,
      } as any);

      expect(result.config?.responseLogprobs).toBe(true);
      expect(result.config?.logprobs).toBe(5);
    });

    it("maps n to candidateCount", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        n: 3,
      } as any);

      expect(result.config?.candidateCount).toBe(3);
    });

    it("converts web_search_options to googleSearch tool", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Search" }],
        web_search_options: {},
      } as any);

      const tools = result.config?.tools as any[];
      expect(tools.some((t: any) => t.googleSearch != null)).toBe(true);
    });

    it("converts file content part", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [
          {
            role: "user",
            content: [{ type: "file", file: { file_data: "data:application/pdf;base64,abc123" } }],
          },
        ],
      } as any);

      const parts = (result.contents as any[])[0].parts!;
      expect(parts[0]).toEqual({
        inlineData: { mimeType: "application/pdf", data: "abc123" },
      });
    });

    it("converts input_audio content part", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [
          {
            role: "user",
            content: [{ type: "input_audio", input_audio: { format: "mp3", data: "audiodata" } }],
          },
        ],
      } as any);

      const parts = (result.contents as any[])[0].parts!;
      expect(parts[0]).toEqual({
        inlineData: { mimeType: "audio/mp3", data: "audiodata" },
      });
    });

    it("uses parametersJsonSchema for function tools", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "Hi" }],
        tools: [
          {
            type: "function",
            function: {
              name: "fn",
              parameters: { type: "object" },
            },
          },
        ],
      });

      const tools = result.config?.tools as any[];
      const fd = tools[0].functionDeclarations[0];
      expect(fd.parametersJsonSchema).toEqual({ type: "object" });
      expect(fd.parameters).toBeUndefined();
    });
  });

  // ===== convertResponse (Gemini → CC, backward) =====

  describe("convertResponse", () => {
    function makeResponse(
      overrides: Partial<GenerateContentResponse> = {}
    ): GenerateContentResponse {
      return {
        candidates: [
          {
            content: {
              role: "model",
              parts: [{ text: "Hello!" }],
            },
            finishReason: "STOP",
          } as Candidate,
        ],
        usageMetadata: {
          promptTokenCount: 10,
          candidatesTokenCount: 5,
          totalTokenCount: 15,
        },
        responseId: "resp_123",
        modelVersion: "gemini-2.0-flash",
        ...overrides,
      } as GenerateContentResponse;
    }

    it("converts a basic text response", () => {
      const result = converter.convertResponse(makeResponse());

      expect(result.id).toBe("resp_123");
      expect(result.model).toBe("gemini-2.0-flash");
      expect(result.object).toBe("chat.completion");
      expect(result.choices[0].message.content).toBe("Hello!");
      expect(result.choices[0].finish_reason).toBe("stop");
      expect(result.usage?.prompt_tokens).toBe(10);
      expect(result.usage?.completion_tokens).toBe(5);
      expect(result.usage?.total_tokens).toBe(15);
    });

    it("converts functionCall parts to tool_calls", () => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: {
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
              finishReason: "STOP",
            } as any,
          ],
        })
      );

      expect(result.choices[0].message.tool_calls).toEqual([
        {
          id: "call_1",
          type: "function",
          function: { name: "get_weather", arguments: '{"city":"SF"}' },
        },
      ]);
    });

    it("converts thought parts to reasoning", () => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: {
                role: "model",
                parts: [
                  { text: "Let me think...", thought: true, thoughtSignature: "sig123" },
                  { text: "42" },
                ],
              },
              finishReason: "STOP",
            } as any,
          ],
        })
      );

      expect(result.choices[0].message.content).toBe("42");
      expect((result.choices[0].message as any).reasoning).toBe("Let me think...");
      expect((result.choices[0].message as any).reasoning_details[0]).toEqual({
        type: "reasoning.text",
        text: "Let me think...",
        signature: "sig123",
      });
    });

    it("maps MAX_TOKENS finish reason to length", () => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Partial" }] },
              finishReason: "MAX_TOKENS",
            } as any,
          ],
        })
      );

      expect(result.choices[0].finish_reason).toBe("length");
    });

    it("maps SAFETY finish reason to content_filter", () => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "" }] },
              finishReason: "SAFETY",
            } as any,
          ],
        })
      );

      expect(result.choices[0].finish_reason).toBe("content_filter");
    });

    it("converts usage with cached and thoughts tokens", () => {
      const result = converter.convertResponse(
        makeResponse({
          usageMetadata: {
            promptTokenCount: 100,
            candidatesTokenCount: 50,
            totalTokenCount: 150,
            cachedContentTokenCount: 30,
            thoughtsTokenCount: 20,
          } as any,
        })
      );

      expect(result.usage?.prompt_tokens).toBe(100);
      expect(result.usage?.completion_tokens).toBe(70);
      expect(result.usage?.prompt_tokens_details?.cached_tokens).toBe(30);
      expect(result.usage?.completion_tokens_details?.reasoning_tokens).toBe(20);
    });

    it("includes toolUsePromptTokenCount in prompt_tokens", () => {
      const result = converter.convertResponse(
        makeResponse({
          usageMetadata: {
            promptTokenCount: 100,
            candidatesTokenCount: 50,
            totalTokenCount: 165,
            toolUsePromptTokenCount: 15,
          } as any,
        })
      );

      expect(result.usage?.prompt_tokens).toBe(115);
    });

    it("overrides finish_reason to tool_calls when function calls present", () => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: {
                role: "model",
                parts: [{ functionCall: { id: "c1", name: "fn", args: {} } }],
              },
              finishReason: "STOP",
            } as any,
          ],
        })
      );

      expect(result.choices[0].finish_reason).toBe("tool_calls");
    });

    it("extracts grounding metadata as annotations", () => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Result" }] },
              finishReason: "STOP",
              groundingMetadata: {
                groundingChunks: [{ web: { title: "Example", uri: "https://example.com" } }],
              },
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
            end_index: 0,
          },
        },
      ]);
    });

    it("handles empty candidates", () => {
      const result = converter.convertResponse(makeResponse({ candidates: [] }));

      expect(result.choices[0].message.content).toBeNull();
      expect(result.choices[0].finish_reason).toBe("stop");
    });
  });

  // ===== convertStreamChunk (Gemini → CC, backward) =====

  describe("convertStreamChunk", () => {
    function makeStreamChunk(
      overrides: Partial<GenerateContentResponse> = {}
    ): GenerateContentResponse {
      return {
        candidates: [
          {
            content: { role: "model", parts: [] },
          } as any,
        ],
        responseId: "resp_1",
        modelVersion: "gemini-2.0-flash",
        ...overrides,
      } as GenerateContentResponse;
    }

    it("emits role chunk on first chunk", () => {
      const c = new ChatCompletionToGeminiConverter();
      const events = c.convertStreamChunk(makeStreamChunk());

      expect(events.length).toBeGreaterThanOrEqual(1);
      expect(events[0].choices[0].delta.role).toBe("assistant");
    });

    it("emits text delta for new text", () => {
      const c = new ChatCompletionToGeminiConverter();
      c.convertStreamChunk(makeStreamChunk());

      const events = c.convertStreamChunk(
        makeStreamChunk({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Hello" }] },
            } as any,
          ],
        })
      );

      const textChunk = events.find(e => e.choices[0]?.delta?.content === "Hello");
      expect(textChunk).toBeDefined();
    });

    it("emits incremental text delta", () => {
      const c = new ChatCompletionToGeminiConverter();
      c.convertStreamChunk(makeStreamChunk());

      c.convertStreamChunk(
        makeStreamChunk({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Hello" }] },
            } as any,
          ],
        })
      );

      const events = c.convertStreamChunk(
        makeStreamChunk({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Hello world" }] },
            } as any,
          ],
        })
      );

      const textChunk = events.find(e => e.choices[0]?.delta?.content === " world");
      expect(textChunk).toBeDefined();
    });

    it("emits tool_calls for function call", () => {
      const c = new ChatCompletionToGeminiConverter();
      c.convertStreamChunk(makeStreamChunk());

      const events = c.convertStreamChunk(
        makeStreamChunk({
          candidates: [
            {
              content: {
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
            } as any,
          ],
        })
      );

      const toolChunk = events.find(
        e => e.choices[0]?.delta?.tool_calls?.[0]?.function?.name === "get_weather"
      );
      expect(toolChunk).toBeDefined();
    });

    it("emits reasoning for thought parts", () => {
      const c = new ChatCompletionToGeminiConverter();
      c.convertStreamChunk(makeStreamChunk());

      const events = c.convertStreamChunk(
        makeStreamChunk({
          candidates: [
            {
              content: {
                role: "model",
                parts: [{ text: "Thinking...", thought: true }],
              },
            } as any,
          ],
        })
      );

      const reasoningChunk = events.find(
        e => (e.choices[0]?.delta as any)?.reasoning === "Thinking..."
      );
      expect(reasoningChunk).toBeDefined();
    });

    it("emits finish_reason on final chunk", () => {
      const c = new ChatCompletionToGeminiConverter();
      c.convertStreamChunk(makeStreamChunk());
      c.convertStreamChunk(
        makeStreamChunk({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Hi" }] },
            } as any,
          ],
        })
      );

      const events = c.convertStreamChunk(
        makeStreamChunk({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Hi" }] },
              finishReason: "STOP",
            } as any,
          ],
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 5,
            totalTokenCount: 15,
          } as any,
        })
      );

      const finishChunk = events.find(e => e.choices[0]?.finish_reason === "stop");
      expect(finishChunk).toBeDefined();
      expect(finishChunk!.usage?.prompt_tokens).toBe(10);
      expect(finishChunk!.usage?.completion_tokens).toBe(5);
    });
  });

  // ===== convertStream (AsyncIterable) =====

  describe("convertStream", () => {
    it("converts a full stream", async () => {
      const c = new ChatCompletionToGeminiConverter();

      async function* makeStream(): AsyncIterable<GenerateContentResponse> {
        yield {
          candidates: [{ content: { role: "model", parts: [] } } as any],
          responseId: "resp_1",
          modelVersion: "gemini-2.0-flash",
        } as GenerateContentResponse;
        yield {
          candidates: [{ content: { role: "model", parts: [{ text: "Hello" }] } } as any],
        } as GenerateContentResponse;
        yield {
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Hello world" }] },
              finishReason: "STOP",
            } as any,
          ],
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 5,
            totalTokenCount: 15,
          },
        } as GenerateContentResponse;
      }

      const chunks: OpenAI.ChatCompletionChunk[] = [];
      for await (const chunk of c.convertStream(makeStream())) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThanOrEqual(3);
      expect(chunks[0].choices[0].delta.role).toBe("assistant");

      const textChunks = chunks.filter(c => c.choices[0]?.delta?.content);
      expect(textChunks.length).toBeGreaterThanOrEqual(1);

      const finishChunk = chunks.find(c => c.choices[0]?.finish_reason === "stop");
      expect(finishChunk).toBeDefined();
    });
  });
});
