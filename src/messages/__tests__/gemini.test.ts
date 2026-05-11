import type Anthropic from "@anthropic-ai/sdk";
import type { GenerateContentResponse, Candidate } from "@google/genai";
import { MessagesToGeminiConverter } from "../gemini";

describe("MessagesToGeminiConverter", () => {
  const converter = new MessagesToGeminiConverter();

  describe("convertRequest", () => {
    it("converts basic user message", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
      });

      expect(result.model).toBe("gemini-2.0-flash");
      expect(result.contents as any[]).toEqual([{ role: "user", parts: [{ text: "Hello" }] }]);
    });

    it("converts string system to systemInstruction", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        system: "Be helpful.",
        messages: [{ role: "user", content: "Hi" }],
      });

      expect(result.config?.systemInstruction).toEqual({
        parts: [{ text: "Be helpful." }],
      });
    });

    it("converts TextBlockParam[] system to systemInstruction", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        system: [
          { type: "text", text: "Rule 1." },
          { type: "text", text: "Rule 2." },
        ],
        messages: [{ role: "user", content: "Hi" }],
      });

      expect(result.config?.systemInstruction).toEqual({
        parts: [{ text: "Rule 1." }, { text: "Rule 2." }],
      });
    });

    it("maps max_tokens to maxOutputTokens", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 2000,
        messages: [{ role: "user", content: "Hi" }],
      });
      expect(result.config?.maxOutputTokens).toBe(2000);
    });

    it("maps temperature, top_p, stop_sequences", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        temperature: 0.7,
        top_p: 0.9,
        stop_sequences: ["END"],
      });
      expect(result.config?.temperature).toBe(0.7);
      expect(result.config?.topP).toBe(0.9);
      expect(result.config?.stopSequences).toEqual(["END"]);
    });

    it("maps top_k to config.topK", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        top_k: 40,
      });
      expect(result.config?.topK).toBe(40);
    });

    it("converts assistant message to model role", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [
          { role: "user", content: "Hello" },
          { role: "assistant", content: "Hi there!" },
          { role: "user", content: "How are you?" },
        ],
      });

      expect((result.contents as any[])[1]).toEqual({
        role: "model",
        parts: [{ text: "Hi there!" }],
      });
    });

    it("converts tool_use blocks in assistant to functionCall parts", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [
          { role: "user", content: "Weather?" },
          {
            role: "assistant",
            content: [
              {
                type: "tool_use",
                id: "tu_1",
                name: "get_weather",
                input: { city: "SF" },
              },
            ],
          },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "tu_1",
                content: "72F",
              },
            ],
          },
        ],
      });

      const modelParts = (result.contents as any[])[1].parts;
      expect(modelParts[0].functionCall).toEqual({
        id: "tu_1",
        name: "get_weather",
        args: { city: "SF" },
      });

      const toolParts = (result.contents as any[])[2].parts;
      expect(toolParts[0].functionResponse).toBeDefined();
      expect(toolParts[0].functionResponse.id).toBe("tu_1");
      expect(toolParts[0].functionResponse.response).toEqual({ output: "72F" });
    });

    it("converts image block (base64) to inlineData", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is this?" },
              {
                type: "image",
                source: {
                  type: "base64",
                  media_type: "image/png",
                  data: "abc123",
                },
              },
            ],
          },
        ],
      });

      const parts = (result.contents as any[])[0].parts;
      expect(parts[0]).toEqual({ text: "What is this?" });
      expect(parts[1]).toEqual({
        inlineData: { mimeType: "image/png", data: "abc123" },
      });
    });

    it("converts image block (url) to fileData", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "image",
                source: {
                  type: "url",
                  url: "https://example.com/img.png",
                },
              },
            ],
          },
        ],
      });

      const parts = (result.contents as any[])[0].parts;
      expect(parts[0]).toEqual({
        fileData: { fileUri: "https://example.com/img.png", mimeType: "image/*" },
      });
    });

    it("converts document block (base64) to inlineData", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "document",
                source: {
                  type: "base64",
                  media_type: "application/pdf",
                  data: "pdfdata",
                },
              },
            ],
          },
        ],
      });

      const parts = (result.contents as any[])[0].parts;
      expect(parts[0]).toEqual({
        inlineData: { mimeType: "application/pdf", data: "pdfdata" },
      });
    });

    it("converts tools with input_schema to functionDeclarations", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        tools: [
          {
            name: "get_weather",
            description: "Get weather",
            input_schema: {
              type: "object" as const,
              properties: { city: { type: "string" } },
            },
          },
        ],
      });

      const tools = result.config?.tools as any[];
      expect(tools[0].functionDeclarations[0]).toEqual({
        name: "get_weather",
        description: "Get weather",
        parametersJsonSchema: {
          type: "object",
          properties: { city: { type: "string" } },
        },
      });
    });

    it("converts web_search tool to googleSearch", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Search" }],
        tools: [{ type: "web_search_20250305" } as any],
      });

      const tools = result.config?.tools as any[];
      expect(tools.some((t: any) => t.googleSearch != null)).toBe(true);
    });

    it("maps tool_choice", () => {
      const auto = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: { type: "auto" },
      });
      expect(auto.config?.toolConfig?.functionCallingConfig?.mode).toBe("AUTO");

      const any_ = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: { type: "any" },
      });
      expect(any_.config?.toolConfig?.functionCallingConfig?.mode).toBe("ANY");

      const none = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: { type: "none" },
      });
      expect(none.config?.toolConfig?.functionCallingConfig?.mode).toBe("NONE");
    });

    it("maps named tool_choice to ANY with allowedFunctionNames", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: { type: "tool", name: "get_weather" },
      });
      const fcc = result.config?.toolConfig?.functionCallingConfig;
      expect(fcc?.mode).toBe("ANY");
      expect(fcc?.allowedFunctionNames).toEqual(["get_weather"]);
    });

    it("converts thinking enabled to thinkingConfig", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        thinking: { type: "enabled", budget_tokens: 8192 },
      });

      expect(result.config?.thinkingConfig).toEqual({
        includeThoughts: true,
        thinkingBudget: 8192,
      });
    });

    it("converts thinking disabled to thinkingBudget 0", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        thinking: { type: "disabled" },
      });

      expect(result.config?.thinkingConfig).toEqual({
        thinkingBudget: 0,
      });
    });

    it("converts output_config json_schema", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hi" }],
        output_config: {
          format: {
            type: "json_schema",
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
  });

  // ===== convertResponse (Gemini → Messages, backward) =====

  describe("convertResponse", () => {
    function makeResponse(
      overrides: Partial<GenerateContentResponse> = {}
    ): GenerateContentResponse {
      return {
        candidates: [
          {
            content: { role: "model", parts: [{ text: "Hello!" }] },
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
      expect(result.type).toBe("message");
      expect(result.role).toBe("assistant");
      expect(result.stop_reason).toBe("end_turn");

      const textBlock = result.content.find((b: any) => b.type === "text") as any;
      expect(textBlock.text).toBe("Hello!");

      expect(result.usage.input_tokens).toBe(10);
      expect(result.usage.output_tokens).toBe(5);
    });

    it("converts functionCall parts to tool_use blocks", () => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: {
                role: "model",
                parts: [
                  { functionCall: { id: "call_1", name: "get_weather", args: { city: "SF" } } },
                ],
              },
              finishReason: "STOP",
            } as any,
          ],
        })
      );

      const toolUse = result.content.find((b: any) => b.type === "tool_use") as any;
      expect(toolUse).toBeDefined();
      expect(toolUse.name).toBe("get_weather");
      expect(toolUse.input).toEqual({ city: "SF" });
      expect(result.stop_reason).toBe("tool_use");
    });

    it("converts thought parts to thinking blocks", () => {
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

      const thinkingBlock = result.content.find((b: any) => b.type === "thinking") as any;
      expect(thinkingBlock).toBeDefined();
      expect(thinkingBlock.thinking).toBe("Let me think...");
      expect(thinkingBlock.signature).toBe("sig123");

      const textBlock = result.content.find((b: any) => b.type === "text") as any;
      expect(textBlock.text).toBe("42");
    });

    it("maps MAX_TOKENS to max_tokens stop_reason", () => {
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
      expect(result.stop_reason).toBe("max_tokens");
    });

    it("maps SAFETY to refusal stop_reason", () => {
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
      expect(result.stop_reason).toBe("refusal");
    });

    it("extracts grounding metadata as web_search_tool_result", () => {
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

      const wsBlock = result.content.find((b: any) => b.type === "web_search_tool_result") as any;
      expect(wsBlock).toBeDefined();
      expect(wsBlock.content[0].url).toBe("https://example.com");
    });

    it("includes toolUsePromptTokenCount and thoughtsTokenCount in usage", () => {
      const result = converter.convertResponse(
        makeResponse({
          usageMetadata: {
            promptTokenCount: 100,
            candidatesTokenCount: 50,
            totalTokenCount: 180,
            toolUsePromptTokenCount: 10,
            thoughtsTokenCount: 20,
          } as any,
        })
      );

      expect(result.usage.input_tokens).toBe(110);
      expect(result.usage.output_tokens).toBe(70);
    });

    it("enriches usage with OpenAI-style fields", () => {
      const result = converter.convertResponse(
        makeResponse({
          usageMetadata: {
            promptTokenCount: 100,
            candidatesTokenCount: 50,
            totalTokenCount: 180,
            toolUsePromptTokenCount: 10,
            thoughtsTokenCount: 20,
            cachedContentTokenCount: 30,
          } as any,
        })
      );

      const usage = result.usage as any;
      expect(usage.prompt_tokens).toBe(110);
      expect(usage.completion_tokens).toBe(70);
      expect(usage.total_tokens).toBe(180);
      expect(usage.prompt_tokens_details.cached_tokens).toBe(30);
      expect(usage.completion_tokens_details.reasoning_tokens).toBe(20);
      expect(usage.tool_use).toBe(10);
    });

    it("counts web_search and web_search_queries from groundingMetadata", () => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Result" }] },
              finishReason: "STOP",
              groundingMetadata: {
                webSearchQueries: ["query1", "query2", "query3"],
              },
            } as any,
          ],
        })
      );

      const usage = result.usage as any;
      expect(usage.web_search).toBe(1);
      expect(usage.web_search_queries).toBe(3);
    });

    it("extracts audio tokens from promptTokensDetails and cacheTokensDetails", () => {
      const result = converter.convertResponse(
        makeResponse({
          usageMetadata: {
            promptTokenCount: 50,
            candidatesTokenCount: 20,
            totalTokenCount: 70,
            promptTokensDetails: [
              { modality: "TEXT", tokenCount: 30 },
              { modality: "AUDIO", tokenCount: 20 },
            ],
            cacheTokensDetails: [{ modality: "AUDIO", tokenCount: 5 }],
          } as any,
        })
      );

      const usage = result.usage as any;
      expect(usage.prompt_tokens_details.audio_tokens).toBe(20);
      expect(usage.prompt_tokens_details.audio_cached_tokens).toBe(5);
      expect(usage.audio_input_tokens).toBe(20);
      expect(usage.audio_cache_read_tokens).toBe(5);
    });

    it("passes trafficType through", () => {
      const result = converter.convertResponse(
        makeResponse({
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 5,
            totalTokenCount: 15,
            trafficType: "ON_DEMAND",
          } as any,
        })
      );

      const usage = result.usage as any;
      expect(usage.trafficType).toBe("ON_DEMAND");
    });
  });

  // ===== convertStreamChunk (Gemini → Messages, backward) =====

  describe("convertStreamChunk", () => {
    function makeChunk(overrides: Partial<GenerateContentResponse> = {}): GenerateContentResponse {
      return {
        candidates: [{ content: { role: "model", parts: [] } } as any],
        responseId: "resp_1",
        modelVersion: "gemini-2.0-flash",
        ...overrides,
      } as GenerateContentResponse;
    }

    it("emits message_start on first chunk", () => {
      const c = new MessagesToGeminiConverter();
      const events = c.convertStreamChunk(makeChunk());

      const msgStart = events.find(e => e.type === "message_start") as any;
      expect(msgStart).toBeDefined();
      expect(msgStart.message.role).toBe("assistant");
    });

    it("emits text content_block_start and text_delta", () => {
      const c = new MessagesToGeminiConverter();
      c.convertStreamChunk(makeChunk());

      const events = c.convertStreamChunk(
        makeChunk({
          candidates: [{ content: { role: "model", parts: [{ text: "Hello" }] } } as any],
        })
      );

      const blockStart = events.find(e => e.type === "content_block_start") as any;
      expect(blockStart).toBeDefined();
      expect(blockStart.content_block.type).toBe("text");

      const delta = events.find(e => e.type === "content_block_delta") as any;
      expect(delta).toBeDefined();
      expect(delta.delta.type).toBe("text_delta");
      expect(delta.delta.text).toBe("Hello");
    });

    it("emits each incremental text chunk as delta", () => {
      const c = new MessagesToGeminiConverter();
      c.convertStreamChunk(makeChunk());

      c.convertStreamChunk(
        makeChunk({
          candidates: [{ content: { role: "model", parts: [{ text: "Hello" }] } } as any],
        })
      );

      const events = c.convertStreamChunk(
        makeChunk({
          candidates: [{ content: { role: "model", parts: [{ text: " world" }] } } as any],
        })
      );

      const delta = events.find(e => e.type === "content_block_delta") as any;
      expect(delta.delta.text).toBe(" world");
    });

    it("emits tool_use content_block_start for function call", () => {
      const c = new MessagesToGeminiConverter();
      c.convertStreamChunk(makeChunk());

      const events = c.convertStreamChunk(
        makeChunk({
          candidates: [
            {
              content: {
                role: "model",
                parts: [
                  { functionCall: { id: "call_1", name: "get_weather", args: { city: "SF" } } },
                ],
              },
            } as any,
          ],
        })
      );

      const blockStart = events.find(e => e.type === "content_block_start") as any;
      expect(blockStart).toBeDefined();
      expect(blockStart.content_block.type).toBe("tool_use");
      expect(blockStart.content_block.name).toBe("get_weather");
    });

    it("emits thinking_delta for thought parts", () => {
      const c = new MessagesToGeminiConverter();
      c.convertStreamChunk(makeChunk());

      const events = c.convertStreamChunk(
        makeChunk({
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

      const blockStart = events.find(e => e.type === "content_block_start") as any;
      expect(blockStart.content_block.type).toBe("thinking");

      const delta = events.find(e => e.type === "content_block_delta") as any;
      expect(delta.delta.type).toBe("thinking_delta");
      expect(delta.delta.thinking).toBe("Thinking...");
    });

    it("emits message_delta and message_stop on finish", () => {
      const c = new MessagesToGeminiConverter();
      c.convertStreamChunk(makeChunk());
      c.convertStreamChunk(
        makeChunk({
          candidates: [{ content: { role: "model", parts: [{ text: "Hi" }] } } as any],
        })
      );

      const events = c.convertStreamChunk(
        makeChunk({
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

      const msgDelta = events.find(e => e.type === "message_delta") as any;
      expect(msgDelta).toBeDefined();
      expect(msgDelta.delta.stop_reason).toBe("end_turn");

      const msgStop = events.find(e => e.type === "message_stop");
      expect(msgStop).toBeDefined();
    });

    it("emits tool_use stop_reason when function calls present", () => {
      const c = new MessagesToGeminiConverter();
      c.convertStreamChunk(makeChunk());
      c.convertStreamChunk(
        makeChunk({
          candidates: [
            {
              content: {
                role: "model",
                parts: [{ functionCall: { id: "c1", name: "fn", args: {} } }],
              },
            } as any,
          ],
        })
      );

      const events = c.convertStreamChunk(
        makeChunk({
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

      const msgDelta = events.find(e => e.type === "message_delta") as any;
      expect(msgDelta.delta.stop_reason).toBe("tool_use");
    });
  });

  // ===== convertStream (AsyncIterable) =====

  describe("convertStream", () => {
    it("converts a full stream", async () => {
      const c = new MessagesToGeminiConverter();

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

      const events: Anthropic.RawMessageStreamEvent[] = [];
      for await (const event of c.convertStream(makeStream())) {
        events.push(event);
      }

      const types = events.map(e => e.type);
      expect(types).toContain("message_start");
      expect(types).toContain("content_block_start");
      expect(types).toContain("content_block_delta");
      expect(types).toContain("message_delta");
      expect(types).toContain("message_stop");
    });
  });
});
