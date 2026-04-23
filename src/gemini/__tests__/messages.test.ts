import type Anthropic from "@anthropic-ai/sdk";
import type { GenerateContentResponse, Candidate } from "@google/genai";
import { GeminiToMessagesConverter } from "../messages";

describe("GeminiToMessagesConverter", () => {
  const converter = new GeminiToMessagesConverter();

  describe("convertRequest", () => {
    it("converts basic user content", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
      });

      expect(result.model).toBe("gemini-2.0-flash");
      expect(result.messages[0]).toEqual({
        role: "user",
        content: [{ type: "text", text: "Hello" }],
      });
    });

    it("converts string contents", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: "Hello" as any,
      });
      expect(result.messages[0]).toEqual({
        role: "user",
        content: [{ type: "text", text: "Hello" }],
      });
    });

    it("converts systemInstruction to system", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: { systemInstruction: { parts: [{ text: "Be helpful." }] } },
      });
      expect(result.system).toBe("Be helpful.");
    });

    it("maps maxOutputTokens to max_tokens", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: { maxOutputTokens: 2000 },
      });
      expect(result.max_tokens).toBe(2000);
    });

    it("uses default max_tokens when not specified", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
      });
      expect(result.max_tokens).toBe(4096);
    });

    it("maps temperature, topP, stopSequences", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: { temperature: 0.7, topP: 0.9, stopSequences: ["END"] },
      });
      expect(result.temperature).toBe(0.7);
      expect(result.top_p).toBe(0.9);
      expect(result.stop_sequences).toEqual(["END"]);
    });

    it("converts model content to assistant message", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [
          { role: "user", parts: [{ text: "Hello" }] },
          { role: "model", parts: [{ text: "Hi!" }] },
        ],
      });
      expect(result.messages[1]).toEqual({
        role: "assistant",
        content: [{ type: "text", text: "Hi!" }],
      });
    });

    it("converts functionCall parts to tool_use blocks", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [
          { role: "user", parts: [{ text: "Weather?" }] },
          {
            role: "model",
            parts: [{ functionCall: { id: "tu_1", name: "get_weather", args: { city: "SF" } } }],
          },
          {
            role: "user",
            parts: [
              {
                functionResponse: { id: "tu_1", name: "get_weather", response: { output: "72F" } },
              },
            ],
          },
        ],
      });

      const assistantContent = result.messages[1].content as any[];
      expect(assistantContent[0]).toEqual({
        type: "tool_use",
        id: "tu_1",
        name: "get_weather",
        input: { city: "SF" },
      });

      const userContent = result.messages[2].content as any[];
      expect(userContent[0].type).toBe("tool_result");
      expect(userContent[0].tool_use_id).toBe("tu_1");
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
        name: "get_weather",
        description: "Get weather",
        input_schema: { type: "object", properties: { city: { type: "string" } } },
      });
    });

    it("converts googleSearch to web_search tool", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Search" }] }],
        config: { tools: [{ googleSearch: {} }] },
      });
      expect(result.tools!.some((t: any) => t.type === "web_search_20250305")).toBe(true);
    });

    it("maps toolConfig to tool_choice", () => {
      const auto = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: { toolConfig: { functionCallingConfig: { mode: "AUTO" as any } } },
      });
      expect(auto.tool_choice).toEqual({ type: "auto" });

      const any_ = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: { toolConfig: { functionCallingConfig: { mode: "ANY" as any } } },
      });
      expect(any_.tool_choice).toEqual({ type: "any" });

      const named = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          toolConfig: {
            functionCallingConfig: { mode: "ANY" as any, allowedFunctionNames: ["fn"] },
          },
        },
      });
      expect(named.tool_choice).toEqual({ type: "tool", name: "fn" });
    });

    it("converts thinkingConfig to thinking", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: { thinkingConfig: { includeThoughts: true, thinkingBudget: 8192 } },
      });
      expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 8192 });
    });

    it("converts thinkingConfig disabled", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: { thinkingConfig: { thinkingBudget: 0 } },
      });
      expect(result.thinking).toEqual({ type: "disabled" });
    });

    it("converts json response config to output_config", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        config: {
          responseMimeType: "application/json",
          responseJsonSchema: { type: "object" },
        },
      });
      expect(result.output_config).toEqual({
        format: { type: "json_schema", schema: { type: "object" } },
      });
    });

    it("converts inlineData to image block", () => {
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

      const content = result.messages[0].content as any[];
      expect(content[1]).toEqual({
        type: "image",
        source: { type: "base64", media_type: "image/png", data: "abc123" },
      });
    });
  });

  // ===== convertResponse (Messages → Gemini, backward) =====

  describe("convertResponse", () => {
    function makeMessage(overrides: Partial<Anthropic.Message> = {}): Anthropic.Message {
      return {
        id: "msg_123",
        type: "message",
        role: "assistant",
        model: "gemini-2.0-flash" as Anthropic.Model,
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

      expect(result.responseId).toBe("msg_123");
      expect(result.modelVersion).toBe("gemini-2.0-flash");
      expect(result.candidates![0].content!.parts!.some(p => p.text === "Hello!")).toBe(true);
      expect(result.candidates![0].finishReason).toBe("STOP");
    });

    it("converts tool_use to functionCall parts", () => {
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

      const parts = result.candidates![0].content!.parts!;
      expect(parts[0].functionCall).toEqual({
        id: "tu_1",
        name: "get_weather",
        args: { city: "SF" },
      });
    });

    it("converts thinking blocks to thought parts", () => {
      const result = converter.convertResponse(
        makeMessage({
          content: [
            { type: "thinking", thinking: "Let me think...", signature: "sig123" },
            { type: "text", text: "42", citations: null },
          ],
        })
      );

      const parts = result.candidates![0].content!.parts!;
      expect(parts[0].thought).toBe(true);
      expect(parts[0].text).toBe("Let me think...");
      expect(parts[1].text).toBe("42");
    });

    it("maps max_tokens stop_reason to MAX_TOKENS", () => {
      const result = converter.convertResponse(makeMessage({ stop_reason: "max_tokens" }));
      expect(result.candidates![0].finishReason).toBe("MAX_TOKENS");
    });

    it("converts usage", () => {
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

      const usage = result.usageMetadata as any;
      expect(usage.promptTokenCount).toBe(100);
      expect(usage.candidatesTokenCount).toBe(50);
      expect(usage.cachedContentTokenCount).toBe(30);
    });
  });

  // ===== convertStreamEvent (Messages → Gemini, backward) =====

  describe("convertStreamEvent", () => {
    function initStream(c: GeminiToMessagesConverter) {
      c.convertStreamEvent({
        type: "message_start",
        message: {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "gemini-2.0-flash" as Anthropic.Model,
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
    }

    it("emits initial chunk on message_start", () => {
      const c = new GeminiToMessagesConverter();
      const result = c.convertStreamEvent({
        type: "message_start",
        message: {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "gemini-2.0-flash" as Anthropic.Model,
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

      expect(result).not.toBeNull();
      expect(result!.responseId).toBe("msg_1");
    });

    it("accumulates text on text_delta", () => {
      const c = new GeminiToMessagesConverter();
      initStream(c);
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "", citations: null },
      });

      const r1 = c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "Hello" },
      });
      expect(r1!.candidates![0].content!.parts!.some(p => p.text === "Hello")).toBe(true);

      const r2 = c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: " world" },
      });
      expect(r2!.candidates![0].content!.parts!.some(p => p.text === "Hello world")).toBe(true);
    });

    it("emits functionCall on tool_use block start", () => {
      const c = new GeminiToMessagesConverter();
      initStream(c);

      const result = c.convertStreamEvent({
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

      const parts = result!.candidates![0].content!.parts!;
      expect(parts.some(p => p.functionCall?.name === "get_weather")).toBe(true);
    });

    it("accumulates thinking on thinking_delta", () => {
      const c = new GeminiToMessagesConverter();
      initStream(c);
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "thinking", thinking: "", signature: "" },
      });

      const result = c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "Thinking..." } as any,
      });

      const parts = result!.candidates![0].content!.parts!;
      expect(parts.some(p => p.thought === true && p.text === "Thinking...")).toBe(true);
    });

    it("emits finishReason on message_stop", () => {
      const c = new GeminiToMessagesConverter();
      initStream(c);
      c.convertStreamEvent({
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "", citations: null },
      });
      c.convertStreamEvent({
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "Hi" },
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

      const result = c.convertStreamEvent({ type: "message_stop" });
      expect(result).not.toBeNull();
      expect(result!.candidates![0].finishReason).toBe("STOP");
    });
  });

  // ===== convertStream (AsyncIterable) =====

  describe("convertStream", () => {
    it("converts a full stream", async () => {
      const c = new GeminiToMessagesConverter();

      async function* makeStream(): AsyncIterable<Anthropic.RawMessageStreamEvent> {
        yield {
          type: "message_start",
          message: {
            id: "msg_1",
            type: "message",
            role: "assistant",
            model: "gemini-2.0-flash" as Anthropic.Model,
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
