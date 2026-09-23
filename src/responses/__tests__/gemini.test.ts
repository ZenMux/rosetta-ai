import type OpenAI from "openai";
import type { GenerateContentResponse, Candidate } from "@google/genai";
import { ResponsesToGeminiConverter } from "../gemini";

describe("ResponsesToGeminiConverter", () => {
  const converter = new ResponsesToGeminiConverter();

  describe("convertRequest", () => {
    it("converts string input to user content", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: "Hello",
      });

      expect(result.model).toBe("gemini-2.0-flash");
      expect(result.contents as any[]).toEqual([{ role: "user", parts: [{ text: "Hello" }] }]);
    });

    it("converts instructions to systemInstruction", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: "Hi",
        instructions: "Be helpful.",
      });

      expect(result.config?.systemInstruction).toEqual({
        parts: [{ text: "Be helpful." }],
      });
    });

    it("maps max_output_tokens to maxOutputTokens", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: "Hi",
        max_output_tokens: 2000,
      });
      expect(result.config?.maxOutputTokens).toBe(2000);
    });

    it("maps temperature and top_p", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: "Hi",
        temperature: 0.7,
        top_p: 0.9,
      });
      expect(result.config?.temperature).toBe(0.7);
      expect(result.config?.topP).toBe(0.9);
    });

    it("converts input array with user/assistant messages", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: [
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

    it.each([
      { output: "72F" },
      {
        output: [
          { type: "input_text" as const, text: "72" },
          { type: "input_text" as const, text: "F" },
        ],
      },
      { output: [] },
    ])("converts function_call and preserves function_call_output: %j", ({ output }) => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: [
          { role: "user", content: "Weather?" },
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
            output,
          },
        ],
      });

      const modelParts = (result.contents as any[])[1].parts;
      expect(modelParts[0].functionCall).toEqual({
        id: "call_1",
        name: "get_weather",
        args: { city: "SF" },
      });

      const toolParts = (result.contents as any[])[2].parts;
      expect(toolParts[0].functionResponse).toEqual({
        id: "call_1",
        name: "get_weather",
        response: { output },
      });
    });

    it("converts function tools to functionDeclarations", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
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

      const tools = result.config?.tools as any[];
      expect(tools[0].functionDeclarations[0]).toEqual({
        name: "get_weather",
        description: "Get weather",
        parametersJsonSchema: { type: "object", properties: { city: { type: "string" } } },
      });
    });

    it("expands namespace tools into namespaced functionDeclarations", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: "Call agents.spawn_agent",
        tools: [
          {
            type: "namespace",
            name: "agents",
            description: "Multi-agent collaboration tools.",
            tools: [
              {
                type: "function",
                name: "spawn_agent",
                description: "Spawn a child agent.",
                strict: false,
                parameters: {
                  type: "object",
                  properties: { task_name: { type: "string" }, message: { type: "string" } },
                },
              },
            ],
          },
        ] as any,
      });

      const tools = result.config?.tools as any[];
      expect(tools[0].functionDeclarations[0]).toEqual({
        name: "agents___spawn_agent",
        description: "Spawn a child agent.",
        parametersJsonSchema: {
          type: "object",
          properties: { task_name: { type: "string" }, message: { type: "string" } },
        },
      });
    });

    it("converts web_search tools to googleSearch", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: "Hi",
        tools: [{ type: "web_search_preview" }],
      } as any);

      const tools = result.config?.tools as any[];
      expect(tools.some((t: any) => t.googleSearch != null)).toBe(true);
    });

    it("maps tool_choice", () => {
      expect(
        converter.convertRequest({
          model: "gemini-2.0-flash",
          input: "Hi",
          tool_choice: "auto",
        }).config?.toolConfig?.functionCallingConfig?.mode
      ).toBe("AUTO");

      expect(
        converter.convertRequest({
          model: "gemini-2.0-flash",
          input: "Hi",
          tool_choice: "required",
        }).config?.toolConfig?.functionCallingConfig?.mode
      ).toBe("ANY");

      expect(
        converter.convertRequest({
          model: "gemini-2.0-flash",
          input: "Hi",
          tool_choice: "none",
        }).config?.toolConfig?.functionCallingConfig?.mode
      ).toBe("NONE");
    });

    it("maps reasoning.effort to thinkingConfig", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: "Hi",
        reasoning: { effort: "high" },
      });

      expect(result.config?.thinkingConfig).toEqual({
        includeThoughts: true,
        thinkingBudget: 10240,
      });
    });

    it("converts json_schema text format", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: "Hi",
        text: {
          format: {
            type: "json_schema",
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

    it("extracts system messages from input to systemInstruction", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: [
          { role: "system", content: "You are an assistant." } as any,
          { role: "user", content: "Hello" },
        ],
      });

      expect(result.config?.systemInstruction).toEqual({
        parts: [{ text: "You are an assistant." }],
      });
    });
  });

  // ===== convertResponse (Gemini → Responses, backward) =====

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
      expect(result.object).toBe("response");
      expect(result.status).toBe("completed");

      const msgOutput = result.output.find((o: any) => o.type === "message") as any;
      expect(msgOutput).toBeDefined();
      expect(msgOutput.content[0].type).toBe("output_text");
      expect(msgOutput.content[0].text).toBe("Hello!");
    });

    it("converts functionCall parts to function_call output items", () => {
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

      const fcOutput = result.output.find((o: any) => o.type === "function_call") as any;
      expect(fcOutput).toBeDefined();
      expect(fcOutput.name).toBe("get_weather");
      expect(fcOutput.arguments).toBe('{"city":"SF"}');
      expect(result.status).toBe("completed");
    });

    it("splits a namespaced functionCall name into { namespace, name }", () => {
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
                      name: "agents___spawn_agent",
                      args: { task_name: "child" },
                    },
                  },
                ],
              },
              finishReason: "STOP",
            } as any,
          ],
        })
      );

      const fcOutput = result.output.find((o: any) => o.type === "function_call") as any;
      expect(fcOutput.name).toBe("spawn_agent");
      expect(fcOutput.namespace).toBe("agents");
    });

    it("converts thought parts to reasoning output", () => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: {
                role: "model",
                parts: [{ text: "Let me think...", thought: true }, { text: "42" }],
              },
              finishReason: "STOP",
            } as any,
          ],
        })
      );

      const reasoning = result.output.find((o: any) => o.type === "reasoning") as any;
      expect(reasoning).toBeDefined();
      expect(reasoning.summary[0].text).toBe("Let me think...");

      const msg = result.output.find((o: any) => o.type === "message") as any;
      expect(msg.content[0].text).toBe("42");
    });

    it("maps MAX_TOKENS to incomplete status", () => {
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

      expect(result.status).toBe("incomplete");
      expect(result.incomplete_details).toEqual({ reason: "max_output_tokens" });
    });

    it("maps SAFETY to failed status", () => {
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

      expect(result.status).toBe("failed");
    });

    it("extracts grounding annotations", () => {
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

      const msg = result.output.find((o: any) => o.type === "message") as any;
      expect(msg.content[0].annotations[0].url).toBe("https://example.com");
    });

    it.each([
      {},
      { segment: { partIndex: 1, text: "北京" } },
      { segment: { partIndex: 1, endIndex: 6 } },
      { segment: { partIndex: 1, startIndex: 0 } },
      { segment: { partIndex: 9, startIndex: 0, endIndex: 6 } },
      { segment: { partIndex: 1, startIndex: 0, endIndex: 999 } },
      { segment: { partIndex: 1, startIndex: 6, endIndex: 0 } },
    ])("retains the source at 0–0 when its citation cannot be located: %j", support => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: { parts: [{ text: "参考：" }, { text: "北京" }] },
              finishReason: "STOP",
              groundingMetadata: {
                groundingChunks: [{ web: { uri: "https://example.com", title: "Source" } }],
                groundingSupports: [{ ...support, groundingChunkIndices: [0] }],
              },
            } as Candidate,
          ],
        })
      );
      const message = result.output.find(item => item.type === "message")!;
      expect(message.content[0]).toMatchObject({
        annotations: [
          {
            type: "url_citation",
            url: "https://example.com",
            title: "Source",
            start_index: 0,
            end_index: 0,
          },
        ],
      });
    });

    it("keeps accurate citations alongside sources with missing or unusable associations", () => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: { parts: [{ text: "参考：" }, { text: "北京" }] },
              finishReason: "STOP",
              groundingMetadata: {
                groundingChunks: [
                  { web: { uri: "https://example.com", title: "Located" } },
                  { web: { title: "Unassociated" } },
                  { web: { uri: "https://example.com/missing", title: "No indices" } },
                ],
                groundingSupports: [
                  {
                    segment: { partIndex: 1, startIndex: 0, endIndex: 6 },
                    groundingChunkIndices: [0, 0],
                  },
                  { segment: { partIndex: 1, startIndex: 0, endIndex: 6 } },
                  { groundingChunkIndices: [99] },
                ],
              },
            } as Candidate,
          ],
        })
      );
      const message = result.output.find(item => item.type === "message")!;
      expect(message.content[0]).toMatchObject({
        annotations: [
          { url: "https://example.com", start_index: 3, end_index: 5 },
          { url: "https://example.com", start_index: 3, end_index: 5 },
          { url: "", title: "Unassociated", start_index: 0, end_index: 0 },
          { url: "https://example.com/missing", start_index: 0, end_index: 0 },
        ],
      });
    });

    it("converts usage with toolUsePromptTokenCount and thoughtsTokenCount", () => {
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

      expect(result.usage?.input_tokens).toBe(110);
      expect(result.usage?.output_tokens).toBe(70);
      expect(result.usage?.output_tokens_details?.reasoning_tokens).toBe(20);
    });

    it("keeps the original reasoning/tool/text grouping and distinct unnamed call IDs", () => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: {
                role: "model",
                parts: [
                  { text: "Before " },
                  { thought: true, text: "First thought" },
                  { functionCall: { name: "lookup", args: { key: 1 } } },
                  { text: "after" },
                  { thought: true, text: "Second thought" },
                  { functionCall: { name: "lookup", args: { key: 2 } } },
                ],
              },
              finishReason: "STOP",
            } as Candidate,
          ],
        })
      );
      expect(result.output.map(item => item.type)).toEqual([
        "reasoning",
        "reasoning",
        "function_call",
        "function_call",
        "message",
      ]);
      const calls = result.output.filter(item => item.type === "function_call");
      expect(calls[0].call_id).not.toBe(calls[1].call_id);
      const message = result.output.find(item => item.type === "message")!;
      expect(message.content[0]).toMatchObject({ text: "Before after" });
    });

    it("converts responses independently without consuming an existing stream", () => {
      const converter = new ResponsesToGeminiConverter();
      converter.convertStreamChunk(
        makeResponse({
          responseId: "stream_id",
          candidates: [{ content: { parts: [{ text: "Stream " }] } } as Candidate],
        })
      );
      const first = converter.convertResponse(makeResponse({ responseId: "first" }));
      const second = converter.convertResponse(makeResponse({ responseId: "second" }));
      expect(first.id).toBe("first");
      expect(second.id).toBe("second");
      expect(second.output).toHaveLength(1);
      expect(second.output[0]).toMatchObject({ content: [{ text: "Hello!" }] });
      const terminal = converter
        .convertStreamChunk(
          makeResponse({
            responseId: "stream_id",
            candidates: [
              {
                content: { parts: [{ text: "end" }] },
                finishReason: "STOP",
              } as Candidate,
            ],
          })
        )
        .find(event => event.type === "response.completed")!;
      expect(terminal.response.id).toBe("stream_id");
      expect(terminal.response.output[0]).toMatchObject({ content: [{ text: "Stream end" }] });
    });

    it("preserves signatures and namespaced function calls when replaying the output", () => {
      const response = makeResponse({
        candidates: [
          {
            content: {
              role: "model",
              parts: [
                { thought: true, text: "Thinking", thoughtSignature: "thought-signature" },
                { functionCall: { name: "agents___lookup", args: { key: 1 } } },
                { thoughtSignature: "call-signature" },
                { text: "Answer", thoughtSignature: "text-signature" },
                { text: " tail" },
              ],
            },
            finishReason: "STOP",
          } as Candidate,
        ],
      });
      const original = structuredClone(response);
      const result = converter.convertResponse(response);
      const replay = converter.convertRequest({
        model: "gemini",
        input: result.output.filter(
          item =>
            item.type === "message" || item.type === "reasoning" || item.type === "function_call"
        ),
      });
      const parts = (replay.contents as any[]).flatMap(content => content.parts);
      expect(parts).toEqual([
        { thought: true, text: "Thinking", thoughtSignature: "thought-signature" },
        {
          functionCall: {
            id: expect.any(String),
            name: "agents___lookup",
            args: { key: 1 },
          },
          thoughtSignature: "call-signature",
        },
        { text: "Answer", thoughtSignature: "text-signature" },
        { text: " tail" },
      ]);
      expect(response).toEqual(original);
    });

    it("maps citation byte offsets to the merged text and keeps logprobs null", () => {
      const result = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: {
                parts: [{ thought: true, text: "Thinking" }, { text: "前缀" }, { text: "你好🌍" }],
              },
              finishReason: "STOP",
              groundingMetadata: {
                groundingChunks: [{ web: { uri: "https://example.com", title: "Source" } }],
                groundingSupports: [
                  {
                    segment: { partIndex: 2, startIndex: 6, endIndex: 10 },
                    groundingChunkIndices: [0],
                  },
                ],
              },
              logprobsResult: {
                chosenCandidates: [{ token: "前缀你好🌍", logProbability: -0.2 }],
                topCandidates: [{ candidates: [{ token: "前缀你好🌍", logProbability: -0.2 }] }],
              },
            } as Candidate,
          ],
        })
      );
      const message = result.output.find(item => item.type === "message")!;
      expect(message.content[0]).toMatchObject({
        text: "前缀你好🌍",
        annotations: [{ url: "https://example.com", start_index: 4, end_index: 5 }],
        logprobs: [],
      });
    });

    it("does not mark truncated tool responses or blocked prompts as completed", () => {
      const truncated = converter.convertResponse(
        makeResponse({
          candidates: [
            {
              content: {
                parts: [{ functionCall: { name: "lookup", args: {} } }, { text: "Partial" }],
              },
              finishReason: "MAX_TOKENS",
            } as Candidate,
          ],
        })
      );
      expect(truncated.status).toBe("incomplete");
      expect(truncated.output.find(item => item.type === "message")?.status).toBe("incomplete");
      const blocked = converter.convertResponse(
        makeResponse({
          candidates: [],
          promptFeedback: { blockReason: "SAFETY" } as any,
        })
      );
      expect(blocked.status).toBe("failed");
      expect(blocked.error).toMatchObject({ code: "invalid_prompt" });
      expect(blocked.output).toEqual([]);
    });

    it("uses default response configuration in both response modes", () => {
      const converter = new ResponsesToGeminiConverter();
      const params: OpenAI.Responses.ResponseCreateParams = {
        model: "gemini",
        input: "Hello",
        instructions: "Use JSON",
        temperature: 0.4,
        max_output_tokens: 100,
        parallel_tool_calls: false,
        tools: [{ type: "function", name: "lookup", parameters: null, strict: false }],
        text: { format: { type: "json_object" } },
      };
      converter.convertRequest(params);
      const expected = {
        instructions: null,
        temperature: null,
        max_output_tokens: null,
        parallel_tool_calls: true,
        text: { format: { type: "text" } },
        tools: [],
      };
      expect(converter.convertResponse(makeResponse())).toMatchObject(expected);
      const terminal = converter
        .convertStreamChunk(makeResponse())
        .find(event => event.type === "response.completed")!;
      expect(terminal.response).toMatchObject(expected);
    });
  });

  // ===== convertStreamChunk (Gemini → Responses, backward) =====

  describe("convertStreamChunk", () => {
    function makeChunk(overrides: Partial<GenerateContentResponse> = {}): GenerateContentResponse {
      return {
        candidates: [{ content: { role: "model", parts: [] } } as any],
        responseId: "resp_1",
        modelVersion: "gemini-2.0-flash",
        ...overrides,
      } as GenerateContentResponse;
    }

    it("emits response.created and response.in_progress on first chunk", () => {
      const c = new ResponsesToGeminiConverter();
      const events = c.convertStreamChunk(makeChunk());

      const types = events.map(e => e.type);
      expect(types).toContain("response.created");
      expect(types).toContain("response.in_progress");
    });

    it("emits output_text.delta for new text", () => {
      const c = new ResponsesToGeminiConverter();
      c.convertStreamChunk(makeChunk());

      const events = c.convertStreamChunk(
        makeChunk({
          candidates: [{ content: { role: "model", parts: [{ text: "Hello" }] } } as any],
        })
      );

      const textDelta = events.find(e => e.type === "response.output_text.delta") as any;
      expect(textDelta).toBeDefined();
      expect(textDelta.delta).toBe("Hello");
    });

    it("emits a complete function call once per call ID", () => {
      const c = new ResponsesToGeminiConverter();
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

      const itemAdded = events.find(e => e.type === "response.output_item.added") as any;
      expect(itemAdded).toBeDefined();
      expect(itemAdded.item.type).toBe("function_call");
      expect(
        events.find(event => event.type === "response.function_call_arguments.delta")
      ).toMatchObject({ item_id: itemAdded.item.id, delta: '{"city":"SF"}' });
      const itemDone = events.find(event => event.type === "response.output_item.done")!;
      expect(itemDone.item).toMatchObject({
        id: itemAdded.item.id,
        call_id: "call_1",
        arguments: '{"city":"SF"}',
        status: "completed",
      });

      const terminalEvents = c.convertStreamChunk(
        makeChunk({
          candidates: [
            {
              content: {
                parts: [
                  { functionCall: { id: "call_1", name: "get_weather", args: { city: "NY" } } },
                ],
              },
              finishReason: "STOP",
            } as Candidate,
          ],
        })
      );
      expect(terminalEvents.map(event => event.type)).toEqual(["response.completed"]);
      const terminal = terminalEvents.find(event => event.type === "response.completed")!;
      expect(terminal.response.output).toEqual([itemDone.item]);
    });

    it("emits reasoning_summary_text.delta for thought parts", () => {
      const c = new ResponsesToGeminiConverter();
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

      const reasoningDelta = events.find(
        e => e.type === "response.reasoning_summary_text.delta"
      ) as any;
      expect(reasoningDelta).toBeDefined();
      expect(reasoningDelta.delta).toBe("Thinking...");
    });

    it.each([
      {},
      { segment: { partIndex: 2, text: "北京" } },
      { segment: { partIndex: 2, endIndex: 6 } },
      { segment: { partIndex: 2, startIndex: 0 } },
      { segment: { partIndex: 9, startIndex: 0, endIndex: 6 } },
      { segment: { partIndex: 0, startIndex: 0, endIndex: 3 } },
      { segment: { partIndex: 2, startIndex: 0, endIndex: 999 } },
      { segment: { partIndex: 2, startIndex: 6, endIndex: 0 } },
      { segment: { partIndex: 2, startIndex: -1, endIndex: 6 } },
      { segment: { partIndex: 2, startIndex: 0.5, endIndex: 6 } },
    ])("keeps an unlocatable streamed citation at 0–0: %j", support => {
      const c = new ResponsesToGeminiConverter();
      c.convertStreamChunk(
        makeChunk({
          candidates: [
            {
              content: {
                parts: [{ thought: true, text: "Thinking" }, { text: "参考：" }, { text: "北京" }],
              },
            },
          ],
        })
      );
      const events = c.convertStreamChunk(
        makeChunk({
          candidates: [
            {
              finishReason: "STOP",
              groundingMetadata: {
                groundingChunks: [{ web: { uri: "https://example.com", title: "Source" } }],
                groundingSupports: [{ ...support, groundingChunkIndices: [0] }],
              },
            } as Candidate,
          ],
        })
      );
      const expected = {
        type: "url_citation",
        url: "https://example.com",
        title: "Source",
        start_index: 0,
        end_index: 0,
      };
      const terminal = events.find(event => event.type === "response.completed")!;
      const message = terminal.response.output.find(item => item.type === "message")!;
      expect(message.content[0]).toMatchObject({ annotations: [expected] });
      expect(
        events.find(event => event.type === "response.output_text.annotation.added")
      ).toMatchObject({ output_index: 1, annotation: expected });
    });

    it("keeps streamed sources at 0–0 when supports are absent", () => {
      const events = new ResponsesToGeminiConverter().convertStreamChunk(
        makeChunk({
          candidates: [
            {
              content: { parts: [{ text: "Answer" }] },
              finishReason: "STOP",
              groundingMetadata: {
                groundingChunks: [{ web: { uri: "https://example.com", title: "Source" } }],
              },
            } as Candidate,
          ],
        })
      );
      const terminal = events.find(event => event.type === "response.completed")!;
      const message = terminal.response.output.find(item => item.type === "message")!;
      expect(message.content[0]).toMatchObject({
        annotations: [{ url: "https://example.com", start_index: 0, end_index: 0 }],
      });
    });

    it("keeps precise streamed spans, duplicate references and unassociated sources", () => {
      const events = new ResponsesToGeminiConverter().convertStreamChunk(
        makeChunk({
          candidates: [
            {
              content: {
                parts: [
                  { text: "First" },
                  { functionCall: { id: "call_1", name: "lookup", args: {} } },
                  { text: "参考：" },
                  { text: "你好🌍" },
                ],
              },
              finishReason: "STOP",
              groundingMetadata: {
                groundingChunks: [
                  { web: { uri: "https://example.com", title: "Located" } },
                  { web: { title: "No text association" } },
                  { web: { uri: "https://example.com/extra", title: "Unassociated" } },
                ],
                groundingSupports: [
                  {
                    segment: { partIndex: 3, startIndex: 6, endIndex: 10 },
                    groundingChunkIndices: [0, 0],
                  },
                  {
                    segment: { partIndex: 1, startIndex: 0, endIndex: 1 },
                    groundingChunkIndices: [1],
                  },
                ],
              },
            } as Candidate,
          ],
        })
      );
      const terminal = events.find(event => event.type === "response.completed")!;
      const messages = terminal.response.output.filter(item => item.type === "message");
      expect(messages[0].content[0]).toMatchObject({
        annotations: [
          { url: "", title: "No text association", start_index: 0, end_index: 0 },
          { url: "https://example.com/extra", start_index: 0, end_index: 0 },
        ],
      });
      expect(messages[1].content[0]).toMatchObject({
        text: "参考：你好🌍",
        annotations: [
          { url: "https://example.com", start_index: 5, end_index: 6 },
          { url: "https://example.com", start_index: 5, end_index: 6 },
        ],
      });
      expect(
        events.filter(event => event.type === "response.output_text.annotation.added")
      ).toMatchObject([
        { output_index: 0, annotation_index: 0 },
        { output_index: 0, annotation_index: 1 },
        { output_index: 2, annotation_index: 0 },
        { output_index: 2, annotation_index: 1 },
      ]);
    });

    it("emits response.completed on finish", () => {
      const c = new ResponsesToGeminiConverter();
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

      const completed = events.find(e => e.type === "response.completed") as any;
      expect(completed).toBeDefined();
      expect(completed.response.status).toBe("completed");
    });

    it("emits response.incomplete on MAX_TOKENS", () => {
      const c = new ResponsesToGeminiConverter();
      c.convertStreamChunk(makeChunk());

      const events = c.convertStreamChunk(
        makeChunk({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Partial" }] },
              finishReason: "MAX_TOKENS",
            } as any,
          ],
        })
      );

      const incomplete = events.find(e => e.type === "response.incomplete") as any;
      expect(incomplete).toBeDefined();
    });
  });

  // ===== convertStream (AsyncIterable) =====

  describe("convertStream", () => {
    it("converts a full stream", async () => {
      const c = new ResponsesToGeminiConverter();

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
              logprobsResult: {
                chosenCandidates: [{ token: "Hello world", logProbability: -0.2 }],
              },
            } as any,
          ],
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 5,
            totalTokenCount: 15,
          },
        } as GenerateContentResponse;
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
      const terminal = events.find(event => event.type === "response.completed")!;
      const message = terminal.response.output.find(item => item.type === "message")!;
      expect(message.content[0]).toMatchObject({ type: "output_text", logprobs: [] });
      for (const event of events) {
        if (
          event.type === "response.output_text.delta" ||
          event.type === "response.output_text.done"
        )
          expect(event.logprobs).toEqual([]);
      }
    });
  });

  describe("convertRequest - advanced", () => {
    it("converts include logprobs", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: "Hi",
        include: ["message.output_text.logprobs"],
      } as any);

      expect(result.config?.responseLogprobs).toBe(true);
      expect(result.config?.logprobs).toBe(20);
    });

    it("converts assistant message input items to model content", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: [
          { type: "message", role: "assistant", content: [] },
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Hello" }],
          },
        ],
      } as any);

      expect(result.contents).toHaveLength(1);
      expect((result.contents as any[])[0].role).toBe("model");
      expect((result.contents as any[])[0].parts[0].text).toBe("Hello");
    });

    it("converts system message input items to systemInstruction", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: [{ role: "system", content: "Be helpful." } as any, { role: "user", content: "Hi" }],
      });

      expect(result.config?.systemInstruction).toBeDefined();
    });

    it("converts input_image to inlineData", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_image", image_url: "data:image/png;base64,abc123" }],
          },
        ],
      } as any);

      const parts = (result.contents as any[])[0].parts;
      expect(parts[0].inlineData).toEqual({ mimeType: "image/png", data: "abc123" });
    });

    it("converts input_image URL to fileData", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_image", image_url: "https://example.com/img.png" }],
          },
        ],
      } as any);

      const parts = (result.contents as any[])[0].parts;
      expect(parts[0].fileData).toEqual({
        fileUri: "https://example.com/img.png",
        mimeType: "image/*",
      });
    });

    it("converts input_file to inlineData", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_file", file_data: "data:application/pdf;base64,pdfdata" }],
          },
        ],
      } as any);

      const parts = (result.contents as any[])[0].parts;
      expect(parts[0].inlineData).toEqual({ mimeType: "application/pdf", data: "pdfdata" });
    });

    it("converts input_file URL to fileData", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_file", file_data: "https://example.com/doc.pdf" }],
          },
        ],
      } as any);

      const parts = (result.contents as any[])[0].parts;
      expect(parts[0].fileData).toBeDefined();
    });

    it("merges consecutive function_call_output into same user content", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
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

      const contents = result.contents as any[];
      const toolResponses = contents.find(
        (c: any) => c.role === "user" && c.parts?.some((p: any) => p.functionResponse)
      );
      expect(toolResponses).toBeDefined();
      expect(toolResponses.parts.length).toBeGreaterThanOrEqual(2);
    });

    it("handles json_object text format", () => {
      const result = converter.convertRequest({
        model: "gemini-2.0-flash",
        input: "Hi",
        text: { format: { type: "json_object" } },
      } as any);

      expect(result.config?.responseMimeType).toBe("application/json");
    });
  });
});
