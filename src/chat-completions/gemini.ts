import type OpenAI from "openai";
import type {
  GenerateContentParameters,
  GenerateContentConfig,
  GenerateContentResponse,
  Content,
  Part,
  FunctionDeclaration,
  FunctionCallingConfigMode,
  FinishReason,
} from "@google/genai";

interface StreamState {
  id: string;
  model: string;
  started: boolean;
  toolCallCounter: number;
  prevText: string;
  prevThought: string;
  seenFunctionCallIds: Set<string>;
}

export class ChatCompletionToGeminiConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (CC → Gemini, forward) ---

  convertRequest(params: OpenAI.ChatCompletionCreateParams): GenerateContentParameters {
    const systemParts: Part[] = [];
    const contents: Content[] = [];

    for (const msg of params.messages) {
      if (msg.role === "system" || msg.role === "developer") {
        const text =
          typeof msg.content === "string" ? msg.content : msg.content.map(p => p.text).join("\n");
        systemParts.push({ text });
      } else if (msg.role === "user") {
        contents.push({
          role: "user",
          parts: this.convertUserParts(msg.content),
        });
      } else if (msg.role === "assistant") {
        contents.push({
          role: "model",
          parts: this.convertAssistantParts(msg),
        });
      } else if (msg.role === "tool") {
        this.appendToolResponse(contents, msg);
      }
    }

    const config: GenerateContentConfig = {};

    if (systemParts.length > 0) {
      config.systemInstruction = { parts: systemParts };
    }
    if (params.max_tokens != null || params.max_completion_tokens != null) {
      config.maxOutputTokens = params.max_tokens ?? params.max_completion_tokens ?? undefined;
    }
    if (params.temperature != null) {
      config.temperature = params.temperature as number;
    }
    if (params.top_p != null) {
      config.topP = params.top_p as number;
    }
    if (params.stop != null) {
      const stop = params.stop;
      config.stopSequences = typeof stop === "string" ? [stop] : (stop as string[]);
    }
    if (params.seed != null) {
      config.seed = params.seed as number;
    }
    if (params.frequency_penalty != null) {
      config.frequencyPenalty = params.frequency_penalty as number;
    }
    if (params.presence_penalty != null) {
      config.presencePenalty = params.presence_penalty as number;
    }
    if ((params as any).n != null) {
      config.candidateCount = (params as any).n;
    }
    if (params.logprobs != null) {
      config.responseLogprobs = params.logprobs as boolean;
    }
    if (params.top_logprobs != null) {
      config.logprobs = params.top_logprobs as number;
    }
    if (params.tools) {
      const { tools, hasWebSearch } = this.convertTools(params.tools);
      if (tools.length > 0) {
        config.tools = [{ functionDeclarations: tools }];
      }
      if (hasWebSearch) {
        config.tools = [...(config.tools ?? []), { googleSearch: {} }];
      }
    }
    if ((params as any).web_search_options != null) {
      config.tools = [...(config.tools ?? []), { googleSearch: {} }];
    }
    if (params.tool_choice !== undefined) {
      config.toolConfig = {
        functionCallingConfig: this.convertToolChoice(params.tool_choice),
      };
    }
    if (params.response_format) {
      this.applyResponseFormat(config, params.response_format);
    }
    if (params.reasoning_effort != null) {
      config.thinkingConfig = this.convertReasoningEffort(params.reasoning_effort as string);
    }

    return {
      model: params.model,
      contents,
      config,
    };
  }

  // --- Response conversion (Gemini → CC, backward) ---

  convertResponse(response: GenerateContentResponse): OpenAI.ChatCompletion {
    const candidate = response.candidates?.[0];
    const parts = candidate?.content?.parts ?? [];

    const textParts: string[] = [];
    const thinkingParts: string[] = [];
    const reasoningDetails: Array<Record<string, unknown>> = [];
    const toolCalls: OpenAI.ChatCompletionMessageToolCall[] = [];
    const annotations: OpenAI.ChatCompletionMessage.Annotation[] = [];

    for (const part of parts) {
      if (part.thought && part.text) {
        thinkingParts.push(part.text);
        reasoningDetails.push({
          type: "reasoning.text",
          text: part.text,
          signature: part.thoughtSignature ?? undefined,
        });
      } else if (part.functionCall) {
        const fc = part.functionCall;
        toolCalls.push({
          id: fc.id ?? fc.name ?? `call_${this.generateId()}`,
          type: "function",
          function: {
            name: fc.name ?? "",
            arguments: JSON.stringify(fc.args ?? {}),
          },
        });
      } else if (part.text != null) {
        textParts.push(part.text);
      }
    }

    this.extractGroundingAnnotations(candidate, annotations);

    const assistantMessage: OpenAI.ChatCompletionMessage = {
      role: "assistant",
      content: textParts.length > 0 ? textParts.join("") : null,
      refusal: null,
    };

    if (thinkingParts.length > 0) {
      Object.assign(assistantMessage, {
        reasoning: thinkingParts.join(""),
        reasoning_details: reasoningDetails,
      });
    }
    if (toolCalls.length > 0) {
      assistantMessage.tool_calls = toolCalls;
    }
    if (annotations.length > 0) {
      assistantMessage.annotations = annotations;
    }

    let finishReason = this.mapFinishReason(candidate?.finishReason);
    if (toolCalls.length > 0) {
      finishReason = "tool_calls";
    }

    const usage = response.usageMetadata;
    const promptTokens = (usage?.promptTokenCount ?? 0) + (usage?.toolUsePromptTokenCount ?? 0);
    const thoughtsTokens = usage?.thoughtsTokenCount ?? 0;
    const candidatesTokens = (usage?.candidatesTokenCount ?? 0) + thoughtsTokens;

    return {
      id: response.responseId ?? `chatcmpl-${this.generateId()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: response.modelVersion ?? "",
      choices: [
        {
          index: 0,
          message: assistantMessage,
          finish_reason: finishReason,
          logprobs: null,
        },
      ],
      usage: {
        prompt_tokens: promptTokens,
        completion_tokens: candidatesTokens,
        total_tokens: usage?.totalTokenCount ?? 0,
        prompt_tokens_details: {
          cached_tokens: usage?.cachedContentTokenCount ?? 0,
        },
        completion_tokens_details: {
          reasoning_tokens: thoughtsTokens,
        },
      },
    };
  }

  // --- Stream conversion (Gemini → CC, backward) ---

  async *convertStream(
    stream: AsyncIterable<GenerateContentResponse>
  ): AsyncIterable<OpenAI.ChatCompletionChunk> {
    for await (const chunk of stream) {
      const events = this.convertStreamChunk(chunk);
      for (const event of events) {
        yield event;
      }
    }
  }

  convertStreamChunk(chunk: GenerateContentResponse): OpenAI.ChatCompletionChunk[] {
    const state = this.streamState;
    const events: OpenAI.ChatCompletionChunk[] = [];
    const candidate = chunk.candidates?.[0];
    const parts = candidate?.content?.parts ?? [];

    if (chunk.modelVersion) {
      state.model = chunk.modelVersion;
    }
    if (chunk.responseId) {
      state.id = chunk.responseId;
    }

    if (!state.started) {
      state.started = true;
      if (!state.id) {
        state.id = `chatcmpl-${this.generateId()}`;
      }
      events.push(this.makeChunk({ role: "assistant" }));
    }

    for (const part of parts) {
      if (part.thought && part.text) {
        const newThought = part.text.slice(state.prevThought.length);
        if (newThought) {
          state.prevThought = part.text;
          events.push(
            this.makeChunk({
              content: "",
              ...{
                reasoning: newThought,
              },
            })
          );
        }
      } else if (part.functionCall) {
        const fc = part.functionCall;
        const fcId = fc.id ?? fc.name ?? "";
        if (!state.seenFunctionCallIds.has(fcId)) {
          state.seenFunctionCallIds.add(fcId);
          const index = state.toolCallCounter++;
          events.push(
            this.makeChunk({
              tool_calls: [
                {
                  index,
                  id: fc.id ?? `call_${this.generateId()}`,
                  type: "function",
                  function: {
                    name: fc.name ?? "",
                    arguments: JSON.stringify(fc.args ?? {}),
                  },
                },
              ],
            })
          );
        }
      } else if (part.text != null) {
        const newText = part.text.slice(state.prevText.length);
        if (newText) {
          state.prevText = part.text;
          events.push(this.makeChunk({ content: newText }));
        }
      }
    }

    const annotations: OpenAI.ChatCompletionMessage.Annotation[] = [];
    this.extractGroundingAnnotations(candidate, annotations);
    if (annotations.length > 0) {
      events.push(
        this.makeChunk({
          content: "",
          ...{ annotations },
        })
      );
    }

    if (candidate?.finishReason) {
      let finishReason = this.mapFinishReason(candidate.finishReason);
      if (state.toolCallCounter > 0) {
        finishReason = "tool_calls";
      }
      const usage = chunk.usageMetadata;

      const finishChunk: OpenAI.ChatCompletionChunk = {
        id: state.id,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: state.model,
        choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
      };

      if (usage) {
        const promptTokens = (usage.promptTokenCount ?? 0) + (usage.toolUsePromptTokenCount ?? 0);
        const thoughtsTokens = usage.thoughtsTokenCount ?? 0;
        const candidatesTokens = (usage.candidatesTokenCount ?? 0) + thoughtsTokens;

        finishChunk.usage = {
          prompt_tokens: promptTokens,
          completion_tokens: candidatesTokens,
          total_tokens: usage.totalTokenCount ?? 0,
          prompt_tokens_details: {
            cached_tokens: usage.cachedContentTokenCount ?? 0,
          },
          completion_tokens_details: {
            reasoning_tokens: thoughtsTokens,
          },
        };
      }

      events.push(finishChunk);
    }

    return events;
  }

  // --- Private: request helpers ---

  private convertUserParts(content: OpenAI.ChatCompletionUserMessageParam["content"]): Part[] {
    if (typeof content === "string") {
      return [{ text: content }];
    }
    return content.map(part => {
      if (part.type === "text") {
        return { text: part.text };
      }
      if (part.type === "image_url") {
        return this.convertImageUrl(part);
      }
      if (part.type === "file") {
        return this.convertFile(part as any);
      }
      if (part.type === "input_audio") {
        return this.convertInputAudio(part as any);
      }
      return { text: `[Unsupported content type: ${(part as any).type}]` };
    });
  }

  private convertImageUrl(part: OpenAI.ChatCompletionContentPartImage): Part {
    const url = part.image_url.url;
    const dataUriMatch = url.match(/^data:([^;]+);base64,(.+)$/);

    if (dataUriMatch) {
      return {
        inlineData: {
          mimeType: dataUriMatch[1],
          data: dataUriMatch[2],
        },
      };
    }

    return {
      fileData: {
        fileUri: url,
        mimeType: "image/*",
      },
    };
  }

  private convertAssistantParts(msg: OpenAI.ChatCompletionAssistantMessageParam): Part[] {
    const parts: Part[] = [];

    if (msg.content) {
      if (typeof msg.content === "string") {
        parts.push({ text: msg.content });
      } else {
        for (const p of msg.content) {
          if (p.type === "text") {
            parts.push({ text: p.text });
          }
        }
      }
    }

    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (tc.type === "function") {
          let args: Record<string, unknown>;
          try {
            args = JSON.parse(tc.function.arguments || "{}");
          } catch {
            args = {};
          }
          parts.push({
            functionCall: {
              id: tc.id,
              name: tc.function.name,
              args,
            },
          });
        }
      }
    }

    return parts;
  }

  private appendToolResponse(
    contents: Content[],
    msg: OpenAI.ChatCompletionToolMessageParam
  ): void {
    const responsePart: Part = {
      functionResponse: {
        id: msg.tool_call_id,
        name: msg.tool_call_id,
        response: {
          output:
            typeof msg.content === "string" ? msg.content : msg.content.map(p => p.text).join("\n"),
        },
      },
    };

    const last = contents[contents.length - 1];
    if (last && last.role === "user" && last.parts?.length) {
      const lastPart = last.parts[last.parts.length - 1];
      if (lastPart.functionResponse) {
        last.parts.push(responsePart);
        return;
      }
    }

    contents.push({ role: "user", parts: [responsePart] });
  }

  private convertFile(part: any): Part {
    const fileData = part.file?.file_data;
    if (!fileData) return { text: "[Missing file data]" };

    if (fileData.startsWith("data:")) {
      const match = fileData.match(/^data:([^;]+);base64,(.+)$/);
      if (match) {
        return { inlineData: { mimeType: match[1], data: match[2] } };
      }
    }

    return { fileData: { fileUri: fileData, mimeType: "application/octet-stream" } };
  }

  private convertInputAudio(part: any): Part {
    const inputAudio = part.input_audio;
    if (!inputAudio) return { inlineData: {} };

    const mimeMap: Record<string, string> = {
      mp3: "audio/mp3",
      wav: "audio/wav",
    };
    return {
      inlineData: {
        mimeType: mimeMap[inputAudio.format] ?? "audio/wav",
        data: inputAudio.data,
      },
    };
  }

  private convertTools(tools: OpenAI.ChatCompletionTool[]): {
    tools: FunctionDeclaration[];
    hasWebSearch: boolean;
  } {
    let hasWebSearch = false;
    const functionTools = tools
      .filter(t => {
        if (t.type === "function") return true;
        const tt = t as any;
        if (
          tt.type === "web_search" ||
          tt.type === "web_search_preview" ||
          tt.type === "web_search_preview_2025_03_11"
        ) {
          hasWebSearch = true;
        }
        return false;
      })
      .map(t => ({
        name: (t as OpenAI.ChatCompletionFunctionTool).function.name,
        description: (t as OpenAI.ChatCompletionFunctionTool).function.description,
        parametersJsonSchema: (t as OpenAI.ChatCompletionFunctionTool).function.parameters,
      }));
    return { tools: functionTools, hasWebSearch };
  }

  private convertToolChoice(choice: OpenAI.ChatCompletionToolChoiceOption): {
    mode?: FunctionCallingConfigMode;
    allowedFunctionNames?: string[];
  } {
    if (choice === "auto") return { mode: "AUTO" as FunctionCallingConfigMode };
    if (choice === "required") return { mode: "ANY" as FunctionCallingConfigMode };
    if (choice === "none") return { mode: "NONE" as FunctionCallingConfigMode };

    if (typeof choice === "object" && "type" in choice) {
      if (choice.type === "function" && "function" in choice) {
        const name = (choice as OpenAI.ChatCompletionNamedToolChoice).function.name;
        return {
          mode: "ANY" as FunctionCallingConfigMode,
          allowedFunctionNames: [name],
        };
      }
    }

    return { mode: "AUTO" as FunctionCallingConfigMode };
  }

  private applyResponseFormat(
    config: GenerateContentConfig,
    format: NonNullable<OpenAI.ChatCompletionCreateParams["response_format"]>
  ): void {
    if ("json_schema" in format && format.type === "json_schema") {
      config.responseMimeType = "application/json";
      config.responseJsonSchema = format.json_schema.schema;
    } else if (format.type === "json_object") {
      config.responseMimeType = "application/json";
    }
  }

  private convertReasoningEffort(effort: string | null): {
    includeThoughts?: boolean;
    thinkingBudget?: number;
  } {
    if (!effort || effort === "none" || effort === "minimal") {
      return { thinkingBudget: 0 };
    }
    const budgetMap: Record<string, number> = {
      low: 2048,
      medium: 5120,
      high: 10240,
      xhigh: 20480,
    };
    return {
      includeThoughts: true,
      thinkingBudget: budgetMap[effort] ?? 10240,
    };
  }

  // --- Private: response helpers ---

  private extractGroundingAnnotations(
    candidate: any,
    annotations: OpenAI.ChatCompletionMessage.Annotation[]
  ): void {
    if (!candidate?.groundingMetadata?.groundingChunks) return;
    for (const chunk of candidate.groundingMetadata.groundingChunks) {
      const web = chunk.web;
      if (web) {
        annotations.push({
          type: "url_citation",
          url_citation: {
            title: web.title || "",
            url: web.uri || "",
            start_index: 0,
            end_index: 0,
          },
        });
      }
    }
  }

  private mapFinishReason(
    reason: FinishReason | string | null | undefined
  ): OpenAI.ChatCompletion.Choice["finish_reason"] {
    switch (reason) {
      case "STOP":
        return "stop";
      case "MAX_TOKENS":
        return "length";
      case "SAFETY":
      case "RECITATION":
      case "BLOCKLIST":
      case "PROHIBITED_CONTENT":
        return "content_filter";
      default:
        return "stop";
    }
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  // --- Private: stream helpers ---

  private createStreamState(): StreamState {
    return {
      id: "",
      model: "",
      started: false,
      toolCallCounter: 0,
      prevText: "",
      prevThought: "",
      seenFunctionCallIds: new Set(),
    };
  }

  private makeChunk(
    delta: OpenAI.ChatCompletionChunk.Choice.Delta,
    finish_reason: OpenAI.ChatCompletionChunk.Choice["finish_reason"] = null
  ): OpenAI.ChatCompletionChunk {
    return {
      id: this.streamState.id,
      object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000),
      model: this.streamState.model,
      choices: [{ index: 0, delta, finish_reason }],
    };
  }
}
