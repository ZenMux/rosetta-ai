import type OpenAI from "openai";
import type {
  GenerateContentParameters,
  GenerateContentConfig,
  GenerateContentResponse,
  GenerateContentResponseUsageMetadata,
  Candidate,
  Content,
  Part,
  FunctionDeclaration,
  FunctionCallingConfigMode,
  FinishReason,
} from "@google/genai";

interface ChoiceStreamState {
  toolCallCounter: number;
  toolCallIndexes: Map<string, number>;
}

interface StreamState {
  id: string;
  model: string;
  started: boolean;
  choices: Map<number, ChoiceStreamState>;
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
    const toolCallNames = new Map<string, string>();

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
        for (const toolCall of msg.tool_calls ?? []) {
          if (toolCall.type === "function") {
            toolCallNames.set(toolCall.id, toolCall.function.name);
          }
        }
        contents.push({
          role: "model",
          parts: this.convertAssistantParts(msg),
        });
      } else if (msg.role === "tool") {
        this.appendToolResponse(contents, msg, toolCallNames.get(msg.tool_call_id));
      }
    }

    const config: GenerateContentConfig = {};

    if (systemParts.length > 0) {
      config.systemInstruction = { parts: systemParts };
    }
    if (params.max_completion_tokens != null || params.max_tokens != null) {
      config.maxOutputTokens = params.max_completion_tokens ?? params.max_tokens ?? undefined;
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
    const choices = (response.candidates ?? []).map((candidate, index) =>
      this.convertCandidate(candidate, candidate.index ?? index)
    );

    if (choices.length === 0) {
      const refusal = this.getPromptFeedbackRefusal(response);
      if (refusal) {
        choices.push({
          index: 0,
          message: { role: "assistant", content: null, refusal },
          finish_reason: "content_filter",
          logprobs: null,
        });
      }
    }

    const usage = this.convertUsage(response.usageMetadata);

    return {
      id: response.responseId ?? `chatcmpl-${this.generateId()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: response.modelVersion ?? "",
      choices,
      ...(usage && { usage }),
    };
  }

  // --- Stream conversion (Gemini → CC, backward) ---

  async *convertStream(
    stream: AsyncIterable<GenerateContentResponse>
  ): AsyncIterable<OpenAI.ChatCompletionChunk> {
    this.streamState = this.createStreamState();
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
    const candidates = chunk.candidates ?? [];

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
      const indexes =
        candidates.length > 0
          ? candidates.map((candidate, index) => candidate.index ?? index)
          : [0];
      events.push(
        this.makeChunk(
          indexes.map(index => ({
            index,
            delta: { role: "assistant" },
            finish_reason: null,
          }))
        )
      );
    }

    if (candidates.length === 0) {
      const refusal = this.getPromptFeedbackRefusal(chunk);
      if (refusal) {
        events.push(
          this.makeChunk([
            {
              index: 0,
              delta: { refusal },
              finish_reason: "content_filter",
            },
          ])
        );
      }
    }

    for (let position = 0; position < candidates.length; position++) {
      const candidate = candidates[position];
      const choiceIndex = candidate.index ?? position;
      const choiceState = this.getChoiceStreamState(choiceIndex);

      for (const part of candidate.content?.parts ?? []) {
        if (part.thought) {
          continue;
        }
        if (part.functionCall) {
          const functionCall = part.functionCall;
          const functionCallId = functionCall.id ?? `call_${this.generateId()}`;
          let toolCallIndex = choiceState.toolCallIndexes.get(functionCallId);
          if (toolCallIndex == null) {
            toolCallIndex = choiceState.toolCallCounter++;
            choiceState.toolCallIndexes.set(functionCallId, toolCallIndex);
          }
          events.push(
            this.makeChunk([
              {
                index: choiceIndex,
                delta: {
                  tool_calls: [
                    {
                      index: toolCallIndex,
                      id: functionCallId,
                      type: "function",
                      function: {
                        name: functionCall.name ?? "",
                        arguments: JSON.stringify(functionCall.args ?? {}),
                      },
                    },
                  ],
                },
                finish_reason: null,
              },
            ])
          );
        } else if (part.text != null && part.text !== "") {
          events.push(
            this.makeChunk([
              {
                index: choiceIndex,
                delta: { content: part.text },
                finish_reason: null,
                logprobs: this.convertLogprobs(candidate),
              },
            ])
          );
        }
      }

      const annotations: OpenAI.ChatCompletionMessage.Annotation[] = [];
      this.extractGroundingAnnotations(candidate, annotations);
      if (annotations.length > 0) {
        events.push(
          this.makeChunk([
            {
              index: choiceIndex,
              delta: { content: "", ...{ annotations } },
              finish_reason: null,
            },
          ])
        );
      }

      if (candidate.finishReason) {
        events.push(
          this.makeChunk([
            {
              index: choiceIndex,
              delta: {},
              finish_reason:
                choiceState.toolCallCounter > 0
                  ? "tool_calls"
                  : this.mapFinishReason(candidate.finishReason),
            },
          ])
        );
      }
    }

    const usage = this.convertUsage(chunk.usageMetadata);
    if (usage) {
      events.push({
        ...this.makeChunk([]),
        usage,
      });
    }

    return events;
  }

  // --- Private: request helpers ---

  private convertUserParts(content: OpenAI.ChatCompletionUserMessageParam["content"]): Part[] {
    if (typeof content === "string") {
      return [{ text: content }];
    }
    return content.flatMap(part => {
      if (part.type === "text") {
        return [{ text: part.text }];
      }
      if (part.type === "image_url") {
        return [this.convertImageUrl(part)];
      }
      if (part.type === "file") {
        return [this.convertFile(part)];
      }
      if (part.type === "input_audio") {
        return [this.convertInputAudio(part)];
      }
      return [];
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
        mimeType: "image/jpeg",
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
    msg: OpenAI.ChatCompletionToolMessageParam,
    functionName?: string
  ): void {
    const responsePart: Part = {
      functionResponse: {
        id: msg.tool_call_id,
        name: functionName ?? msg.tool_call_id,
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

  private convertFile(part: OpenAI.ChatCompletionContentPart.File): Part {
    const fileData = part.file?.file_data;
    if (!fileData) {
      throw new Error("Chat Completions file content requires file_data for Gemini conversion");
    }

    if (fileData.startsWith("data:")) {
      const match = fileData.match(/^data:([^;]+);base64,(.+)$/);
      if (match) {
        return { inlineData: { mimeType: match[1], data: match[2] } };
      }
    }

    return {
      fileData: {
        fileUri: fileData,
        mimeType: this.inferFileMimeType(part.file.filename ?? fileData),
      },
    };
  }

  private inferFileMimeType(source: string): string {
    const pathname = source.split(/[?#]/, 1)[0].toLowerCase();
    const mimeTypes: Record<string, string> = {
      ".pdf": "application/pdf",
      ".mp4": "video/mp4",
      ".avi": "video/x-msvideo",
      ".mov": "video/quicktime",
      ".mpeg": "video/mpeg",
      ".webm": "video/webm",
    };
    const extension = Object.keys(mimeTypes).find(candidate => pathname.endsWith(candidate));
    return extension ? mimeTypes[extension] : "application/octet-stream";
  }

  private convertInputAudio(part: OpenAI.ChatCompletionContentPartInputAudio): Part {
    const inputAudio = part.input_audio;
    if (!inputAudio) {
      return { inlineData: {} };
    }

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
    if (format.type === "text") {
      config.responseMimeType = "text/plain";
    } else if ("json_schema" in format && format.type === "json_schema") {
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

  private convertCandidate(candidate: Candidate, index: number): OpenAI.ChatCompletion.Choice {
    const textParts: string[] = [];
    const toolCalls: OpenAI.ChatCompletionMessageToolCall[] = [];
    const annotations: OpenAI.ChatCompletionMessage.Annotation[] = [];

    for (const part of candidate.content?.parts ?? []) {
      if (part.thought) {
        continue;
      }
      if (part.functionCall) {
        const functionCall = part.functionCall;
        toolCalls.push({
          id: functionCall.id ?? functionCall.name ?? `call_${this.generateId()}`,
          type: "function",
          function: {
            name: functionCall.name ?? "",
            arguments: JSON.stringify(functionCall.args ?? {}),
          },
        });
      } else if (part.text != null) {
        textParts.push(part.text);
      }
    }

    this.extractGroundingAnnotations(candidate, annotations);

    return {
      index,
      message: {
        role: "assistant",
        content: textParts.join(""),
        refusal: null,
        ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
        ...(annotations.length > 0 && { annotations }),
      },
      finish_reason:
        toolCalls.length > 0 ? "tool_calls" : this.mapFinishReason(candidate.finishReason),
      logprobs: this.convertLogprobs(candidate),
    };
  }

  private getPromptFeedbackRefusal(response: GenerateContentResponse): string | null {
    const feedback = response.promptFeedback;
    if (!feedback?.blockReason) {
      return null;
    }
    return feedback.blockReasonMessage ?? feedback.blockReason;
  }

  private convertUsage(
    usage?: GenerateContentResponseUsageMetadata
  ): OpenAI.CompletionUsage | undefined {
    if (
      !usage ||
      [
        usage.promptTokenCount,
        usage.candidatesTokenCount,
        usage.totalTokenCount,
        usage.cachedContentTokenCount,
        usage.thoughtsTokenCount,
        usage.toolUsePromptTokenCount,
        usage.promptTokensDetails,
        usage.candidatesTokensDetails,
        usage.cacheTokensDetails,
        usage.toolUsePromptTokensDetails,
      ].every(value => value == null)
    ) {
      return undefined;
    }

    const thoughtsTokens = usage.thoughtsTokenCount ?? 0;
    const promptTokens = (usage.promptTokenCount ?? 0) + (usage.toolUsePromptTokenCount ?? 0);
    const completionTokens = (usage.candidatesTokenCount ?? 0) + thoughtsTokens;
    const promptAudioTokens = usage.promptTokensDetails?.find(
      detail => detail.modality === "AUDIO"
    )?.tokenCount;

    return {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: usage.totalTokenCount ?? 0,
      prompt_tokens_details: {
        cached_tokens: usage.cachedContentTokenCount ?? 0,
        audio_tokens: promptAudioTokens ?? 0,
      },
      completion_tokens_details: {
        reasoning_tokens: thoughtsTokens,
      },
    };
  }

  private convertLogprobs(candidate: Candidate): OpenAI.ChatCompletion.Choice.Logprobs | null {
    const chosenCandidates = candidate.logprobsResult?.chosenCandidates;
    if (!chosenCandidates || chosenCandidates.length === 0) {
      return null;
    }

    return {
      content: chosenCandidates.map((chosen, index) => ({
        token: chosen.token ?? "",
        bytes: this.toUtf8Bytes(chosen.token),
        logprob: chosen.logProbability ?? -9999,
        top_logprobs: (candidate.logprobsResult?.topCandidates?.[index]?.candidates ?? []).map(
          topCandidate => ({
            token: topCandidate.token ?? "",
            bytes: this.toUtf8Bytes(topCandidate.token),
            logprob: topCandidate.logProbability ?? -9999,
          })
        ),
      })),
      refusal: null,
    };
  }

  private toUtf8Bytes(value?: string): number[] | null {
    return value == null ? null : Array.from(new TextEncoder().encode(value));
  }

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
      choices: new Map(),
    };
  }

  private getChoiceStreamState(index: number): ChoiceStreamState {
    let choiceState = this.streamState.choices.get(index);
    if (!choiceState) {
      choiceState = {
        toolCallCounter: 0,
        toolCallIndexes: new Map(),
      };
      this.streamState.choices.set(index, choiceState);
    }
    return choiceState;
  }

  private makeChunk(choices: OpenAI.ChatCompletionChunk.Choice[]): OpenAI.ChatCompletionChunk {
    return {
      id: this.streamState.id,
      object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000),
      model: this.streamState.model,
      choices: choices.map(choice => ({
        ...choice,
        logprobs: choice.logprobs ?? null,
      })),
    };
  }
}
