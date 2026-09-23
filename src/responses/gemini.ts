import type OpenAI from "openai";
import type {
  GenerateContentParameters,
  GenerateContentConfig,
  GenerateContentResponse,
  Content,
  Part,
  PartialArg,
  FunctionDeclaration,
  FunctionCallingConfigMode,
  FinishReason,
  Candidate,
  ThinkingConfig,
} from "@google/genai";
import { expandNamespaceTools, denamespaceResponse, denamespaceStreamEvents } from "./utils";

const MIME_BY_EXTENSION: Record<string, string> = {
  pdf: "application/pdf",
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
  gif: "image/gif",
  txt: "text/plain",
  md: "text/markdown",
  csv: "text/csv",
  html: "text/html",
  json: "application/json",
  wav: "audio/wav",
  mp3: "audio/mpeg",
  mp4: "video/mp4",
  mov: "video/quicktime",
  webm: "video/webm",
};

type RespResponse = OpenAI.Responses.Response;
type RespStreamEvent = OpenAI.Responses.ResponseStreamEvent;
const REQUEST_ECHO_FIELDS = [
  "instructions",
  "temperature",
  "top_p",
  "max_output_tokens",
  "previous_response_id",
  "parallel_tool_calls",
  "tool_choice",
  "tools",
  "text",
  "reasoning",
  "truncation",
  "top_logprobs",
  "safety_identifier",
  "service_tier",
  "background",
  "prompt_cache_key",
  "prompt_cache_retention",
] as const;
// Requests can replay SDK output messages. Audio/video retain the existing route extensions.
type MessageContentPart =
  | OpenAI.Responses.ResponseInputContent
  | OpenAI.Responses.ResponseOutputMessage["content"][number]
  | OpenAI.Responses.ResponseInputAudio
  | {
      type: "input_video";
      input_video?: { data?: string | null; url?: string | null; format?: string | null };
    };
type TerminalEvent = Extract<
  RespStreamEvent,
  { type: "response.completed" | "response.incomplete" | "response.failed" }
>;

interface TextItemContext {
  type: "message" | "reasoning";
  id: string;
  index: number;
  text: string;
  completed?: boolean;
  parts: Part[];
  annotations: OpenAI.Responses.ResponseOutputText.URLCitation[];
  logprobs: OpenAI.Responses.ResponseOutputText.Logprob[];
}
interface FunctionContext {
  type: "function_call";
  id: string;
  index: number;
  callId: string;
  name: string;
  args: Record<string, unknown>;
  pending: Set<string>;
  complete: boolean;
  arguments: string;
  parts: Part[];
}
type ItemContext = TextItemContext | FunctionContext;
interface StreamState {
  id: string;
  model: string;
  createdAt: number;
  seq: number;
  started: boolean;
  finished: boolean;
  current: TextItemContext | null;
  activeFunction: FunctionContext | null;
  items: ItemContext[];
  functions: Map<string, FunctionContext>;
  sourceParts: Array<{ ctx: ItemContext; part: Part }>;
  usage?: GenerateContentResponse["usageMetadata"];
  grounding?: Candidate["groundingMetadata"];
}

export class ResponsesToGeminiConverter {
  private streamState: StreamState;
  private requestEcho: Record<string, unknown> = {};

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (Responses → Gemini, forward) ---

  convertRequest(params: OpenAI.Responses.ResponseCreateParams): GenerateContentParameters {
    const systemParts: Part[] = [];
    const contents: Content[] = [];

    if (params.instructions) {
      systemParts.push({ text: params.instructions });
    }

    this.convertInput(systemParts, contents, params.input);

    const config: GenerateContentConfig = {};

    if (systemParts.length > 0) {
      config.systemInstruction = { parts: systemParts };
    }
    if (params.max_output_tokens != null) {
      config.maxOutputTokens = params.max_output_tokens;
    }
    if (params.temperature != null) {
      config.temperature = params.temperature;
    }
    if (params.top_p != null) {
      config.topP = params.top_p;
    }
    let requestTools = params.tools;
    let toolChoice = params.tool_choice;
    if (toolChoice && typeof toolChoice === "object" && toolChoice.type === "allowed_tools") {
      const allowed = toolChoice.tools;
      requestTools = expandNamespaceTools(requestTools)?.filter(tool =>
        allowed.some(choice =>
          choice.type === "function" && tool.type === "function"
            ? qualifiedFunctionName(choice) === tool.name
            : this.isWebSearch(choice) && this.isWebSearch(tool)
        )
      );
      toolChoice = toolChoice.mode;
    }
    if (toolChoice === "none") requestTools = [];
    if (requestTools) {
      const { tools, hasGoogleSearch } = this.convertTools(requestTools);
      if (tools.length > 0) {
        config.tools = [{ functionDeclarations: tools }];
      }
      if (hasGoogleSearch) {
        config.tools = [...(config.tools ?? []), { googleSearch: {} }];
      }
    }
    if (toolChoice != null) {
      config.toolConfig = {
        functionCallingConfig: this.convertToolChoice(toolChoice),
      };
    }
    if (params.reasoning) {
      config.thinkingConfig = this.convertReasoning(params.reasoning);
    }
    if (params.text?.format) {
      this.applyTextFormat(config, params.text.format);
    }
    if (params.include) {
      for (const inc of params.include) {
        if (inc === "message.output_text.logprobs") {
          config.responseLogprobs = true;
          config.logprobs = (params as any).top_logprobs ?? 20;
          break;
        }
      }
    }

    // Snapshot only response metadata; do not retain input content or the request body.
    this.requestEcho = structuredClone(
      Object.fromEntries(
        REQUEST_ECHO_FIELDS.map(key => [key, (params as any)[key]]).filter(
          ([, value]) => value !== undefined
        )
      )
    );

    return {
      model: params.model as string,
      contents,
      config,
    };
  }

  // --- Response conversion (Gemini → Responses, backward) ---

  convertResponse(response: GenerateContentResponse): RespResponse {
    // Use the same item assembler for both response modes, without consuming this instance's stream.
    const converter = new ResponsesToGeminiConverter();
    converter.requestEcho = this.requestEcho;
    const result = converter.consumeChunk(response) ?? converter.finishResponse("completed", "");
    if (result.status === "completed" && result.output.length === 0) {
      result.output.push({
        type: "message",
        id: `msg_${this.generateId()}`,
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "", annotations: [], logprobs: null as any }],
      });
    }
    result.usage = converter.convertUsage(response.usageMetadata);
    denamespaceResponse(result);
    return result;
  }

  // --- Stream conversion (Gemini → Responses, backward) ---

  /** Wait for the stream tail before emitting the terminal response, including late usage. */
  async *convertStream(
    stream: AsyncIterable<GenerateContentResponse>
  ): AsyncIterable<RespStreamEvent> {
    let terminal: TerminalEvent | undefined;
    for await (const chunk of stream) {
      for (const event of this.convertStreamChunk(chunk)) {
        if (
          event.type === "response.completed" ||
          event.type === "response.incomplete" ||
          event.type === "response.failed"
        ) {
          terminal = event;
        } else {
          yield event;
        }
      }
    }
    if (terminal) {
      if (this.streamState.usage) {
        terminal.response.usage = this.convertUsage(this.streamState.usage);
      }
      yield terminal;
    } else if (this.streamState.started) {
      // A clean EOF without finishReason must not report a successful response.
      const events: RespStreamEvent[] = [];
      this.finishResponse("failed", "Stream ended without finishReason", events);
      yield* denamespaceStreamEvents(events);
    }
  }

  /**
   * Convert one chunk, emitting the terminal event at finishReason for existing callers.
   * Use convertStream when the provider can send usage after finishReason.
   */
  convertStreamChunk(chunk: GenerateContentResponse): RespStreamEvent[] {
    const events: RespStreamEvent[] = [];
    this.consumeChunk(chunk, events);
    return denamespaceStreamEvents(events);
  }

  // Both modes share assembly; omitting events also skips constructing event payloads.
  private consumeChunk(
    chunk: GenerateContentResponse,
    events?: RespStreamEvent[]
  ): RespResponse | undefined {
    const state = this.streamState;
    const candidate = chunk.candidates?.[0];
    const parts = candidate?.content?.parts ?? [];
    if (chunk.usageMetadata) state.usage = { ...state.usage, ...chunk.usageMetadata };
    if (state.finished) return;
    if (chunk.modelVersion) state.model = chunk.modelVersion;
    if (candidate?.groundingMetadata)
      state.grounding = { ...state.grounding, ...candidate.groundingMetadata };
    if (!state.started) {
      state.started = true;
      state.id = chunk.responseId ?? `resp_${this.generateId()}`;
      events?.push(
        {
          type: "response.created",
          response: this.makeSkeletonResponse(),
          sequence_number: state.seq++,
        },
        {
          type: "response.in_progress",
          response: this.makeSkeletonResponse(),
          sequence_number: state.seq++,
        }
      );
    }
    const logprobs = geminiLogprobs(candidate);
    let logprobIndex = 0;
    try {
      for (const [partIndex, part] of parts.entries()) {
        if (part.functionCall) {
          if (events && state.current?.type === "reasoning")
            this.finishReasoning(state.current, "completed", events);
          state.current = null;
          this.convertFunctionPart(part, events);
        } else if (part.text != null && part.text !== "") {
          const type = part.thought ? "reasoning" : "message";
          if (state.current?.type !== type) {
            if (events && state.current?.type === "reasoning")
              this.finishReasoning(state.current, "completed", events);
            const ctx: TextItemContext = {
              type,
              id: `${type === "reasoning" ? "rs" : "msg"}_${this.generateId()}`,
              index: state.items.length,
              text: "",
              parts: [],
              annotations: [],
              logprobs: [],
            };
            state.items.push(ctx);
            state.current = ctx;
            events?.push({
              type: "response.output_item.added",
              output_index: ctx.index,
              sequence_number: state.seq++,
              item:
                type === "reasoning"
                  ? { type: "reasoning", id: ctx.id, summary: [], status: "in_progress" }
                  : {
                      type: "message",
                      id: ctx.id,
                      role: "assistant",
                      status: "in_progress",
                      content: [],
                    },
            });
            events?.push(
              type === "reasoning"
                ? {
                    type: "response.reasoning_summary_part.added",
                    item_id: ctx.id,
                    output_index: ctx.index,
                    summary_index: 0,
                    part: { type: "summary_text", text: "" },
                    sequence_number: state.seq++,
                  }
                : {
                    type: "response.content_part.added",
                    item_id: ctx.id,
                    output_index: ctx.index,
                    content_index: 0,
                    part: { type: "output_text", text: "", annotations: [], logprobs: [] },
                    sequence_number: state.seq++,
                  }
            );
          }
          const ctx = state.current;
          const last = state.sourceParts.at(-1);
          // The first text part of the next chunk can continue the preceding Gemini Part.
          if (
            partIndex === 0 &&
            last?.ctx === ctx &&
            last.part.text != null &&
            !last.part.thoughtSignature
          ) {
            last.part.text += part.text;
            if (part.thoughtSignature) last.part.thoughtSignature = part.thoughtSignature;
          } else {
            const saved = structuredClone(part);
            ctx.parts.push(saved);
            state.sourceParts.push({ ctx, part: saved });
          }
          ctx.text += part.text;
          if (type === "reasoning")
            events?.push({
              type: "response.reasoning_summary_text.delta",
              item_id: ctx.id,
              output_index: ctx.index,
              summary_index: 0,
              delta: part.text,
              sequence_number: state.seq++,
            });
          else {
            const tokenLogprobs: typeof logprobs = [];
            let tokenText = "";
            while (logprobIndex < logprobs.length && tokenText.length < part.text.length) {
              const value = logprobs[logprobIndex++];
              tokenText += value.token;
              tokenLogprobs.push(value);
            }
            ctx.logprobs.push(...tokenLogprobs);
            events?.push({
              type: "response.output_text.delta",
              item_id: ctx.id,
              output_index: ctx.index,
              content_index: 0,
              delta: part.text,
              logprobs: tokenLogprobs,
              sequence_number: state.seq++,
            });
          }
        } else if (part.thoughtSignature) {
          const last = state.sourceParts.at(-1);
          if (last) last.part.thoughtSignature = part.thoughtSignature;
        }
      }
      if (!parts.some(part => part.text && !part.thought) && logprobs.length) {
        const lastMessage = [...state.items]
          .reverse()
          .find((item): item is TextItemContext => item.type === "message");
        lastMessage?.logprobs.push(...logprobs);
      }
    } catch (error) {
      return this.finishResponse(
        "failed",
        error instanceof Error ? error.message : String(error),
        events
      );
    }
    if (chunk.promptFeedback?.blockReason)
      return this.finishResponse(
        "failed",
        `Gemini blocked the prompt: ${chunk.promptFeedback.blockReason}`,
        events
      );
    else if (candidate?.finishReason && candidate.finishReason !== "FINISH_REASON_UNSPECIFIED")
      return this.finishResponse(
        this.finishReasonToStatus(candidate.finishReason),
        `Gemini finished with ${candidate.finishReason}`,
        events
      );
  }

  private convertFunctionPart(part: Part, events?: RespStreamEvent[]): void {
    const state = this.streamState;
    const fc = part.functionCall!;
    let ctx = fc.id ? state.functions.get(fc.id) : undefined;
    if (
      !ctx &&
      !fc.id &&
      state.activeFunction &&
      !state.activeFunction.complete &&
      (!fc.name || state.activeFunction.name === fc.name)
    )
      ctx = state.activeFunction;
    if (!ctx) {
      ctx = {
        type: "function_call",
        id: `fc_${this.generateId()}`,
        index: state.items.length,
        callId: fc.id ?? `call_${this.generateId()}`,
        name: fc.name ?? "",
        args: {},
        pending: new Set(),
        complete: false,
        arguments: "",
        parts: [],
      };
      const saved: Part = { functionCall: { id: ctx.callId, name: ctx.name, args: ctx.args } };
      ctx.parts.push(saved);
      state.sourceParts.push({ ctx, part: saved });
      state.items.push(ctx);
      if (fc.id) state.functions.set(fc.id, ctx);
      events?.push({
        type: "response.output_item.added",
        output_index: ctx.index,
        sequence_number: state.seq++,
        item: {
          type: "function_call",
          id: ctx.id,
          call_id: ctx.callId,
          name: ctx.name,
          arguments: "",
          status: "in_progress",
        },
      });
    }
    if (part.thoughtSignature) ctx.parts[0].thoughtSignature = part.thoughtSignature;
    // Keep the original first-call-wins behavior for repeated completed call IDs.
    if (ctx.complete) return;
    state.activeFunction = ctx;
    if (fc.args != null) {
      ctx.args = structuredClone(fc.args);
    }
    if (fc.partialArgs) applyPartialArgs(ctx.args, ctx.pending, fc.partialArgs);
    ctx.parts[0].functionCall!.args = ctx.args;
    if (fc.willContinue !== true && ctx.pending.size === 0) {
      ctx.complete = true;
      ctx.arguments = JSON.stringify(ctx.args);
      events?.push({
        type: "response.function_call_arguments.delta",
        item_id: ctx.id,
        output_index: ctx.index,
        delta: ctx.arguments,
        sequence_number: state.seq++,
      });
      if (events) this.finishFunction(ctx, events);
    }
    return;
  }

  // --- Private: request helpers ---

  private convertInput(
    systemParts: Part[],
    contents: Content[],
    input: OpenAI.Responses.ResponseCreateParams["input"]
  ): void {
    if (typeof input === "string") {
      contents.push({ role: "user", parts: [{ text: input }] });
      return;
    }

    const signed = unpackGeminiSignatures(input);
    const calls = new Map<string, string>();
    const append = (role: "user" | "model", parts: Part[]) => {
      if (!parts.length) return;
      const last = contents.at(-1);
      const previous = last?.parts?.at(-1);
      const signedTurn =
        role === "model" &&
        (parts.some(part => part.thoughtSignature) ||
          last?.parts?.some(part => part.thoughtSignature));
      // Preserve message boundaries; only group tool exchanges and signed native turns.
      if (
        last?.role === role &&
        (signedTurn ||
          (previous?.functionCall && parts[0].functionCall) ||
          (previous?.functionResponse && parts[0].functionResponse))
      )
        last.parts!.push(...parts);
      else contents.push({ role, parts });
    };
    for (const item of input ?? []) {
      const typed = item as any;
      const signedParts = signed.get(typed.id);
      if (typed.type === "reasoning") {
        if (signedParts) append("model", signedParts);
      } else if (typed.type === "message" || (typed.type == null && typed.role)) {
        if (typed.role === "system" || typed.role === "developer") {
          const text = this.extractText(typed.content);
          if (text) systemParts.push({ text });
        } else if (typed.role === "user" || typed.role === "assistant") {
          const parts = this.convertMessageContent(typed.content, typed.role === "user");
          const original =
            typed.role === "assistant" &&
            signedParts &&
            signedParts.map(part => part.text ?? "").join("") ===
              parts.map(part => part.text ?? "").join("");
          append(typed.role === "assistant" ? "model" : "user", original ? signedParts : parts);
        }
      } else if (typed.type === "function_call") {
        const name = qualifiedFunctionName(typed);
        let args: Record<string, unknown>;
        try {
          args = JSON.parse(typed.arguments);
        } catch {
          args = {};
        }
        calls.set(typed.call_id, name);
        const part: Part = { functionCall: { id: typed.call_id, name, args } };
        const original = signedParts?.[0]?.functionCall;
        if (
          signedParts?.length === 1 &&
          original?.id === typed.call_id &&
          original?.name === name &&
          sameJson(original?.args ?? {}, args)
        ) {
          part.functionCall = structuredClone(original);
          part.thoughtSignature = signedParts[0].thoughtSignature;
        }
        append("model", [part]);
      } else if (typed.type === "function_call_output") {
        const output = typed.output ?? "";
        const text =
          Array.isArray(output) && output.every(part => part.type === "input_text")
            ? output.map(part => part.text).join("")
            : output;
        append("user", [
          {
            functionResponse: {
              id: typed.call_id,
              name: calls.get(typed.call_id) ?? typed.call_id,
              response: { output: text },
            },
          },
        ]);
      }
    }
  }

  private extractText(content: any): string {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .filter((p: any) => p.type === "input_text")
        .map((p: any) => p.text)
        .join("\n");
    }
    return "";
  }

  private convertMessageContent(
    content: string | MessageContentPart[],
    emptyFallback = true
  ): Part[] {
    if (typeof content === "string") {
      return [{ text: content }];
    }
    if (!Array.isArray(content)) return emptyFallback ? [{ text: "" }] : [];

    const parts: Part[] = [];
    for (const p of content) {
      if (p.type === "input_text" || p.type === "output_text") {
        parts.push({ text: p.text });
      } else if (p.type === "refusal") {
        parts.push({ text: p.refusal });
      } else if (p.type === "input_image") {
        const url: string = p.image_url || "";
        const dataUrl = url.match(/^data:([^;,]+);base64,([\s\S]+)$/);
        if (dataUrl) {
          parts.push({ inlineData: { mimeType: dataUrl[1], data: dataUrl[2] } });
        } else {
          const mimeType =
            MIME_BY_EXTENSION[url.split(/[?#]/)[0].split(".").pop()?.toLowerCase() ?? ""] ??
            "image/*";
          parts.push({ fileData: { fileUri: url, mimeType } });
        }
      } else if (p.type === "input_file") {
        const data: string = p.file_data || p.file_url || "";
        const dataUrl = data.match(/^data:([^;,]+);base64,([\s\S]+)$/);
        if (dataUrl) {
          parts.push({ inlineData: { mimeType: dataUrl[1], data: dataUrl[2] } });
        } else {
          const filename = typeof p.filename === "string" ? p.filename : "";
          const mimeType =
            MIME_BY_EXTENSION[filename.split(/[?#]/)[0].split(".").pop()?.toLowerCase() ?? ""] ??
            MIME_BY_EXTENSION[data.split(/[?#]/)[0].split(".").pop()?.toLowerCase() ?? ""] ??
            "application/octet-stream";
          if (p.file_data && /^[A-Za-z0-9+/]+={0,2}$/.test(data.replace(/\s/g, ""))) {
            parts.push({ inlineData: { mimeType, data: data.replace(/\s/g, "") } });
          } else {
            parts.push({ fileData: { fileUri: data, mimeType } });
          }
        }
      } else if (p.type === "input_audio") {
        const audio = p.input_audio;
        parts.push({
          inlineData: {
            mimeType: audio?.format === "mp3" ? "audio/mp3" : "audio/wav",
            data: audio?.data,
          },
        });
      } else if (p.type === "input_video") {
        const video = p.input_video;
        if (video?.data) {
          parts.push({
            inlineData: { mimeType: `video/${video.format ?? "mp4"}`, data: video.data },
          });
        } else {
          const url: string = video?.url || "";
          const dataUrl = url.match(/^data:(video[^;]*);base64,([\s\S]*)$/);
          if (!url) {
            parts.push({ inlineData: {} });
          } else if (dataUrl) {
            parts.push({ inlineData: { mimeType: dataUrl[1], data: dataUrl[2] } });
          } else {
            parts.push({ fileData: { fileUri: url, mimeType: "video/mp4" } });
          }
        }
      }
    }
    return parts.length > 0 || !emptyFallback ? parts : [{ text: "" }];
  }

  private convertTools(tools: OpenAI.Responses.ResponseCreateParams["tools"]): {
    tools: FunctionDeclaration[];
    hasGoogleSearch: boolean;
  } {
    tools = expandNamespaceTools(tools);
    if (!tools) return { tools: [], hasGoogleSearch: false };

    const functionTools: FunctionDeclaration[] = [];
    let hasGoogleSearch = false;

    for (const t of tools) {
      const tt = t as any;
      if (tt.type === "function") {
        functionTools.push({
          name: tt.name,
          description: tt.description,
          parametersJsonSchema: tt.parameters,
        });
      } else if (this.isWebSearch(tt)) {
        hasGoogleSearch = true;
      }
    }

    return { tools: functionTools, hasGoogleSearch };
  }

  private isWebSearch(tool: any): boolean {
    const t = tool.type;
    return (
      t === "web_search" ||
      t === "web_search_2025_08_26" ||
      t === "web_search_preview" ||
      t === "web_search_preview_2025_03_11"
    );
  }

  private convertToolChoice(choice: OpenAI.Responses.ResponseCreateParams["tool_choice"]): {
    mode?: FunctionCallingConfigMode;
    allowedFunctionNames?: string[];
  } {
    if (typeof choice === "string") {
      switch (choice) {
        case "auto":
          return { mode: "AUTO" as FunctionCallingConfigMode };
        case "required":
          return { mode: "ANY" as FunctionCallingConfigMode };
        case "none":
          return { mode: "NONE" as FunctionCallingConfigMode };
        default:
          return { mode: "AUTO" as FunctionCallingConfigMode };
      }
    }
    if (typeof choice === "object" && choice !== null) {
      if (choice.type === "function" && choice.name) {
        return {
          mode: "ANY" as FunctionCallingConfigMode,
          allowedFunctionNames: [qualifiedFunctionName(choice)],
        };
      }
    }
    return { mode: "AUTO" as FunctionCallingConfigMode };
  }

  private convertReasoning(
    reasoning: NonNullable<OpenAI.Responses.ResponseCreateParams["reasoning"]>
  ): ThinkingConfig {
    const effort = reasoning.effort;
    if (!effort?.trim()) return {};
    if (effort === "none") return { includeThoughts: false, thinkingBudget: 0 };
    const budgets: Record<string, number> = {
      minimal: 128,
      low: 2048,
      medium: 5120,
      high: 10240,
    };
    return { includeThoughts: true, thinkingBudget: budgets[effort] ?? 10240 };
  }

  private applyTextFormat(config: GenerateContentConfig, format: any): void {
    if (format.type === "json_schema") {
      config.responseMimeType = "application/json";
      config.responseJsonSchema = format.schema;
    } else if (format.type === "json_object") {
      config.responseMimeType = "application/json";
    }
  }

  // --- Private: response helpers ---

  private finishReasonToStatus(
    reason: FinishReason | string | null | undefined
  ): RespResponse["status"] {
    switch (reason) {
      case "STOP":
        return "completed";
      case "MAX_TOKENS":
        return "incomplete";
      case "SAFETY":
      case "RECITATION":
      case "BLOCKLIST":
      case "PROHIBITED_CONTENT":
      case "SPII":
      case "MALFORMED_FUNCTION_CALL":
      case "UNEXPECTED_TOOL_CALL":
      case "LANGUAGE":
      case "OTHER":
      case "IMAGE_SAFETY":
      case "IMAGE_PROHIBITED_CONTENT":
      case "IMAGE_RECITATION":
      case "IMAGE_OTHER":
      case "NO_IMAGE":
        return "failed";
      default:
        return "completed";
    }
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  // --- Private: stream helpers ---

  private convertUsage(
    usage: GenerateContentResponse["usageMetadata"]
  ): NonNullable<RespResponse["usage"]> {
    const inputTokens = (usage?.promptTokenCount ?? 0) + (usage?.toolUsePromptTokenCount ?? 0);
    const reasoningTokens = usage?.thoughtsTokenCount ?? 0;
    const outputTokens = (usage?.candidatesTokenCount ?? 0) + reasoningTokens;
    return {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      total_tokens: usage?.totalTokenCount ?? 0,
      input_tokens_details: { cached_tokens: usage?.cachedContentTokenCount ?? 0 },
      output_tokens_details: { reasoning_tokens: reasoningTokens },
    };
  }

  private applyGrounding(): void {
    const state = this.streamState;
    const metadata = state.grounding;
    if (!metadata) return;
    const supports = metadata.groundingSupports ?? [];
    const add = (ctx: TextItemContext, uri: string, title: string, start: number, end: number) => {
      const annotation = {
        type: "url_citation" as const,
        url: uri,
        title,
        start_index: start,
        end_index: end,
      };
      if (
        !ctx.annotations.some(existing => JSON.stringify(existing) === JSON.stringify(annotation))
      )
        ctx.annotations.push(annotation);
    };
    for (const support of supports) {
      const source = state.sourceParts[support.segment?.partIndex ?? 0];
      if (!source || source.ctx.type !== "message" || source.part.text == null) continue;
      const ctx = source.ctx;
      let base = 0;
      for (const part of ctx.parts) {
        if (part === source.part) break;
        base += Array.from(part.text ?? "").length;
      }
      const segment = support.segment;
      let start =
        segment?.startIndex == null
          ? 0
          : byteToCharacterOffset(source.part.text, segment.startIndex);
      let end =
        segment?.endIndex == null
          ? Array.from(source.part.text).length
          : byteToCharacterOffset(source.part.text, segment.endIndex);
      if (segment?.text && segment.startIndex == null) {
        const offset = source.part.text.indexOf(segment.text);
        if (offset < 0) continue;
        start = Array.from(source.part.text.slice(0, offset)).length;
        end = start + Array.from(segment.text).length;
      }
      for (const index of support.groundingChunkIndices ?? []) {
        const web = metadata.groundingChunks?.[index]?.web;
        if (web?.uri) add(ctx, web.uri, web.title ?? "", base + start, base + end);
      }
    }
    // Older providers supply sources without spans; associate them with the full answer.
    if (!supports.length) {
      const ctx = state.items.find((item): item is TextItemContext => item.type === "message");
      if (ctx)
        for (const source of metadata.groundingChunks ?? []) {
          if (source.web?.uri)
            add(ctx, source.web.uri, source.web.title ?? "", 0, Array.from(ctx.text).length);
        }
    }
  }

  private finishFunction(
    ctx: FunctionContext,
    events?: RespStreamEvent[]
  ): OpenAI.Responses.ResponseFunctionToolCall {
    const item: OpenAI.Responses.ResponseFunctionToolCall = {
      type: "function_call",
      id: ctx.id,
      call_id: ctx.callId,
      name: ctx.name,
      arguments: ctx.arguments,
      status: ctx.complete ? "completed" : "incomplete",
    };
    events?.push(
      {
        type: "response.function_call_arguments.done",
        item_id: ctx.id,
        output_index: ctx.index,
        name: ctx.name,
        arguments: ctx.arguments,
        sequence_number: this.streamState.seq++,
      },
      {
        type: "response.output_item.done",
        item: { ...item },
        output_index: ctx.index,
        sequence_number: this.streamState.seq++,
      }
    );
    return item;
  }

  private finishReasoning(
    ctx: TextItemContext,
    status: "completed" | "incomplete",
    events?: RespStreamEvent[]
  ): OpenAI.Responses.ResponseReasoningItem {
    const part = { type: "summary_text" as const, text: ctx.text };
    const item: OpenAI.Responses.ResponseReasoningItem = {
      type: "reasoning",
      id: ctx.id,
      summary: [part],
      status: ctx.completed ? "completed" : status,
    };
    // Keep an already completed reasoning item stable and do not emit done events twice.
    if (!ctx.completed) {
      events?.push(
        {
          type: "response.reasoning_summary_text.done",
          item_id: ctx.id,
          output_index: ctx.index,
          summary_index: 0,
          text: ctx.text,
          sequence_number: this.streamState.seq++,
        },
        {
          type: "response.reasoning_summary_part.done",
          item_id: ctx.id,
          output_index: ctx.index,
          summary_index: 0,
          part: { ...part },
          sequence_number: this.streamState.seq++,
        },
        {
          type: "response.output_item.done",
          item: structuredClone(item),
          output_index: ctx.index,
          sequence_number: this.streamState.seq++,
        }
      );
      ctx.completed = item.status === "completed";
    }
    return item;
  }

  private finishResponse(
    status: RespResponse["status"],
    failureMessage: string,
    events?: RespStreamEvent[]
  ): RespResponse {
    const state = this.streamState;
    state.finished = true;
    if (
      status === "completed" &&
      state.items.some(item => item.type === "function_call" && !item.complete)
    ) {
      status = "failed";
      failureMessage = "Gemini finished with incomplete function arguments";
    }
    this.applyGrounding();
    const output: RespResponse["output"] = [];
    for (const ctx of state.items) {
      let item: OpenAI.Responses.ResponseOutputItem;
      if (ctx.type === "function_call") {
        if (!ctx.complete) {
          ctx.arguments = JSON.stringify(ctx.args);
          events?.push({
            type: "response.function_call_arguments.delta",
            item_id: ctx.id,
            output_index: ctx.index,
            delta: ctx.arguments,
            sequence_number: state.seq++,
          });
        }
        output.push(this.finishFunction(ctx, ctx.complete ? undefined : events));
        continue;
      } else if (ctx.type === "reasoning") {
        output.push(
          this.finishReasoning(ctx, status === "completed" ? "completed" : "incomplete", events)
        );
        continue;
      } else {
        const part: OpenAI.Responses.ResponseOutputText = {
          type: "output_text",
          text: ctx.text,
          annotations: ctx.annotations,
          logprobs: events || ctx.logprobs.length ? ctx.logprobs : (null as any),
        };
        item = {
          type: "message",
          id: ctx.id,
          role: "assistant",
          status: status === "completed" ? "completed" : "incomplete",
          content: [part],
        };
        ctx.annotations.forEach((annotation, annotationIndex) =>
          events?.push({
            type: "response.output_text.annotation.added",
            item_id: ctx.id,
            output_index: ctx.index,
            content_index: 0,
            annotation_index: annotationIndex,
            annotation: structuredClone(annotation),
            sequence_number: state.seq++,
          })
        );
        events?.push(
          {
            type: "response.output_text.done",
            item_id: ctx.id,
            output_index: ctx.index,
            content_index: 0,
            text: ctx.text,
            logprobs: ctx.logprobs,
            sequence_number: state.seq++,
          },
          {
            type: "response.content_part.done",
            item_id: ctx.id,
            output_index: ctx.index,
            content_index: 0,
            part: structuredClone(part),
            sequence_number: state.seq++,
          }
        );
      }
      output.push(item);
      events?.push({
        type: "response.output_item.done",
        item: structuredClone(item),
        output_index: ctx.index,
        sequence_number: state.seq++,
      });
    }
    const signed = state.items
      .filter(ctx => ctx.parts.some(part => part.thoughtSignature))
      .map(ctx => ({ id: ctx.id, parts: ctx.parts }));
    if (signed.length) {
      const id = `rs_${this.generateId()}`;
      const index = output.length;
      const item: OpenAI.Responses.ResponseReasoningItem = {
        type: "reasoning",
        id,
        summary: [],
        status: "completed",
        encrypted_content: packGeminiSignatures(signed),
      };
      events?.push(
        {
          type: "response.output_item.added",
          item: { type: "reasoning", id, summary: [], status: "in_progress" },
          output_index: index,
          sequence_number: state.seq++,
        },
        {
          type: "response.output_item.done",
          item: structuredClone(item),
          output_index: index,
          sequence_number: state.seq++,
        }
      );
      output.push(item);
    }
    const response = this.makeSkeletonResponse();
    response.status = status;
    response.output = structuredClone(output);
    if (state.usage) response.usage = this.convertUsage(state.usage);
    if (status === "incomplete") response.incomplete_details = { reason: "max_output_tokens" };
    if (status === "failed")
      response.error = {
        code: failureMessage.includes("blocked the prompt") ? "invalid_prompt" : "server_error",
        message: failureMessage,
      };
    events?.push({
      type:
        status === "failed"
          ? "response.failed"
          : status === "incomplete"
            ? "response.incomplete"
            : "response.completed",
      response,
      sequence_number: state.seq++,
    });
    return response;
  }

  private createStreamState(): StreamState {
    return {
      id: "",
      model: "",
      createdAt: Math.floor(Date.now() / 1000),
      seq: 0,
      started: false,
      finished: false,
      current: null,
      activeFunction: null,
      items: [],
      functions: new Map(),
      sourceParts: [],
    };
  }

  private makeSkeletonResponse(): RespResponse {
    return {
      id: this.streamState.id,
      object: "response",
      created_at: this.streamState.createdAt,
      model: this.streamState.model,
      output: [],
      status: "in_progress",
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
      truncation: null,
      user: undefined,
      ...structuredClone(this.requestEcho),
    } as unknown as RespResponse;
  }
}

function isObject(value: unknown): value is Record<string, any> {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function qualifiedFunctionName(tool: { name?: string; namespace?: string }): string {
  return tool.namespace ? `${tool.namespace}___${tool.name}` : tool.name!;
}

function sameJson(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b))
    return a.length === b.length && a.every((value, index) => sameJson(value, b[index]));
  if (!isObject(a) || !isObject(b)) return false;
  const keys = Object.keys(a);
  return (
    keys.length === Object.keys(b).length &&
    keys.every(key => Object.hasOwn(b, key) && sameJson(a[key], b[key]))
  );
}

const SIGNATURE_PREFIX = "rosetta:gemini:signatures:v1:";
interface SignedItem {
  id: string;
  parts: Part[];
}

/** This is an opaque transport envelope, not locally encrypted data. Signatures remain provider-owned. */
function packGeminiSignatures(items: SignedItem[]): string {
  return SIGNATURE_PREFIX + Buffer.from(JSON.stringify(items), "utf8").toString("base64url");
}

function unpackGeminiSignatures(input: unknown): Map<string, Part[]> {
  const result = new Map<string, Part[]>();
  if (!Array.isArray(input)) return result;
  for (const item of input) {
    if (
      item?.type !== "reasoning" ||
      typeof item.encrypted_content !== "string" ||
      !item.encrypted_content.startsWith(SIGNATURE_PREFIX)
    )
      continue;
    try {
      const entries: unknown = JSON.parse(
        Buffer.from(item.encrypted_content.slice(SIGNATURE_PREFIX.length), "base64url").toString(
          "utf8"
        )
      );
      if (!Array.isArray(entries)) continue;
      for (const entry of entries) {
        if (
          !isObject(entry) ||
          typeof entry.id !== "string" ||
          !Array.isArray(entry.parts) ||
          !entry.parts.length
        )
          continue;
        if (
          !entry.parts.every(
            (p: unknown) =>
              isObject(p) &&
              (typeof p.text === "string" || isObject(p.functionCall)) &&
              (p.thoughtSignature == null || typeof p.thoughtSignature === "string")
          )
        )
          continue;
        if (!entry.parts.some((p: Part) => p.thoughtSignature)) continue;
        result.set(entry.id, structuredClone(entry.parts));
      }
    } catch {
      // An unreadable envelope cannot restore signatures; retain the ordinary input history.
    }
  }
  return result;
}

/** Parse only concrete JSONPath members; no evaluation, wildcards or inherited properties. */
function pathKeys(path: string): Array<string | number> {
  if (!path.startsWith("$")) return [];
  const keys: Array<string | number> = [];
  let rest = path.slice(1);
  while (rest) {
    const dot = rest.match(/^\.([A-Za-z_$][\w$]*)/);
    const index = rest.match(/^\[(0|[1-9]\d*)\]/);
    const quoted = rest.match(/^\[("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\]/);
    if (dot) {
      keys.push(dot[1]);
      rest = rest.slice(dot[0].length);
    } else if (index) {
      const value = Number(index[1]);
      if (!Number.isSafeInteger(value) || value > 100000) return [];
      keys.push(value);
      rest = rest.slice(index[0].length);
    } else if (quoted) {
      try {
        const value = quoted[1].startsWith('"')
          ? JSON.parse(quoted[1])
          : JSON.parse(
              '"' +
                quoted[1]
                  .slice(1, -1)
                  .replace(/\\.|"/g, escape =>
                    escape === "\\'" ? "'" : escape === '"' ? '\\"' : escape
                  ) +
                '"'
            );
        keys.push(value);
      } catch {
        return [];
      }
      rest = rest.slice(quoted[0].length);
    } else return [];
  }
  return keys;
}

function applyPartialArgs(
  args: Record<string, unknown>,
  pending: Set<string>,
  parts: PartialArg[]
): void {
  for (const part of parts) {
    if (typeof part.jsonPath !== "string" || !part.jsonPath) continue;
    const keys = pathKeys(part.jsonPath);
    if (!keys?.length) continue;
    const fields = ["stringValue", "numberValue", "boolValue", "nullValue"].filter(field =>
      Object.hasOwn(part, field)
    );
    if (fields.length !== 1) continue;
    const path = JSON.stringify(keys);
    let target: any = args;
    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      if (!Object.hasOwn(target, key))
        Object.defineProperty(target, key, {
          value: typeof keys[i + 1] === "number" ? [] : {},
          enumerable: true,
          writable: true,
          configurable: true,
        });
      if (target[key] == null || typeof target[key] !== "object") {
        target = null;
        break;
      }
      target = target[key];
    }
    if (target == null) continue;
    const key = keys[keys.length - 1];
    let value: unknown = part.stringValue ?? part.numberValue ?? part.boolValue ?? null;
    if (pending.has(path)) {
      if (
        typeof value !== "string" ||
        !Object.hasOwn(target, key) ||
        typeof target[key] !== "string"
      )
        continue;
      value = target[key] + value;
    }
    Object.defineProperty(target, key, {
      value,
      enumerable: true,
      writable: true,
      configurable: true,
    });
    if (part.willContinue) pending.add(path);
    else pending.delete(path);
  }
}

function geminiLogprobs(candidate?: Candidate): OpenAI.Responses.ResponseOutputText.Logprob[] {
  return (candidate?.logprobsResult?.chosenCandidates ?? []).map((chosen, index) => ({
    token: chosen.token ?? "",
    bytes: Array.from(new TextEncoder().encode(chosen.token ?? "")),
    logprob: chosen.logProbability ?? -9999,
    top_logprobs: (candidate?.logprobsResult?.topCandidates?.[index]?.candidates ?? []).map(
      top => ({
        token: top.token ?? "",
        bytes: Array.from(new TextEncoder().encode(top.token ?? "")),
        logprob: top.logProbability ?? -9999,
      })
    ),
  }));
}

/** Gemini offsets are UTF-8 bytes; Responses annotations use character offsets. */
function byteToCharacterOffset(text: string, byteOffset: number): number {
  const bytes = new TextEncoder().encode(text);
  return Array.from(new TextDecoder().decode(bytes.slice(0, Math.max(0, byteOffset)))).length;
}
