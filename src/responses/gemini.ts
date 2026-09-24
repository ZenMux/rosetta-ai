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
  Candidate,
  ThinkingConfig,
} from "@google/genai";
import { expandNamespaceTools, denamespaceResponse, denamespaceStreamEvents } from "./utils";

type RespResponse = OpenAI.Responses.Response;
type RespStreamEvent = OpenAI.Responses.ResponseStreamEvent;

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
}
interface FunctionContext {
  type: "function_call";
  id: string;
  index: number;
  callId: string;
  name: string;
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
  items: ItemContext[];
  functions: Map<string, FunctionContext>;
  sourceParts: Array<{ ctx: ItemContext; part: Part }>;
  usage?: GenerateContentResponse["usageMetadata"];
  grounding?: Candidate["groundingMetadata"];
}

export class ResponsesToGeminiConverter {
  private streamState: StreamState;

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
    if (params.tools) {
      const { tools, hasGoogleSearch } = this.convertTools(params.tools);
      if (tools.length > 0) {
        config.tools = [{ functionDeclarations: tools }];
      }
      if (hasGoogleSearch) {
        config.tools = [...(config.tools ?? []), { googleSearch: {} }];
      }
    }
    if (params.tool_choice != null) {
      config.toolConfig = {
        functionCallingConfig: this.convertToolChoice(params.tool_choice),
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

    return {
      model: params.model as string,
      contents,
      config,
    };
  }

  // --- Response conversion (Gemini → Responses, backward) ---

  convertResponse(response: GenerateContentResponse): RespResponse {
    const candidate = response.candidates?.[0];
    const parts = (candidate?.content?.parts ?? []).map(part => ({ ...part }));
    const output: OpenAI.Responses.ResponseOutputItem[] = [];
    const signed: SignedItem[] = [];

    // A signature-only Part belongs to the preceding text or function call.
    let previous: Part | undefined;
    for (const part of parts) {
      if (part.functionCall || part.text) previous = part;
      else if (part.thoughtSignature && previous) previous.thoughtSignature = part.thoughtSignature;
    }

    for (const part of parts) {
      if (part.thought && part.text) {
        const id = `rs_${this.generateId()}`;
        output.push({
          type: "reasoning",
          id,
          summary: [{ type: "summary_text", text: part.text }],
        });
        if (part.thoughtSignature) signed.push({ id, parts: [part] });
      }
    }

    for (const part of parts) {
      if (part.functionCall) {
        const fc = part.functionCall;
        const id = `fc_${this.generateId()}`;
        const callId = fc.id ?? `call_${this.generateId()}`;
        const args = fc.args ?? {};

        if (part.thoughtSignature) {
          signed.push({
            id,
            parts: [{ ...part, functionCall: { id: callId, name: fc.name ?? "", args } }],
          });
        }

        output.push({
          type: "function_call",
          id,
          call_id: callId,
          name: fc.name ?? "",
          arguments: JSON.stringify(args),
          status: "completed",
        });
      }
    }

    const textParts: Part[] = [];
    const annotations: OpenAI.Responses.ResponseOutputText.URLCitation[] = [];
    for (const part of parts) {
      if (part.text != null && !part.thought) {
        textParts.push(part);
      }
    }
    this.extractGroundingAnnotations(candidate, annotations);

    if (textParts.length > 0 || annotations.length > 0) {
      const id = `msg_${this.generateId()}`;
      output.push({
        type: "message",
        id,
        role: "assistant",
        status: "completed",
        content: [
          {
            type: "output_text",
            text: textParts.map(part => part.text).join(""),
            annotations,
            logprobs: [],
          },
        ],
      });
      if (textParts.some(part => part.thoughtSignature)) signed.push({ id, parts: textParts });
    }

    const blockReason = response.promptFeedback?.blockReason;
    const status = blockReason ? "failed" : this.finishReasonToStatus(candidate?.finishReason);
    const failureMessage = blockReason
      ? `Gemini blocked the prompt: ${blockReason}`
      : `Gemini finished with ${candidate?.finishReason}`;
    for (const item of output) {
      if (item.type === "message" || item.type === "reasoning")
        item.status = status === "completed" ? "completed" : "incomplete";
    }

    if (status === "completed" && output.length === 0) {
      output.push({
        type: "message",
        id: `msg_${this.generateId()}`,
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "", annotations: [], logprobs: [] }],
      });
    }

    if (signed.length) {
      output.push({
        type: "reasoning",
        id: `rs_${this.generateId()}`,
        summary: [],
        status: "completed",
        encrypted_content: packGeminiSignatures(signed),
      });
    }

    const respResult = {
      id: response.responseId ?? `resp_${this.generateId()}`,
      object: "response",
      created_at: Math.floor(Date.now() / 1000),
      model: response.modelVersion ?? "",
      output,
      status,
      error:
        status === "failed"
          ? { code: blockReason ? "invalid_prompt" : "server_error", message: failureMessage }
          : null,
      incomplete_details: status === "incomplete" ? { reason: "max_output_tokens" } : null,
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
      usage: this.convertUsage(response.usageMetadata),
    } as unknown as RespResponse;

    // Split namespaced function_call names into { namespace, name }.
    denamespaceResponse(respResult);
    return respResult;
  }

  // --- Stream conversion (Gemini → Responses, backward) ---

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
    }
  }

  convertStreamChunk(chunk: GenerateContentResponse): RespStreamEvent[] {
    const events: RespStreamEvent[] = [];
    const state = this.streamState;
    const candidate = chunk.candidates?.[0];
    const parts = candidate?.content?.parts ?? [];
    if (chunk.usageMetadata) state.usage = { ...state.usage, ...chunk.usageMetadata };
    if (state.finished) return events;
    if (chunk.modelVersion) state.model = chunk.modelVersion;
    if (candidate?.groundingMetadata)
      state.grounding = { ...state.grounding, ...candidate.groundingMetadata };
    if (!state.started) {
      state.started = true;
      state.id = chunk.responseId ?? `resp_${this.generateId()}`;
      events.push(
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
    try {
      for (const [partIndex, part] of parts.entries()) {
        if (part.functionCall) {
          if (state.current?.type === "reasoning")
            this.finishReasoning(state.current, "completed", events);
          state.current = null;
          this.convertFunctionPart(part, events);
        } else if (part.text != null && part.text !== "") {
          const type = part.thought ? "reasoning" : "message";
          if (state.current?.type !== type) {
            if (state.current?.type === "reasoning")
              this.finishReasoning(state.current, "completed", events);
            const ctx: TextItemContext = {
              type,
              id: `${type === "reasoning" ? "rs" : "msg"}_${this.generateId()}`,
              index: state.items.length,
              text: "",
              parts: [],
              annotations: [],
            };
            state.items.push(ctx);
            state.current = ctx;
            events.push({
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
            events.push(
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
            events.push({
              type: "response.reasoning_summary_text.delta",
              item_id: ctx.id,
              output_index: ctx.index,
              summary_index: 0,
              delta: part.text,
              sequence_number: state.seq++,
            });
          else {
            events.push({
              type: "response.output_text.delta",
              item_id: ctx.id,
              output_index: ctx.index,
              content_index: 0,
              delta: part.text,
              logprobs: [],
              sequence_number: state.seq++,
            });
          }
        } else if (part.thoughtSignature) {
          const last = state.sourceParts.at(-1);
          if (last) last.part.thoughtSignature = part.thoughtSignature;
        }
      }
    } catch (error) {
      this.finishResponse("failed", error instanceof Error ? error.message : String(error), events);
      return denamespaceStreamEvents(events);
    }
    if (chunk.promptFeedback?.blockReason)
      this.finishResponse(
        "failed",
        `Gemini blocked the prompt: ${chunk.promptFeedback.blockReason}`,
        events
      );
    else if (candidate?.finishReason && candidate.finishReason !== "FINISH_REASON_UNSPECIFIED")
      this.finishResponse(
        this.finishReasonToStatus(candidate.finishReason),
        `Gemini finished with ${candidate.finishReason}`,
        events
      );
    return denamespaceStreamEvents(events);
  }

  private convertFunctionPart(part: Part, events: RespStreamEvent[]): void {
    const state = this.streamState;
    const fc = part.functionCall!;
    const existing = fc.id ? state.functions.get(fc.id) : undefined;
    // Keep the original first-call-wins behavior for repeated call IDs.
    if (existing) {
      if (part.thoughtSignature) existing.parts[0].thoughtSignature = part.thoughtSignature;
      return;
    }
    const args = structuredClone(fc.args ?? {});
    const ctx: FunctionContext = {
      type: "function_call",
      id: `fc_${this.generateId()}`,
      index: state.items.length,
      callId: fc.id ?? `call_${this.generateId()}`,
      name: fc.name ?? "",
      arguments: JSON.stringify(args),
      parts: [],
    };
    const saved: Part = { functionCall: { id: ctx.callId, name: ctx.name, args } };
    if (part.thoughtSignature) saved.thoughtSignature = part.thoughtSignature;
    ctx.parts.push(saved);
    state.sourceParts.push({ ctx, part: saved });
    state.items.push(ctx);
    if (fc.id) state.functions.set(fc.id, ctx);
    events.push(
      {
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
      },
      {
        type: "response.function_call_arguments.delta",
        item_id: ctx.id,
        output_index: ctx.index,
        delta: ctx.arguments,
        sequence_number: state.seq++,
      }
    );
    this.finishFunction(ctx, events);
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
        append("user", [
          {
            functionResponse: {
              id: typed.call_id,
              name: calls.get(typed.call_id) ?? typed.call_id,
              response: { output: typed.output ?? "" },
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
        const match = url.match(/^data:([^;]+);base64,(.+)$/);
        if (match) {
          parts.push({
            inlineData: { mimeType: match[1], data: match[2] },
          });
        } else {
          parts.push({ fileData: { fileUri: url, mimeType: "image/*" } });
        }
      } else if (p.type === "input_file") {
        const data: string = p.file_data || p.file_url || "";
        const match = data.match(/^data:([^;]+);base64,(.+)$/);
        if (match) {
          parts.push({
            inlineData: { mimeType: match[1], data: match[2] },
          });
        } else {
          parts.push({
            fileData: {
              fileUri: data,
              mimeType: "application/octet-stream",
            },
          });
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
      xhigh: 20480,
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

  private extractGroundingAnnotations(
    candidate: Candidate | undefined,
    annotations: OpenAI.Responses.ResponseOutputText.URLCitation[]
  ): void {
    const metadata = candidate?.groundingMetadata;
    if (!metadata) return;
    const parts = candidate?.content?.parts ?? [];
    const supports = metadata.groundingSupports ?? [];
    for (const support of supports) {
      const segment = support.segment;
      const partIndex = segment?.partIndex ?? 0;
      const part = parts[partIndex];
      let start = 0;
      let end = 0;
      if (
        part?.text != null &&
        !part.thought &&
        segment?.startIndex != null &&
        segment.endIndex != null &&
        Number.isInteger(segment.startIndex) &&
        Number.isInteger(segment.endIndex) &&
        segment.startIndex >= 0 &&
        segment.endIndex >= segment.startIndex &&
        segment.endIndex <= new TextEncoder().encode(part.text).length
      ) {
        const base = parts
          .slice(0, partIndex)
          .reduce(
            (offset, part) => offset + (part.thought ? 0 : Array.from(part.text ?? "").length),
            0
          );
        start = base + byteToCharacterOffset(part.text, segment.startIndex);
        end = base + byteToCharacterOffset(part.text, segment.endIndex);
      }
      for (const index of support.groundingChunkIndices ?? []) {
        const web = metadata.groundingChunks?.[index]?.web;
        if (!web) continue;
        annotations.push({
          type: "url_citation",
          url: web.uri || "",
          title: web.title || "",
          start_index: start,
          end_index: end,
        });
      }
    }
    for (const [index, chunk] of (metadata.groundingChunks ?? []).entries()) {
      if (chunk.web && !supports.some(support => support.groundingChunkIndices?.includes(index)))
        annotations.push({
          type: "url_citation",
          url: chunk.web.uri || "",
          title: chunk.web.title || "",
          start_index: 0,
          end_index: 0,
        });
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
    const fallback = state.items.find((item): item is TextItemContext => item.type === "message");
    if (!fallback) return;
    const supports = metadata.groundingSupports ?? [];
    const add = (ctx: TextItemContext, uri: string, title: string, start: number, end: number) => {
      ctx.annotations.push({
        type: "url_citation" as const,
        url: uri,
        title,
        start_index: start,
        end_index: end,
      });
    };
    for (const support of supports) {
      const segment = support.segment;
      const source = state.sourceParts[segment?.partIndex ?? 0];
      const ctx = source?.ctx.type === "message" ? source.ctx : fallback;
      let start = 0;
      let end = 0;
      if (
        source?.ctx.type === "message" &&
        source.part.text != null &&
        segment?.startIndex != null &&
        segment.endIndex != null &&
        Number.isInteger(segment.startIndex) &&
        Number.isInteger(segment.endIndex) &&
        segment.startIndex >= 0 &&
        segment.endIndex >= segment.startIndex &&
        segment.endIndex <= new TextEncoder().encode(source.part.text).length
      ) {
        let base = 0;
        for (const part of ctx.parts) {
          if (part === source.part) break;
          base += Array.from(part.text ?? "").length;
        }
        start = base + byteToCharacterOffset(source.part.text, segment.startIndex);
        end = base + byteToCharacterOffset(source.part.text, segment.endIndex);
      }
      for (const index of support.groundingChunkIndices ?? []) {
        const web = metadata.groundingChunks?.[index]?.web;
        if (web) add(ctx, web.uri ?? "", web.title ?? "", start, end);
      }
    }
    for (const [index, chunk] of (metadata.groundingChunks ?? []).entries()) {
      if (chunk.web && !supports.some(support => support.groundingChunkIndices?.includes(index)))
        add(fallback, chunk.web.uri ?? "", chunk.web.title ?? "", 0, 0);
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
      status: "completed",
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
    events: RespStreamEvent[]
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
      events.push(
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
    events: RespStreamEvent[]
  ): void {
    const state = this.streamState;
    state.finished = true;
    this.applyGrounding();
    const output: RespResponse["output"] = [];
    for (const ctx of state.items) {
      let item: OpenAI.Responses.ResponseOutputItem;
      if (ctx.type === "function_call") {
        output.push(this.finishFunction(ctx));
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
          logprobs: [],
        };
        item = {
          type: "message",
          id: ctx.id,
          role: "assistant",
          status: status === "completed" ? "completed" : "incomplete",
          content: [part],
        };
        ctx.annotations.forEach((annotation, annotationIndex) =>
          events.push({
            type: "response.output_text.annotation.added",
            item_id: ctx.id,
            output_index: ctx.index,
            content_index: 0,
            annotation_index: annotationIndex,
            annotation: structuredClone(annotation),
            sequence_number: state.seq++,
          })
        );
        events.push(
          {
            type: "response.output_text.done",
            item_id: ctx.id,
            output_index: ctx.index,
            content_index: 0,
            text: ctx.text,
            logprobs: [],
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
      events.push({
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
      events.push(
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
    events.push({
      type:
        status === "failed"
          ? "response.failed"
          : status === "incomplete"
            ? "response.incomplete"
            : "response.completed",
      response,
      sequence_number: state.seq++,
    });
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

/** Gemini offsets are UTF-8 bytes; Responses annotations use character offsets. */
function byteToCharacterOffset(text: string, byteOffset: number): number {
  const bytes = new TextEncoder().encode(text);
  return Array.from(new TextDecoder().decode(bytes.slice(0, Math.max(0, byteOffset)))).length;
}
