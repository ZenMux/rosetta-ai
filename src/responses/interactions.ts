import type OpenAI from "openai";
import {
  InteractionsResponseError,
  type Interaction,
  type InteractionAnnotation,
  type InteractionCreateParams,
  type InteractionEvent,
  type InteractionStep,
  type InteractionUsage,
} from "../interactions/types";
import {
  convertInteractionResponseUsage,
  type InteractionResponseUsage,
} from "../interactions/usage";
import { convertResponsesInteractionRequest } from "./interactions-request";

type Item = OpenAI.Responses.ResponseOutputItem;
export type InteractionResponse = Omit<
  OpenAI.Responses.Response,
  "usage" | "tools" | "tool_choice"
> &
  Partial<Pick<OpenAI.Responses.Response, "tools" | "tool_choice">> & {
    usage: InteractionResponseUsage | null;
  };
type ReplaceResponse<T> = T extends { response: OpenAI.Responses.Response }
  ? Omit<T, "response"> & { response: InteractionResponse }
  : T;
export type InteractionResponseStreamEvent = ReplaceResponse<OpenAI.Responses.ResponseStreamEvent>;
type WithoutSequence<T> = T extends unknown ? Omit<T, "sequence_number"> : never;

interface StepState {
  step: InteractionStep;
  item: Item;
  outputIndex: number;
  text: string;
  arguments: string;
  hasArgumentDeltas: boolean;
  hasFinalArguments: boolean;
  textStarted: boolean;
  stopped: boolean;
}

/** Direct Responses ↔ Interactions V1 conversion; platform extensions remain in adapters. */
export class ResponsesToInteractionsConverter {
  constructor(private readonly options: { allowUnmappedContent?: boolean } = {}) {}
  private params?: OpenAI.Responses.ResponseCreateParams;
  private request?: InteractionCreateParams;
  private id = "";
  private model = "";
  private created = Math.floor(Date.now() / 1000);
  private sequence = 0;
  private started = false;
  private finished = false;
  private terminal?: Interaction;
  private usage?: InteractionUsage;
  private steps = new Map<number, StepState>();
  private seen = new Set<string>();
  private ranges: { start: number; end: number }[] = [];

  convertRequest(params: OpenAI.Responses.ResponseCreateParams): InteractionCreateParams {
    this.params = structuredClone(params);
    this.model = params.model ?? "";
    this.ranges = [];
    const request = convertResponsesInteractionRequest(params, (index, start, end) => {
      this.ranges[index] = { start, end };
    });
    this.setEffectiveRequest(request);
    return request;
  }

  /** Gateways call this after explicit native overrides, before reading any response. */
  setEffectiveRequest(request: InteractionCreateParams): void {
    this.request = structuredClone(request);
  }

  getInputStepRanges(): ReadonlyArray<{ start: number; end: number }> {
    return this.ranges;
  }

  /** Native step ownership is shared with adapters, not inferred from array positions there. */
  getOutputStepEntries(): Array<{ itemId: string; outputIndex: number; step: InteractionStep }> {
    return [...this.steps.values()].map(state => ({
      itemId: state.item.id!,
      outputIndex: state.outputIndex,
      step: structuredClone(state.step),
    }));
  }

  getUsage(): InteractionUsage | undefined {
    return this.usage;
  }

  convertResponse(response: Interaction): InteractionResponse {
    this.checkStatus(response);
    this.identity(response);
    this.usage = response.usage;
    for (const [index, step] of (response.steps ?? []).entries()) this.startStep(index, step);
    this.terminal = response;
    const events = this.finishStream();
    const last = events.at(-1)!;
    if (!("response" in last)) throw new InteractionsResponseError("Missing final response");
    return last.response;
  }

  async *convertStream(
    stream: AsyncIterable<InteractionEvent>
  ): AsyncIterable<InteractionResponseStreamEvent> {
    for await (const event of stream) yield* this.convertStreamEvent(event);
    yield* this.finishStream();
  }

  convertStreamEvent(event: InteractionEvent): InteractionResponseStreamEvent[] {
    if (event.event_id && this.seen.has(event.event_id)) return [];
    if (event.event_id) this.seen.add(event.event_id);
    if (this.finished) throw new InteractionsResponseError("Received event after stream end");
    const usage = event.interaction?.usage ?? event.metadata?.total_usage ?? event.usage;
    if (usage) this.usage = { ...this.usage, ...structuredClone(usage) };
    if (event.event_type === "error")
      throw new InteractionsResponseError(
        event.error?.message ?? "Interactions stream failed",
        event.error?.code
      );
    if (["failed", "cancelled"].includes(event.interaction?.status ?? event.status ?? "")) {
      throw new InteractionsResponseError(
        event.interaction?.errors?.[0]?.message ??
          `Interaction ${event.interaction?.status ?? event.status}`,
        event.interaction?.errors?.[0]?.code
      );
    }
    // A terminal event may precede late cumulative usage metadata. Delay Responses' terminal
    // event until EOF; never report success before the entire upstream stream is validated.
    if (this.terminal) {
      if (event.event_type === "interaction.completed")
        throw new InteractionsResponseError("Duplicate interaction.completed without event_id");
      if (event.event_type.startsWith("step.") || event.interaction)
        throw new InteractionsResponseError("Output event after interaction.completed");
      return [];
    }
    this.identity(event.interaction, event.interaction_id);
    const events: InteractionResponseStreamEvent[] = [];
    if (!this.started) {
      this.started = true;
      events.push(this.event({ type: "response.created", response: this.response("in_progress") }));
      events.push(
        this.event({ type: "response.in_progress", response: this.response("in_progress") })
      );
    }
    if (event.event_type === "step.start") {
      if (!Number.isSafeInteger(event.index) || event.index! < 0 || !event.step)
        throw new InteractionsResponseError("Invalid step.start");
      if (this.steps.has(event.index!))
        throw new InteractionsResponseError("Duplicate step.start without event_id");
      events.push(...this.startStep(event.index!, event.step));
    } else if (event.event_type === "step.delta") {
      const state = event.index == null ? undefined : this.steps.get(event.index);
      if (!state || state.stopped || !event.delta)
        throw new InteractionsResponseError("step.delta has no active step");
      const delta = event.delta;
      if (delta.type === "text") {
        if (state.step.type !== "model_output")
          throw new InteractionsResponseError("Text delta outside model output");
        const text = delta.text ?? "";
        const content = (state.step.content ??= []);
        const last = content.at(-1);
        if (last?.type === "text") last.text = (last.text ?? "") + text;
        else content.push({ type: "text", text });
        events.push(...this.appendText(state, text));
      } else if (delta.type === "thought_summary") {
        if (state.step.type !== "thought" || delta.content?.type !== "text")
          throw new InteractionsResponseError("Unsupported thought summary delta");
        (state.step.summary ??= []).push(structuredClone(delta.content));
        events.push(...this.appendText(state, delta.content.text ?? ""));
      } else if (delta.type === "thought_signature") {
        if (state.step.type !== "thought" || typeof delta.signature !== "string")
          throw new InteractionsResponseError("Invalid thought signature delta");
        this.unmapped("thought signature");
        state.step.signature = delta.signature;
      } else if (delta.type === "arguments_delta") {
        if (state.item.type !== "function_call")
          throw new InteractionsResponseError("Arguments delta outside function call");
        state.hasArgumentDeltas = true;
        state.arguments += delta.arguments ?? "";
        events.push(
          this.event({
            type: "response.function_call_arguments.delta",
            item_id: state.item.id!,
            output_index: state.outputIndex,
            delta: delta.arguments ?? "",
          })
        );
      } else if (delta.type === "text_annotation_delta") {
        if (state.item.type !== "message")
          throw new InteractionsResponseError("Text annotation outside model output");
        const last = (state.step.content ?? []).filter(part => part.type === "text").at(-1);
        if (!last) throw new InteractionsResponseError("Annotation without text");
        last.annotations = [
          ...(last.annotations ?? []),
          ...structuredClone(delta.annotations ?? []),
        ];
      } else if (["image", "audio", "video", "document"].includes(delta.type)) {
        if (state.step.type !== "model_output")
          throw new InteractionsResponseError("Media delta outside model output");
        this.unmapped(delta.type);
        (state.step.content ??= []).push(structuredClone(delta));
      } else throw new InteractionsResponseError(`Unsupported Interactions delta: ${delta.type}`);
    } else if (event.event_type === "step.stop") {
      const state = event.index == null ? undefined : this.steps.get(event.index);
      if (!state || state.stopped)
        throw new InteractionsResponseError("step.stop has no active step");
      state.stopped = true;
      // Done events wait for the final snapshot, which may complete signatures/annotations.
    } else if (event.event_type === "interaction.completed") {
      if (!event.interaction) throw new InteractionsResponseError("Missing completed interaction");
      this.checkStatus(event.interaction);
      for (const [index, step] of (event.interaction.steps ?? []).entries()) {
        const state = this.steps.get(index);
        if (!state) events.push(...this.startStep(index, step));
        else events.push(...this.reconcile(state, step));
      }
      this.terminal = structuredClone(event.interaction);
    } else if (
      !["interaction.created", "interaction.status_update", "metadata"].includes(
        event.event_type
      ) &&
      !usage
    ) {
      throw new InteractionsResponseError(`Unsupported Interactions event: ${event.event_type}`);
    }
    return events;
  }

  finishStream(): InteractionResponseStreamEvent[] {
    if (this.finished) return [];
    if (!this.terminal)
      throw new InteractionsResponseError("Interactions stream ended before interaction.completed");
    const events: InteractionResponseStreamEvent[] = [];
    const status = this.terminal.status === "incomplete" ? "incomplete" : "completed";
    if (
      this.terminal.status === "requires_action" &&
      ![...this.steps.values()].some(state => state.item.type === "function_call")
    )
      throw new InteractionsResponseError("requires_action omitted function calls");
    for (const state of this.steps.values()) {
      const { item, outputIndex } = state;
      if (item.type === "function_call") {
        const args = state.hasArgumentDeltas
          ? state.arguments
          : JSON.stringify(state.step.arguments ?? {});
        if (status !== "incomplete") {
          const parsed = this.arguments(args);
          if (
            state.hasArgumentDeltas &&
            state.hasFinalArguments &&
            !sameJson(parsed, state.step.arguments)
          )
            throw new InteractionsResponseError("Final arguments disagree with streamed arguments");
          state.step.arguments = parsed;
        }
        if (!state.hasArgumentDeltas)
          events.push(
            this.event({
              type: "response.function_call_arguments.delta",
              item_id: item.id!,
              output_index: outputIndex,
              delta: args,
            })
          );
        item.arguments = args;
        events.push(
          this.event({
            type: "response.function_call_arguments.done",
            item_id: item.id!,
            output_index: outputIndex,
            arguments: args,
            name: item.name,
          })
        );
      } else if (item.type === "message" && state.textStarted) {
        const part: OpenAI.Responses.ResponseOutputText = {
          type: "output_text",
          text: state.text,
          annotations: this.annotations(state.step),
          logprobs: [],
        };
        item.content = [part];
        for (const [annotation_index, annotation] of part.annotations.entries())
          events.push(
            this.event({
              type: "response.output_text.annotation.added",
              item_id: item.id,
              output_index: outputIndex,
              content_index: 0,
              annotation_index,
              annotation,
            })
          );
        events.push(
          this.event({
            type: "response.output_text.done",
            item_id: item.id,
            output_index: outputIndex,
            content_index: 0,
            text: state.text,
            logprobs: [],
          })
        );
        events.push(
          this.event({
            type: "response.content_part.done",
            item_id: item.id,
            output_index: outputIndex,
            content_index: 0,
            part,
          })
        );
      } else if (item.type === "reasoning" && state.textStarted) {
        const part = { type: "summary_text" as const, text: state.text };
        item.summary = [part];
        events.push(
          this.event({
            type: "response.reasoning_summary_text.done",
            item_id: item.id,
            output_index: outputIndex,
            summary_index: 0,
            text: state.text,
          })
        );
        events.push(
          this.event({
            type: "response.reasoning_summary_part.done",
            item_id: item.id,
            output_index: outputIndex,
            summary_index: 0,
            part,
          })
        );
      }
      if (item.type === "message" || item.type === "reasoning" || item.type === "function_call")
        item.status = status;
      events.push(
        this.event({
          type: "response.output_item.done",
          output_index: outputIndex,
          item: structuredClone(item),
        })
      );
    }
    this.finished = true;
    events.push(
      this.event({
        type: status === "incomplete" ? "response.incomplete" : "response.completed",
        response: this.response(status),
      })
    );
    return events;
  }

  private startStep(index: number, raw: InteractionStep): InteractionResponseStreamEvent[] {
    const step = structuredClone(raw);
    let item: Item;
    if (step.type === "function_call") {
      if (!step.id || !step.name)
        throw new InteractionsResponseError("Function call omitted id or name");
      if (
        [...this.steps.values()].some(
          s => s.item.type === "function_call" && s.item.call_id === step.id
        )
      )
        throw new InteractionsResponseError("Duplicate function call ID");
      item = {
        type: "function_call",
        id: this.newId("fc"),
        call_id: step.id,
        name: step.name,
        arguments: "",
        status: "in_progress",
      };
    } else if (step.type === "thought") {
      if (step.signature) this.unmapped("thought signature");
      if (step.summary?.some(part => part.type !== "text"))
        this.unmapped("thought summary content");
      item = { type: "reasoning", id: this.newId("rs"), summary: [], status: "in_progress" };
    } else {
      if (step.type !== "model_output") this.unmapped(step.type);
      for (const part of step.content ?? []) if (part.type !== "text") this.unmapped(part.type);
      item = {
        type: "message",
        id: this.newId("msg"),
        role: "assistant",
        status: "in_progress",
        content: [],
      };
    }
    const state: StepState = {
      step,
      item,
      outputIndex: this.steps.size,
      text: "",
      arguments: "",
      hasArgumentDeltas: false,
      hasFinalArguments: false,
      textStarted: false,
      stopped: false,
    };
    this.steps.set(index, state);
    const events = [
      this.event({
        type: "response.output_item.added",
        output_index: state.outputIndex,
        item: structuredClone(item),
      }),
    ];
    const text = this.text(step);
    if (text || (step.content ?? step.summary)?.some(part => part.type === "text"))
      events.push(...this.appendText(state, text));
    return events;
  }

  private appendText(state: StepState, text: string): InteractionResponseStreamEvent[] {
    const { item, outputIndex } = state;
    const events: InteractionResponseStreamEvent[] = [];
    if (!state.textStarted) {
      state.textStarted = true;
      if (item.type === "message")
        events.push(
          this.event({
            type: "response.content_part.added",
            item_id: item.id,
            output_index: outputIndex,
            content_index: 0,
            part: { type: "output_text", text: "", annotations: [], logprobs: [] },
          })
        );
      else if (item.type === "reasoning")
        events.push(
          this.event({
            type: "response.reasoning_summary_part.added",
            item_id: item.id,
            output_index: outputIndex,
            summary_index: 0,
            part: { type: "summary_text", text: "" },
          })
        );
    }
    state.text += text;
    if (item.type === "message")
      events.push(
        this.event({
          type: "response.output_text.delta",
          item_id: item.id,
          output_index: outputIndex,
          content_index: 0,
          delta: text,
          logprobs: [],
        })
      );
    else if (item.type === "reasoning")
      events.push(
        this.event({
          type: "response.reasoning_summary_text.delta",
          item_id: item.id,
          output_index: outputIndex,
          summary_index: 0,
          delta: text,
        })
      );
    return events;
  }

  private reconcile(state: StepState, step: InteractionStep): InteractionResponseStreamEvent[] {
    if (step.type !== state.step.type)
      throw new InteractionsResponseError("Final step type disagrees with stream");
    if (
      step.type === "function_call" &&
      (step.id !== state.step.id || step.name !== state.step.name)
    )
      throw new InteractionsResponseError("Final function identity disagrees with stream");
    const merged = { ...state.step, ...structuredClone(step) };
    const text = this.text(merged);
    if (!text.startsWith(state.text))
      throw new InteractionsResponseError("Final output disagrees with streamed text");
    if (state.step.signature && step.signature != null && step.signature !== state.step.signature)
      throw new InteractionsResponseError("Final signature disagrees with stream");
    if (step.content) {
      const receivedMedia = (state.step.content ?? []).filter(part => part.type !== "text");
      const finalMedia = step.content.filter(part => part.type !== "text");
      if (receivedMedia.some((part, index) => !sameJson(part, finalMedia[index])))
        throw new InteractionsResponseError("Final media disagrees with stream");
    }
    for (const part of step.content ?? []) if (part.type !== "text") this.unmapped(part.type);
    if (step.signature) this.unmapped("thought signature");
    const events =
      text.length > state.text.length ? this.appendText(state, text.slice(state.text.length)) : [];
    state.step = merged;
    state.hasFinalArguments = step.arguments != null;
    return events;
  }

  private response(status: "in_progress" | "completed" | "incomplete"): InteractionResponse {
    const p = this.params;
    return {
      id: this.id,
      object: "response",
      created_at: this.created,
      model: this.model,
      status,
      output:
        status === "in_progress" ? [] : [...this.steps.values()].map(s => structuredClone(s.item)),
      output_text:
        status === "in_progress"
          ? ""
          : [...this.steps.values()]
              .filter(s => s.item.type === "message")
              .map(s => s.text)
              .join(""),
      error: null,
      incomplete_details: null,
      instructions: this.request?.system_instruction ?? null,
      metadata: this.request?.labels ?? {},
      temperature: null,
      top_p: null,
      max_output_tokens: this.request?.generation_config?.max_output_tokens ?? null,
      previous_response_id: null,
      parallel_tool_calls: p?.parallel_tool_calls ?? true,
      ...this.responseConfig(),
      truncation: p?.truncation ?? "disabled",
      store: this.request?.store ?? true,
      background: false,
      usage: status === "in_progress" ? null : convertInteractionResponseUsage(this.usage),
    } as InteractionResponse;
  }

  private responseConfig() {
    const request = this.request;
    const config = request?.generation_config;
    const choice = config?.tool_choice;
    const format = request?.response_format;
    // Native-only options have no standard echo. Gateways retain the native request.
    const text =
      format == null || (!Array.isArray(format) && format.mime_type === "text/plain")
        ? { format: { type: "text" } }
        : !Array.isArray(format) &&
            format.type === "text" &&
            format.mime_type === "application/json"
          ? {
              format: format.schema
                ? {
                    type: "json_schema",
                    name:
                      this.params?.text?.format?.type === "json_schema"
                        ? this.params.text.format.name
                        : "response",
                    schema: format.schema,
                  }
                : { type: "json_object" },
            }
          : undefined;
    return {
      tool_choice:
        choice == null
          ? "auto"
          : choice === "any"
            ? "required"
            : typeof choice === "object"
              ? {
                  type: "allowed_tools",
                  mode: choice.allowed_tools.mode === "any" ? "required" : "auto",
                  tools: choice.allowed_tools.tools.map(name => ({ type: "function", name })),
                }
              : choice === "validated"
                ? undefined
                : choice,
      tools: request?.tools?.every(tool => ["function", "google_search"].includes(tool.type))
        ? request.tools.map(tool =>
            tool.type === "google_search" ? { type: "web_search" } : { ...tool }
          )
        : request?.tools?.length
          ? undefined
          : [],
      text,
      reasoning:
        config?.thinking_level || config?.thinking_summaries === "auto"
          ? {
              ...(config.thinking_level && { effort: config.thinking_level }),
              ...(config.thinking_summaries === "auto" && { summary: "auto" }),
            }
          : null,
    };
  }

  private event(
    event: WithoutSequence<InteractionResponseStreamEvent>
  ): InteractionResponseStreamEvent {
    return { ...event, sequence_number: this.sequence++ } as InteractionResponseStreamEvent;
  }
  private identity(response?: Interaction, id?: string): void {
    if (!this.id) this.id = response?.id || id || this.newId("resp");
    if (response?.model) this.model = response.model;
    const time = response?.created ? Date.parse(response.created) : NaN;
    if (!this.started && Number.isFinite(time)) this.created = Math.floor(time / 1000);
  }
  private newId(prefix: string): string {
    return `${prefix}_${Math.random().toString(36).slice(2)}`;
  }
  private checkStatus(response: Interaction): void {
    if (typeof response.id !== "string")
      throw new InteractionsResponseError("Missing interaction ID");
    if (!["completed", "requires_action", "incomplete"].includes(response.status))
      throw new InteractionsResponseError(
        response.errors?.[0]?.message ?? `Unexpected interaction status: ${response.status}`,
        response.errors?.[0]?.code
      );
  }
  private unmapped(type: string): void {
    if (!this.options.allowUnmappedContent)
      throw new InteractionsResponseError(`Interactions ${type} requires a Responses extension`);
  }
  private text(step: InteractionStep): string {
    return (step.type === "thought" ? (step.summary ?? []) : (step.content ?? []))
      .filter(part => part.type === "text")
      .map(part => part.text ?? "")
      .join("");
  }
  private arguments(value: string): Record<string, unknown> {
    try {
      const parsed: unknown = JSON.parse(value);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error();
      return parsed as Record<string, unknown>;
    } catch {
      throw new InteractionsResponseError("Incomplete or invalid function arguments in stream");
    }
  }
  private annotations(step: InteractionStep): OpenAI.Responses.ResponseOutputText["annotations"] {
    let offset = 0;
    return (step.content ?? []).flatMap(part => {
      if (part.type !== "text") return [];
      const text = part.text ?? "";
      const annotations = (part.annotations ?? []).flatMap((annotation: InteractionAnnotation) => {
        if (
          annotation.type !== "url_citation" ||
          !annotation.url ||
          annotation.start_index == null ||
          annotation.end_index == null
        )
          return [];
        const index = (bytes: number) =>
          Array.from(new TextDecoder().decode(new TextEncoder().encode(text).slice(0, bytes)))
            .length;
        return [
          {
            type: "url_citation" as const,
            url: annotation.url,
            title: annotation.title ?? annotation.url,
            start_index: offset + index(annotation.start_index),
            end_index: offset + index(annotation.end_index),
          },
        ];
      });
      offset += Array.from(text).length;
      return annotations;
    });
  }
}

function sameJson(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a == null || b == null || typeof a !== "object" || typeof b !== "object") return false;
  const left = a as Record<string, unknown>,
    right = b as Record<string, unknown>;
  return (
    Array.isArray(a) === Array.isArray(b) &&
    Object.keys(left).length === Object.keys(right).length &&
    Object.keys(left).every(key => Object.hasOwn(right, key) && sameJson(left[key], right[key]))
  );
}
