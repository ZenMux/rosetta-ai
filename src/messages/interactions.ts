import type Anthropic from "@anthropic-ai/sdk";
import {
  InteractionsResponseError,
  type Interaction,
  type InteractionCreateParams,
  type InteractionEvent,
  type InteractionStep,
  type InteractionStartStep,
  type InteractionSnapshot,
  type InteractionUsage,
} from "../interactions/types";
import {
  convertInteractionMessageUsage,
  type InteractionMessageUsage,
} from "../interactions/usage";
import {
  eventInteraction,
  eventInteractionId,
  eventStatus,
  eventUsage,
} from "../interactions/events";
import { interactionStepContent, sameJson, validateFinalStep } from "../interactions/steps";
import { convertMessagesInteractionRequest } from "./interactions-request";

export type InteractionMessage = Omit<Anthropic.Message, "usage"> & {
  usage?: Partial<InteractionMessageUsage>;
};
export type InteractionMessageStreamEvent =
  | Exclude<
      Anthropic.RawMessageStreamEvent,
      Anthropic.RawMessageStartEvent | Anthropic.RawMessageDeltaEvent
    >
  | { type: "message_start"; message: InteractionMessage }
  | {
      type: "message_delta";
      delta: Anthropic.RawMessageDeltaEvent["delta"];
      usage?: Partial<InteractionMessageUsage>;
    };
interface StateFields {
  index: number;
  text: string;
  stopped: boolean;
}

type State = StateFields &
  (
    | {
        kind: "function_call";
        step: Extract<InteractionStartStep, { type: "function_call" }>;
        block: Anthropic.ToolUseBlock;
        arguments: string;
        hasDeltas: boolean;
        finalArguments: boolean;
      }
    | {
        kind: "thought";
        step: Extract<InteractionStartStep, { type: "thought" }>;
        block: Anthropic.ThinkingBlock;
      }
    | {
        kind: "output";
        step: Exclude<InteractionStartStep, { type: "function_call" | "thought" }>;
        block: Anthropic.TextBlock;
      }
  );

/** Direct Messages ↔ Interactions V1; only target-protocol content is returned. */
export class MessagesToInteractionsConverter {
  private model = "";
  private id = "";
  private started = false;
  private finished = false;
  private terminal?: InteractionSnapshot;
  private usage?: InteractionUsage;
  private states = new Map<number, State>();
  private seen = new Set<string>();

  convertRequest(params: Anthropic.MessageCreateParams): InteractionCreateParams {
    this.model = params.model;
    return convertMessagesInteractionRequest(params);
  }
  getUsage(): InteractionUsage | undefined {
    return this.usage;
  }
  getStatus(): Interaction["status"] | undefined {
    return this.terminal?.status;
  }

  convertResponse(raw: Interaction): InteractionMessage {
    this.usage = structuredClone(raw.usage);
    this.checkStatus(raw);
    this.identity(raw);
    for (const [index, step] of (raw.steps ?? []).entries()) this.start(index, step);
    this.terminal = structuredClone(raw);
    this.finishStream();
    return this.message(true);
  }

  async *convertStream(
    stream: AsyncIterable<InteractionEvent>
  ): AsyncIterable<InteractionMessageStreamEvent> {
    for await (const event of stream) yield* this.convertStreamEvent(event);
    yield* this.finishStream();
  }

  convertStreamEvent(event: InteractionEvent): InteractionMessageStreamEvent[] {
    if (event.event_id && this.seen.has(event.event_id)) return [];
    if (event.event_id) this.seen.add(event.event_id);
    if (this.finished) fail("Received event after stream end");
    const usage = eventUsage(event);
    const interaction = eventInteraction(event);
    const status = eventStatus(event);
    if (usage) this.usage = { ...this.usage, ...structuredClone(usage) };
    if (event.event_type === "error")
      throw new InteractionsResponseError(
        event.error?.message ?? "Interactions stream failed",
        event.error?.code
      );
    if (["failed", "cancelled"].includes(status ?? ""))
      throw new InteractionsResponseError(
        interaction?.errors?.[0]?.message ?? `Interaction ${status}`,
        interaction?.errors?.[0]?.code
      );
    if (this.terminal) {
      if (
        event.event_type === "interaction.completed" ||
        event.event_type.startsWith("step.") ||
        interaction
      )
        fail("Output event after interaction.completed");
      return [];
    }
    this.identity(interaction, eventInteractionId(event));
    const events: InteractionMessageStreamEvent[] = [];
    if (!this.started) {
      this.started = true;
      events.push({
        type: "message_start",
        message: {
          ...this.message(false),
          usage: convertInteractionMessageUsage(this.usage) ?? { output_tokens_details: null },
        },
      });
    }
    if (event.event_type === "step.start") {
      if (!Number.isSafeInteger(event.index) || event.index! < 0 || !event.step)
        fail("Invalid step.start");
      if (this.states.has(event.index!)) fail("Duplicate step.start without event_id");
      events.push(...this.start(event.index!, event.step));
    } else if (event.event_type === "step.delta") {
      const state = event.index == null ? undefined : this.states.get(event.index);
      if (!state || state.stopped || !event.delta) fail("step.delta has no active step");
      const delta = event.delta;
      if (delta.type === "text") {
        if (state.kind !== "output" || state.step.type !== "model_output")
          fail("Text delta outside model output");
        const content = (state.step.content ??= []),
          last = content.at(-1);
        if (last?.type === "text") last.text = (last.text ?? "") + (delta.text ?? "");
        else content.push({ type: "text", text: delta.text ?? "" });
        events.push(this.append(state, delta.text ?? ""));
      } else if (delta.type === "thought_summary") {
        if (state.kind !== "thought" || delta.content?.type !== "text")
          fail("Invalid thought summary delta");
        (state.step.summary ??= []).push(structuredClone(delta.content));
        events.push(this.append(state, delta.content.text ?? ""));
      } else if (delta.type === "thought_signature") {
        if (state.kind !== "thought" || typeof delta.signature !== "string")
          fail("Invalid thought signature delta");
        state.step.signature = delta.signature;
      } else if (delta.type === "arguments_delta") {
        if (state.kind !== "function_call" || typeof delta.arguments !== "string")
          fail("Invalid function arguments delta");
        state.hasDeltas = true;
        state.arguments += delta.arguments;
        events.push({
          type: "content_block_delta",
          index: state.index,
          delta: { type: "input_json_delta", partial_json: delta.arguments },
        });
      } else if (delta.type === "text_annotation_delta") {
        if (state.kind !== "output" || state.step.type !== "model_output")
          fail("Annotation without model text");
        const last = state.step.content?.filter(part => part.type === "text").at(-1);
        if (!last) fail("Annotation without model text");
        this.unmapped("text annotations");
        last.annotations = [
          ...(last.annotations ?? []),
          ...structuredClone(delta.annotations ?? []),
        ];
      } else if (
        delta.type === "video" ||
        delta.type === "image" ||
        delta.type === "audio" ||
        delta.type === "document"
      ) {
        if (state.kind !== "output" || state.step.type !== "model_output")
          fail("Media outside model output");
        this.unmapped(delta.type);
        (state.step.content ??= []).push(structuredClone(delta));
      } else fail(`Unsupported Interactions delta: ${delta.type}`);
    } else if (event.event_type === "step.stop") {
      const state = event.index == null ? undefined : this.states.get(event.index);
      if (!state || state.stopped) fail("step.stop has no active step");
      state.stopped = true;
    } else if (event.event_type === "interaction.completed") {
      if (!event.interaction) fail("Missing completed interaction");
      this.checkStatus(event.interaction);
      for (const [index, step] of (event.interaction.steps ?? []).entries()) {
        const state = this.states.get(index);
        if (state) events.push(...this.reconcile(state, step));
        else events.push(...this.start(index, step));
      }
      this.terminal = structuredClone(event.interaction);
    } else if (
      !["interaction.created", "interaction.status_update", "metadata"].includes(
        event.event_type
      ) &&
      !usage
    )
      fail(`Unsupported Interactions event: ${event.event_type}`);
    return events;
  }

  finishStream(): InteractionMessageStreamEvent[] {
    if (this.finished) return [];
    if (!this.terminal) fail("Interactions stream ended before interaction.completed");
    if (
      this.terminal.status === "requires_action" &&
      ![...this.states.values()].some(s => s.block.type === "tool_use")
    )
      fail("requires_action omitted function calls");
    const events: InteractionMessageStreamEvent[] = [];
    for (const state of this.states.values()) {
      if (state.kind === "function_call") {
        const argumentsText = state.hasDeltas
          ? state.arguments
          : JSON.stringify(state.step.arguments ?? {});
        const args = parseArguments(argumentsText);
        if (state.hasDeltas && state.finalArguments && !sameJson(args, state.step.arguments))
          fail("Final arguments disagree with streamed arguments");
        state.step.arguments = args;
        state.block.input = args;
        if (!state.hasDeltas)
          events.push({
            type: "content_block_delta",
            index: state.index,
            delta: { type: "input_json_delta", partial_json: argumentsText },
          });
      } else if (state.kind === "thought") {
        state.block.signature = state.step.signature ?? "";
        if (state.step.signature)
          events.push({
            type: "content_block_delta",
            index: state.index,
            delta: { type: "signature_delta", signature: state.step.signature },
          });
      }
      events.push({ type: "content_block_stop", index: state.index });
    }
    this.finished = true;
    events.push({
      type: "message_delta",
      delta: {
        stop_reason: this.stopReason(),
        stop_sequence: null,
        container: null,
        stop_details: null,
      },
      usage: convertInteractionMessageUsage(this.usage) ?? { output_tokens_details: null },
    });
    events.push({ type: "message_stop" });
    return events;
  }

  private start(index: number, raw: InteractionStartStep): InteractionMessageStreamEvent[] {
    const step = structuredClone(raw);
    const fields: StateFields = { index: this.states.size, text: "", stopped: false };
    let state: State;
    if (step.type === "function_call") {
      if (!step.id || !step.name) fail("Function call omitted id or name");
      if (
        [...this.states.values()].some(s => s.block.type === "tool_use" && s.block.id === step.id)
      )
        fail("Duplicate function call ID");
      state = {
        ...fields,
        kind: "function_call",
        step,
        arguments: "",
        hasDeltas: false,
        finalArguments: step.arguments != null,
        block: {
          type: "tool_use",
          id: step.id,
          name: step.name,
          input: {},
          caller: { type: "direct" },
        },
      };
    } else if (step.type === "thought") {
      if (step.summary?.some(part => part.type !== "text")) this.unmapped("thought summary");
      state = {
        ...fields,
        kind: "thought",
        step,
        block: { type: "thinking", thinking: "", signature: "" },
      };
    } else {
      if (step.type !== "model_output") this.unmapped(step.type);
      state = {
        ...fields,
        kind: "output",
        step,
        block: { type: "text", text: "", citations: null },
      };
    }
    this.validateContent(step);
    this.states.set(index, state);
    const events: InteractionMessageStreamEvent[] = [
      {
        type: "content_block_start",
        index: state.index,
        content_block: structuredClone(state.block),
      },
    ];
    const text = this.text(step);
    if (text) events.push(this.append(state, text));
    return events;
  }
  private append(state: State, text: string): InteractionMessageStreamEvent {
    state.text += text;
    if (state.kind === "thought") {
      state.block.thinking = state.text;
      return {
        type: "content_block_delta",
        index: state.index,
        delta: { type: "thinking_delta", thinking: text },
      };
    }
    if (state.block.type !== "text") fail("Text outside a text/thinking block");
    state.block.text = state.text;
    return { type: "content_block_delta", index: state.index, delta: { type: "text_delta", text } };
  }
  private reconcile(state: State, step: InteractionStep): InteractionMessageStreamEvent[] {
    validateFinalStep(state.step, step);
    if (state.kind === "function_call" && step.type === "function_call") {
      state.step = { ...state.step, ...structuredClone(step) };
      state.finalArguments = step.arguments != null;
      return [];
    }
    if (state.kind === "thought" && step.type === "thought") {
      state.step = { ...state.step, ...structuredClone(step) };
    } else if (
      state.kind === "output" &&
      step.type !== "thought" &&
      step.type !== "function_call"
    ) {
      if (state.step.type === "model_output" && step.type === "model_output")
        state.step = { ...state.step, ...structuredClone(step) };
      else if (state.step.type === "user_input" && step.type === "user_input")
        state.step = { ...state.step, ...structuredClone(step) };
      else Object.assign(state.step, structuredClone(step));
    }
    this.validateContent(state.step);
    const text = this.text(state.step);
    if (!text.startsWith(state.text)) fail("Final output disagrees with streamed text");
    const events =
      text.length > state.text.length ? [this.append(state, text.slice(state.text.length))] : [];
    return events;
  }
  private message(final: boolean): InteractionMessage {
    return {
      id: this.id,
      type: "message",
      role: "assistant",
      model: this.model,
      content: final ? [...this.states.values()].map(s => structuredClone(s.block)) : [],
      stop_reason: final ? this.stopReason() : null,
      stop_sequence: null,
      container: null,
      diagnostics: null,
      stop_details: null,
      ...(this.usage && { usage: convertInteractionMessageUsage(this.usage) }),
    };
  }
  private stopReason(): Anthropic.StopReason | null {
    // The native status does not identify why results are incomplete.
    if (this.terminal?.status === "incomplete") return null;
    return [...this.states.values()].some(s => s.block.type === "tool_use")
      ? "tool_use"
      : "end_turn";
  }
  private identity(raw?: InteractionSnapshot, id?: string): void {
    this.id ||= raw?.id || id || `msg_${Math.random().toString(36).slice(2)}`;
    if (raw?.model) this.model = raw.model;
  }
  private checkStatus(raw: InteractionSnapshot): void {
    // Empty strings use the target ID generated by identity(); the field remains required.
    if (typeof raw.id !== "string") fail("Missing interaction ID");
    if (!["completed", "requires_action", "incomplete"].includes(raw.status))
      throw new InteractionsResponseError(
        raw.errors?.[0]?.message ?? `Unexpected interaction status: ${raw.status}`,
        raw.errors?.[0]?.code
      );
  }
  private text(step: InteractionStartStep): string {
    return interactionStepContent(step)
      .filter(p => p.type === "text")
      .map(p => p.text ?? "")
      .join("");
  }
  private validateContent(step: InteractionStartStep): void {
    for (const part of interactionStepContent(step)) {
      if (part.type !== "text") this.unmapped(part.type);
      if (part.type === "text" && part.annotations?.length) this.unmapped("text annotations");
    }
  }
  private unmapped(kind: string): void {
    fail(`Interactions ${kind} is not supported by Messages conversion`);
  }
}
function parseArguments(value: string): Record<string, unknown> {
  try {
    const result: unknown = JSON.parse(value);
    if (!result || typeof result !== "object" || Array.isArray(result)) throw new Error();
    return result as Record<string, unknown>;
  } catch {
    return fail("Incomplete or invalid function arguments in stream");
  }
}
function fail(message: string): never {
  throw new InteractionsResponseError(message);
}
