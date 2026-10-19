import type OpenAI from "openai";
import {
  InteractionsResponseError,
  type Interaction,
  type InteractionAnnotation,
  type InteractionCreateParams,
  type InteractionEvent,
  type InteractionStartStep,
  type InteractionSnapshot,
  type InteractionUsage,
} from "../interactions/types";
import { convertInteractionUsage, type InteractionCompletionUsage } from "../interactions/usage";
import {
  eventInteraction,
  eventInteractionId,
  eventStatus,
  eventUsage,
} from "../interactions/events";
import { validateFinalStep } from "../interactions/steps";
import { convertChatCompletionRequest } from "./interactions-request";

export type InteractionChatCompletion = Omit<OpenAI.ChatCompletion, "usage"> & {
  usage?: InteractionCompletionUsage;
};
export type InteractionChatCompletionChunk = Omit<OpenAI.ChatCompletionChunk, "usage"> & {
  usage?: InteractionCompletionUsage;
};

type StepState = { stopped: boolean } & (
  | {
      kind: "model_output";
      step: Extract<InteractionStartStep, { type: "model_output" }>;
      text: string;
      textOffset: number;
    }
  | {
      kind: "function_call";
      step: Extract<InteractionStartStep, { type: "function_call" }>;
      toolIndex: number;
      arguments: string;
      sentArguments: boolean;
    }
  | {
      kind: "other";
      step: Exclude<InteractionStartStep, { type: "model_output" | "function_call" }>;
    }
);

export class ChatCompletionToInteractionsConverter {
  private id = "";
  private model = "";
  private created = Math.floor(Date.now() / 1000);
  private started = false;
  private completed = false;
  private steps = new Map<number, StepState>();
  private eventIds = new Set<string>();
  private toolCount = 0;
  private textLength = 0;
  private usage?: InteractionUsage;
  convertRequest(params: OpenAI.ChatCompletionCreateParams): InteractionCreateParams {
    this.model = params.model;
    return convertChatCompletionRequest(params);
  }

  convertResponse(response: Interaction): InteractionChatCompletion {
    this.usage = response.usage && structuredClone(response.usage);
    this.checkStatus(response);
    const text: string[] = [];
    const tools: OpenAI.ChatCompletionMessageFunctionToolCall[] = [];
    const annotations: OpenAI.ChatCompletionMessage.Annotation[] = [];
    for (const step of response.steps ?? []) {
      if (step.type === "model_output") {
        for (const part of step.content ?? []) {
          if (part.type !== "text") {
            this.unsupportedContent(part.type);
            continue;
          }
          const offset = Array.from(text.join("")).length;
          const value = part.text ?? "";
          annotations.push(...this.convertAnnotations(part.annotations ?? [], value, offset));
          text.push(value);
        }
      } else if (step.type === "function_call") {
        this.checkFunction(step);
        tools.push({
          id: step.id!,
          type: "function",
          function: { name: step.name!, arguments: JSON.stringify(step.arguments ?? {}) },
        });
      } else this.validateHiddenStep(step);
    }
    return {
      id: response.id,
      object: "chat.completion",
      created: this.timestamp(response.created),
      model: response.model ?? this.model,
      choices: [
        {
          index: 0,
          finish_reason: this.finishReason(response, tools.length > 0),
          logprobs: null,
          message: {
            role: "assistant",
            content: text.length ? text.join("") : null,
            refusal: null,
            ...(tools.length > 0 && { tool_calls: tools }),
            ...(annotations.length > 0 && { annotations }),
          },
        },
      ],
      ...(response.usage && { usage: convertInteractionUsage(response.usage) }),
    };
  }

  async *convertStream(
    stream: AsyncIterable<InteractionEvent>
  ): AsyncIterable<InteractionChatCompletionChunk> {
    for await (const event of stream) yield* this.convertStreamEvent(event);
    this.finishStream();
  }

  convertStreamEvent(event: InteractionEvent): InteractionChatCompletionChunk[] {
    if (event.event_id && this.eventIds.has(event.event_id)) return [];
    if (event.event_id) this.eventIds.add(event.event_id);
    if (this.completed)
      throw new InteractionsResponseError("Received an event after interaction.completed");
    // Usage metadata and step.stop.usage are cumulative snapshots, not deltas.
    const usage = eventUsage(event);
    const interaction = eventInteraction(event);
    const status = eventStatus(event);
    if (usage) this.usage = { ...this.usage, ...structuredClone(usage) };
    if (event.event_type === "error")
      throw new InteractionsResponseError(
        event.error?.message ?? "Interactions stream failed",
        event.error?.code
      );
    if (interaction) {
      this.model = interaction.model ?? this.model;
      this.created = this.timestamp(interaction.created);
      if (["failed", "cancelled"].includes(interaction.status)) this.checkStatus(interaction, true);
    }
    if (status === "failed" || status === "cancelled")
      throw new InteractionsResponseError(`Interaction ${status}`);
    if (!this.id) {
      // Vertex may leave the interaction ID empty throughout a stream. Choose the CC ID
      // once; a later native ID must not change chunks already sent to clients.
      this.id = interaction?.id || eventInteractionId(event) || `chatcmpl-${this.generateId()}`;
    }

    const chunks: InteractionChatCompletionChunk[] = [];
    if (!this.started) {
      this.started = true;
      chunks.push(this.chunk({ role: "assistant", content: "" }));
    }
    if (event.event_type === "step.start") {
      if (event.index == null || !event.step)
        throw new InteractionsResponseError("Invalid step.start");
      if (this.steps.has(event.index))
        throw new InteractionsResponseError("Duplicate step.start without event_id");
      chunks.push(...this.startStep(event.index, event.step));
    } else if (event.event_type === "step.delta") {
      const state = event.index == null ? undefined : this.steps.get(event.index);
      if (!state || state.stopped || !event.delta)
        throw new InteractionsResponseError("step.delta has no active step");
      const delta = event.delta;
      if (delta.type === "text") {
        if (state.kind !== "model_output")
          throw new InteractionsResponseError("Text delta outside model output");
        const text = delta.text ?? "";
        state.text += text;
        this.textLength += Array.from(text).length;
        chunks.push(this.chunk({ content: text }));
      } else if (delta.type === "arguments_delta") {
        if (state.kind !== "function_call")
          throw new InteractionsResponseError("Arguments delta outside function call");
        const value = delta.arguments ?? "";
        state.arguments += value;
        state.sentArguments = true;
        chunks.push(
          this.chunk({ tool_calls: [{ index: state.toolIndex, function: { arguments: value } }] })
        );
      } else if (delta.type === "text_annotation_delta") {
        if (state.kind !== "model_output")
          throw new InteractionsResponseError("Text annotation outside model output");
        const annotations = this.convertAnnotations(
          delta.annotations ?? [],
          state.text,
          state.textOffset
        );
        if (annotations.length) chunks.push(this.chunk({ content: "", ...{ annotations } }));
      } else if (delta.type === "thought_summary" && state.step.type === "thought") {
        if (delta.content?.type !== "text") this.unsupportedContent("thought summary");
      } else if (delta.type === "thought_signature" && state.step.type === "thought") {
        state.step.signature = delta.signature;
      } else this.unsupportedContent(delta.type);
    } else if (event.event_type === "step.stop") {
      const state = event.index == null ? undefined : this.steps.get(event.index);
      if (!state || state.stopped)
        throw new InteractionsResponseError("step.stop has no active step");
      chunks.push(...this.stopStep(state));
    } else if (event.event_type === "interaction.completed") {
      if (!event.interaction) throw new InteractionsResponseError("Missing completed interaction");
      this.checkStatus(event.interaction, true);
      // Some servers supply output only in the final snapshot. Never replay steps
      // already delivered as deltas when the final event also includes a snapshot.
      for (const [index, step] of (event.interaction.steps ?? []).entries()) {
        const state = this.steps.get(index);
        if (state) validateFinalStep(state.step, step);
        if (!state) chunks.push(...this.startStep(index, step));
        else if (step.type === "model_output" && state.kind === "model_output") {
          const text = (step.content ?? [])
            .filter(part => part.type === "text")
            .map(part => part.text ?? "")
            .join("");
          if (!text.startsWith(state.text))
            throw new InteractionsResponseError("Final output disagrees with streamed text");
          const remaining = text.slice(state.text.length);
          if (remaining) {
            state.text = text;
            this.textLength += Array.from(remaining).length;
            chunks.push(this.chunk({ content: remaining }));
          }
          for (const part of step.content ?? [])
            if (part.type !== "text") this.unsupportedContent(part.type);
        } else if (
          step.type === "function_call" &&
          state.kind === "function_call" &&
          !state.stopped
        ) {
          state.step = step;
        }
      }
      for (const state of this.steps.values())
        if (!state.stopped) chunks.push(...this.stopStep(state));
      if (event.interaction.status !== "incomplete") this.validateToolArguments();
      const final = this.chunk({});
      final.choices[0].finish_reason = this.finishReason(event.interaction, this.toolCount > 0);
      chunks.push(final);
      if (this.usage)
        chunks.push({ ...this.chunk({}), choices: [], usage: convertInteractionUsage(this.usage) });
      this.completed = true;
    } else if (
      !["interaction.created", "interaction.status_update", "metadata"].includes(event.event_type)
    ) {
      this.unsupportedContent(event.event_type);
    }
    return chunks;
  }

  // Adapters call this at EOF so a truncated stream is never reported as success.
  finishStream(): void {
    if (!this.completed)
      throw new InteractionsResponseError("Interactions stream ended before interaction.completed");
  }

  getUsage(): InteractionUsage | undefined {
    return this.usage;
  }

  private startStep(index: number, step: InteractionStartStep): InteractionChatCompletionChunk[] {
    if (step.type === "function_call") {
      this.checkFunction(step);
      const state: Extract<StepState, { kind: "function_call" }> = {
        kind: "function_call",
        step,
        toolIndex: this.toolCount++,
        arguments: "",
        sentArguments: false,
        stopped: false,
      };
      this.steps.set(index, state);
      return [
        this.chunk({
          tool_calls: [
            {
              index: state.toolIndex,
              id: step.id!,
              type: "function",
              function: { name: step.name!, arguments: "" },
            },
          ],
        }),
      ];
    }
    if (step.type === "model_output") {
      const state: Extract<StepState, { kind: "model_output" }> = {
        kind: "model_output",
        step,
        text: "",
        textOffset: this.textLength,
        stopped: false,
      };
      this.steps.set(index, state);
      const chunks: InteractionChatCompletionChunk[] = [];
      for (const part of step.content ?? []) {
        if (part.type !== "text") {
          this.unsupportedContent(part.type);
          continue;
        }
        const text = part.text ?? "";
        const annotations = this.convertAnnotations(part.annotations ?? [], text, this.textLength);
        state.text += text;
        this.textLength += Array.from(text).length;
        chunks.push(this.chunk({ content: text, ...(annotations.length > 0 && { annotations }) }));
      }
      return chunks;
    }
    this.validateHiddenStep(step);
    this.steps.set(index, { kind: "other", step, stopped: false });
    return [];
  }

  private stopStep(state: StepState): InteractionChatCompletionChunk[] {
    state.stopped = true;
    if (state.kind !== "function_call") return [];
    if (state.sentArguments) return [];
    return [
      this.chunk({
        tool_calls: [
          {
            index: state.toolIndex,
            function: { arguments: JSON.stringify(state.step.arguments ?? {}) },
          },
        ],
      }),
    ];
  }

  private validateToolArguments(): void {
    for (const state of this.steps.values()) {
      if (state.kind !== "function_call" || !state.sentArguments) continue;
      try {
        const parsed: unknown = JSON.parse(state.arguments);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error();
      } catch {
        throw new InteractionsResponseError("Incomplete or invalid function arguments in stream");
      }
    }
  }

  private chunk(delta: OpenAI.ChatCompletionChunk.Choice.Delta): InteractionChatCompletionChunk {
    return {
      id: this.id,
      object: "chat.completion.chunk",
      model: this.model,
      created: this.created,
      choices: [{ index: 0, delta, finish_reason: null, logprobs: null }],
    };
  }

  private timestamp(value?: string): number {
    const parsed = value ? Date.parse(value) : NaN;
    return Number.isFinite(parsed) ? Math.floor(parsed / 1000) : this.created;
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  private checkStatus(response: InteractionSnapshot, allowEmptyId = false): void {
    if (typeof response.id !== "string" || (!allowEmptyId && !response.id))
      throw new InteractionsResponseError("Missing interaction ID");
    if (!["completed", "requires_action", "incomplete"].includes(response.status)) {
      throw new InteractionsResponseError(
        response.errors?.[0]?.message ?? `Unexpected interaction status: ${response.status}`,
        response.errors?.[0]?.code
      );
    }
  }

  private finishReason(
    response: InteractionSnapshot,
    hasTools: boolean
  ): OpenAI.ChatCompletion.Choice["finish_reason"] {
    if (response.status === "incomplete") return "length";
    return hasTools || response.status === "requires_action" ? "tool_calls" : "stop";
  }

  private checkFunction(step: Extract<InteractionStartStep, { type: "function_call" }>): void {
    if (!step.id || !step.name)
      throw new InteractionsResponseError("Function call omitted id or name");
  }

  private validateHiddenStep(step: InteractionStartStep): void {
    // CC exposes search citations in assistant text, but has no reasoning content field.
    if (step.type === "thought") {
      if (step.summary?.some(part => part.type !== "text"))
        this.unsupportedContent("thought summary");
    } else if (!["google_search_call", "google_search_result"].includes(step.type)) {
      this.unsupportedContent(step.type);
    }
  }

  private unsupportedContent(type: string): never {
    throw new InteractionsResponseError(
      `Interactions ${type} output is not supported by Chat Completions conversion`
    );
  }

  private convertAnnotations(
    annotations: InteractionAnnotation[],
    text: string,
    offset: number
  ): OpenAI.ChatCompletionMessage.Annotation[] {
    return annotations.flatMap(annotation => {
      if (annotation.type !== "url_citation" || annotation.url == null) return [];
      if (annotation.start_index == null || annotation.end_index == null) return [];
      // Google uses UTF-8 byte offsets; CC uses character offsets. Count Unicode
      // code points rather than treating a Chinese character as three positions.
      const index = (bytes: number) =>
        Array.from(new TextDecoder().decode(new TextEncoder().encode(text).slice(0, bytes))).length;
      return [
        {
          type: "url_citation" as const,
          url_citation: {
            url: annotation.url,
            title: annotation.title ?? annotation.url,
            start_index: offset + index(annotation.start_index),
            end_index: offset + index(annotation.end_index),
          },
        },
      ];
    });
  }
}
