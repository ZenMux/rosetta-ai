import type { Interactions } from "@google/genai";

// Standard protocol shapes have one source of truth: the pinned GenAI SDK.
export type InteractionContent = Interactions.Content;
export type InteractionAnnotation = Interactions.Annotation;
export type InteractionStep = Interactions.Step;
export type InteractionUsage = Interactions.Usage;
export type InteractionModalityTokens = NonNullable<
  InteractionUsage["input_tokens_by_modality"]
>[number];
// Vertex stateless responses can omit the ID required by the SDK's full resource.
// Preserve that absence; converters generate only their target-protocol client ID.
type OptionalInteractionId<T extends { id: string }> = Omit<T, "id"> & Partial<Pick<T, "id">>;
export type Interaction = OptionalInteractionId<Interactions.Interaction>;
export type InteractionStatus = Interaction["status"];
export type InteractionGenerationConfig = Interactions.GenerationConfig;
export type InteractionResponseFormat = Exclude<
  NonNullable<Interactions.Interaction["response_format"]>,
  unknown[]
>;

// The converter builds a model request body, not SDK transport/agent options.
export type InteractionCreateParams = Omit<
  Interactions.CreateModelInteractionParamsNonStreaming,
  "input"
> & { input: string | InteractionStep[] };

// Existing stream fixtures start function calls before arguments are available.
// Keep that tolerance local to start events; complete steps retain SDK requirements.
export type InteractionStartStep =
  | Exclude<InteractionStep, { type: "function_call" }>
  | (Omit<Interactions.FunctionCallStep, "arguments"> &
      Partial<Pick<Interactions.FunctionCallStep, "arguments">>);
export type InteractionSnapshot = OptionalInteractionId<
  Interactions.InteractionCompletedEvent["interaction"]
> &
  Pick<Interaction, "errors">;

type OptionalLifecycleId<T> = T extends { interaction: infer Resource extends { id: string } }
  ? Omit<T, "interaction"> & { interaction: OptionalInteractionId<Resource> }
  : T;

// The existing converters also accept independent/late cumulative metadata.
// StepDelta documents metadata on any event; usage retains its existing precedence.
type UsageEnvelope = {
  metadata?: Interactions.StepDelta["metadata"];
  usage?: InteractionUsage;
  // Existing Vertex/fixture envelopes may carry the ID outside the resource.
  interaction_id?: string;
};
export type InteractionEvent = (
  | OptionalLifecycleId<
      Exclude<Interactions.InteractionSSEEvent, { event_type: "step.start" | "error" }>
    >
  | (Omit<Interactions.StepStart, "step"> & { step: InteractionStartStep })
  // Preserve numeric provider errors already handled by the gateway/tests.
  | (Omit<Interactions.ErrorEvent, "error"> & {
      error?: Omit<Interactions.Error, "code"> & { code?: Interactions.Error["code"] | number };
    })
  | { event_type: "metadata"; event_id?: string }
) &
  UsageEnvelope;

export class InteractionsRequestError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InteractionsRequestError";
  }
}

export class InteractionsResponseError extends Error {
  constructor(
    message: string,
    readonly code?: string | number
  ) {
    super(message);
    this.name = "InteractionsResponseError";
  }
}
