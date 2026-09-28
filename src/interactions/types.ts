// Interactions V1 uses steps/step.delta. @google/genai 1.52's outputs/content.delta
// types describe a different wire contract; do not use them for this converter.
// Source: https://ai.google.dev/static/api/interactions-v1.openapi.json
export interface InteractionContent {
  type: string;
  text?: string;
  data?: string;
  uri?: string;
  mime_type?: string;
  annotations?: InteractionAnnotation[];
  [key: string]: unknown;
}

export interface InteractionAnnotation {
  type: string;
  url?: string;
  title?: string;
  start_index?: number;
  end_index?: number;
  [key: string]: unknown;
}

export interface InteractionStep {
  type: string;
  content?: InteractionContent[];
  summary?: InteractionContent[];
  signature?: string;
  id?: string;
  name?: string;
  arguments?: Record<string, unknown>;
  call_id?: string;
  result?: unknown;
  [key: string]: unknown;
}

export interface InteractionModalityTokens {
  modality?: string;
  tokens?: number;
  [key: string]: unknown;
}

export interface InteractionUsage {
  total_input_tokens?: number;
  total_output_tokens?: number;
  total_thought_tokens?: number;
  total_tool_use_tokens?: number;
  total_cached_tokens?: number;
  total_tokens?: number;
  input_tokens_by_modality?: InteractionModalityTokens[];
  output_tokens_by_modality?: InteractionModalityTokens[];
  cached_tokens_by_modality?: InteractionModalityTokens[];
  tool_use_tokens_by_modality?: InteractionModalityTokens[];
  grounding_tool_count?: { type?: string; count?: number; [key: string]: unknown }[];
  [key: string]: unknown;
}

export type InteractionStatus =
  | "in_progress"
  | "requires_action"
  | "completed"
  | "failed"
  | "cancelled"
  | "incomplete";

export interface Interaction {
  id: string;
  status: InteractionStatus;
  model?: string;
  created?: string;
  steps?: InteractionStep[];
  usage?: InteractionUsage;
  errors?: { code?: string | number; message?: string }[];
  [key: string]: unknown;
}

export interface InteractionEvent {
  event_type: string;
  event_id?: string;
  interaction?: Interaction;
  interaction_id?: string;
  status?: InteractionStatus;
  index?: number;
  step?: InteractionStep;
  delta?: InteractionContent & {
    arguments?: string;
    signature?: string;
    content?: InteractionContent;
  };
  metadata?: { total_usage?: InteractionUsage; [key: string]: unknown };
  usage?: InteractionUsage;
  step_usage?: InteractionUsage;
  error?: { code?: string | number; message?: string };
  [key: string]: unknown;
}

export interface InteractionGenerationConfig {
  max_output_tokens?: number;
  seed?: number;
  stop_sequences?: string[];
  thinking_level?: "minimal" | "low" | "medium" | "high";
  thinking_summaries?: "auto" | "none";
  tool_choice?:
    | "auto"
    | "any"
    | "none"
    | "validated"
    | { allowed_tools: { mode: "auto" | "any"; tools: string[] } };
  [key: string]: unknown;
}

export interface InteractionCreateParams {
  model: string;
  input: string | InteractionStep[];
  system_instruction?: string;
  generation_config?: InteractionGenerationConfig;
  tools?: { type: string; name?: string; description?: string; parameters?: object }[];
  response_format?: InteractionResponseFormat | InteractionResponseFormat[];
  store?: boolean;
  stream?: boolean;
  labels?: Record<string, string>;
}

export interface InteractionResponseFormat {
  type: string;
  mime_type?: string;
  schema?: object;
  [key: string]: unknown;
}

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
