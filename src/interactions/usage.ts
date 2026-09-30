import type OpenAI from "openai";
import type Anthropic from "@anthropic-ai/sdk";
import type { InteractionModalityTokens, InteractionUsage } from "./types";

// A partial provider snapshot must stay partial. Filling absent counters with 0
// would turn an interrupted/incomplete usage report into apparently valid billing.
export interface InteractionCompletionUsage extends Partial<OpenAI.CompletionUsage> {
  interactions: InteractionUsage;
}

export interface InteractionResponseUsage extends Partial<OpenAI.Responses.ResponseUsage> {
  interactions: InteractionUsage;
}

export interface InteractionMessageUsage extends Partial<Anthropic.Usage> {
  interactions: InteractionUsage;
}

/** Messages separates uncached input from cache reads; native fee aliases keep the total. */
export function convertInteractionMessageUsage(
  usage: InteractionUsage | undefined
): InteractionMessageUsage | undefined {
  if (!usage) return undefined;
  const result: InteractionMessageUsage = { interactions: structuredClone(usage) };
  if (usage.total_input_tokens != null)
    result.input_tokens = usage.total_input_tokens - (usage.total_cached_tokens ?? 0);
  if (usage.total_cached_tokens != null) result.cache_read_input_tokens = usage.total_cached_tokens;
  if (usage.total_output_tokens != null && usage.total_thought_tokens != null)
    result.output_tokens = usage.total_output_tokens + usage.total_thought_tokens;
  return result;
}

export function convertInteractionResponseUsage(
  usage: InteractionUsage | undefined
): InteractionResponseUsage | null {
  if (!usage) return null;
  const result: InteractionResponseUsage = { interactions: structuredClone(usage) };
  if (usage.total_input_tokens != null) result.input_tokens = usage.total_input_tokens;
  if (usage.total_output_tokens != null && usage.total_thought_tokens != null)
    result.output_tokens = usage.total_output_tokens + usage.total_thought_tokens;
  if (usage.total_tokens != null) result.total_tokens = usage.total_tokens;
  if (usage.total_cached_tokens != null)
    result.input_tokens_details = { cached_tokens: usage.total_cached_tokens };
  if (usage.total_thought_tokens != null)
    result.output_tokens_details = { reasoning_tokens: usage.total_thought_tokens };
  return result;
}

function modalityTokens(
  items: InteractionModalityTokens[] | undefined,
  modality: string
): number | undefined {
  const matches = items?.filter(item => item.modality?.toLowerCase() === modality);
  if (!matches?.length || matches.some(item => item.tokens == null)) return undefined;
  return matches.reduce((sum, item) => sum + item.tokens!, 0);
}

export function convertInteractionUsage(
  usage: InteractionUsage | undefined
): InteractionCompletionUsage | undefined {
  if (!usage) return undefined;
  const result: InteractionCompletionUsage = { interactions: structuredClone(usage) };
  if (usage.total_input_tokens != null) result.prompt_tokens = usage.total_input_tokens;
  if (usage.total_output_tokens != null && usage.total_thought_tokens != null) {
    result.completion_tokens = usage.total_output_tokens + usage.total_thought_tokens;
  }
  if (usage.total_tokens != null) result.total_tokens = usage.total_tokens;

  // V1's function-call example reports input=100, output=25, tool-use=50,
  // total=125. Tool-use is preserved separately, never blindly added to input.
  const audioInput = modalityTokens(usage.input_tokens_by_modality, "audio");
  if (usage.total_cached_tokens != null || audioInput != null) {
    result.prompt_tokens_details = {
      ...(usage.total_cached_tokens != null && { cached_tokens: usage.total_cached_tokens }),
      ...(audioInput != null && { audio_tokens: audioInput }),
    };
  }
  const audioOutput = modalityTokens(usage.output_tokens_by_modality, "audio");
  if (usage.total_thought_tokens != null || audioOutput != null) {
    result.completion_tokens_details = {
      ...(usage.total_thought_tokens != null && { reasoning_tokens: usage.total_thought_tokens }),
      ...(audioOutput != null && { audio_tokens: audioOutput }),
    };
  }
  return result;
}
