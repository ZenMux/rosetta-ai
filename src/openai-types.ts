import type OpenAI from "openai";

// Other providers may not report cache writes; unknown must not become a measured zero.
export type ResponseUsage = Omit<OpenAI.Responses.ResponseUsage, "input_tokens_details"> & {
  input_tokens_details: Omit<
    OpenAI.Responses.ResponseUsage["input_tokens_details"],
    "cache_write_tokens"
  > & { cache_write_tokens: number | null };
};

export type Response = Omit<OpenAI.Responses.Response, "usage"> & {
  usage?: ResponseUsage;
};

type CompatibleEvent<T> = T extends { response: OpenAI.Responses.Response }
  ? Omit<T, "response"> & { response: Response }
  : T extends OpenAI.Responses.ResponseFunctionCallArgumentsDoneEvent
    ? T & { name?: string }
    : T;

// Preserve the existing function name extension and the same usage contract in JSON and SSE.
export type ResponseStreamEvent = CompatibleEvent<OpenAI.Responses.ResponseStreamEvent>;

type WithServiceTier<T> = T extends unknown
  ? Omit<T, "service_tier"> & { service_tier?: OpenAI.Responses.Response["service_tier"] }
  : never;

// The converter forwards service tiers unchanged, including values absent from CC's enum.
export type ChatCompletionCreateParams =
  WithServiceTier<OpenAI.Chat.Completions.ChatCompletionCreateParams>;
export type ChatCompletion = WithServiceTier<OpenAI.ChatCompletion>;
export type ChatCompletionChunk = WithServiceTier<OpenAI.ChatCompletionChunk>;
