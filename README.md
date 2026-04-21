# rosetta-ai

Universal translator between AI provider protocols.

Convert between OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Google Gemini APIs — with full support for tool calling, multimodal content, extended thinking, web search, grounding, and streaming.

## Install

```bash
npm install rosetta-ai
```

## How It Works

Each converter follows the **gateway pattern**: request goes forward (user protocol → backend protocol), while response and stream go backward (backend → user).

```
User ──request──→ [Converter] ──request──→ Backend
User ←─response── [Converter] ←─response── Backend
User ←──stream─── [Converter] ←──stream─── Backend
```

One converter instance handles the full round-trip for a single gateway route.

## Converters

| Converter | User Protocol | Backend Protocol |
|---|---|---|
| `ChatCompletionToMessagesConverter` | OpenAI Chat Completions | Anthropic Messages |
| `ChatCompletionToResponsesConverter` | OpenAI Chat Completions | OpenAI Responses |
| `ChatCompletionToGeminiConverter` | OpenAI Chat Completions | Google Gemini |
| `MessagesToChatCompletionConverter` | Anthropic Messages | OpenAI Chat Completions |
| `MessagesToResponsesConverter` | Anthropic Messages | OpenAI Responses |
| `MessagesToGeminiConverter` | Anthropic Messages | Google Gemini |
| `ResponsesToChatCompletionConverter` | OpenAI Responses | OpenAI Chat Completions |
| `ResponsesToMessagesConverter` | OpenAI Responses | Anthropic Messages |
| `ResponsesToGeminiConverter` | OpenAI Responses | Google Gemini |

## Usage

### OpenAI Chat Completions → Anthropic Messages

```typescript
import { ChatCompletionToMessagesConverter } from "rosetta-ai";

const converter = new ChatCompletionToMessagesConverter();

// Request: CC → Messages (forward)
const anthropicRequest = converter.convertRequest(openaiRequest);

// Call the backend
const anthropicResponse = await anthropicClient.messages.create(anthropicRequest);

// Response: Messages → CC (backward)
const openaiResponse = converter.convertResponse(anthropicResponse);

// Streaming: Messages stream → CC stream (backward)
const stream = await anthropicClient.messages.create({ ...anthropicRequest, stream: true });
for await (const chunk of converter.convertStream(stream)) {
  // chunk is an OpenAI ChatCompletionChunk
}
```

### Anthropic Messages → OpenAI Chat Completions

```typescript
import { MessagesToChatCompletionConverter } from "rosetta-ai";

const converter = new MessagesToChatCompletionConverter();

// Request: Messages → CC (forward)
const openaiRequest = converter.convertRequest(anthropicRequest);

// Call the backend
const openaiResponse = await openaiClient.chat.completions.create(openaiRequest);

// Response: CC → Messages (backward)
const anthropicResponse = converter.convertResponse(openaiResponse);

// Streaming: CC stream → Messages stream (backward)
const stream = await openaiClient.chat.completions.create({ ...openaiRequest, stream: true });
for await (const event of converter.convertStream(stream)) {
  // event is an Anthropic RawMessageStreamEvent
}
```

### OpenAI Chat Completions → OpenAI Responses

```typescript
import { ChatCompletionToResponsesConverter } from "rosetta-ai";

const converter = new ChatCompletionToResponsesConverter();

// Request: CC → Responses (forward)
const responsesRequest = converter.convertRequest(ccRequest);

// Response: Responses → CC (backward)
const ccResponse = converter.convertResponse(responsesResponse);

// Streaming: Responses stream → CC stream (backward)
for await (const chunk of converter.convertStream(responsesStream)) {
  // chunk is an OpenAI ChatCompletionChunk
}
```

### OpenAI Responses → OpenAI Chat Completions

```typescript
import { ResponsesToChatCompletionConverter } from "rosetta-ai";

const converter = new ResponsesToChatCompletionConverter();

// Request: Responses → CC (forward)
const ccRequest = converter.convertRequest(responsesRequest);

// Response: CC → Responses (backward)
const responsesResponse = converter.convertResponse(ccResponse);

// Streaming: CC stream → Responses stream (backward)
for await (const event of converter.convertStream(ccStream)) {
  // event is a Responses ResponseStreamEvent
}
```

### Anthropic Messages → OpenAI Responses

```typescript
import { MessagesToResponsesConverter } from "rosetta-ai";

const converter = new MessagesToResponsesConverter();

// Request: Messages → Responses (forward)
const responsesRequest = converter.convertRequest(messagesRequest);

// Response: Responses → Messages (backward)
const messagesResponse = converter.convertResponse(responsesResponse);

// Streaming: Responses stream → Messages stream (backward)
for await (const event of converter.convertStream(responsesStream)) {
  // event is an Anthropic RawMessageStreamEvent
}
```

### OpenAI Responses → Anthropic Messages

```typescript
import { ResponsesToMessagesConverter } from "rosetta-ai";

const converter = new ResponsesToMessagesConverter();

// Request: Responses → Messages (forward)
const messagesRequest = converter.convertRequest(responsesRequest);

// Response: Messages → Responses (backward)
const responsesResponse = converter.convertResponse(anthropicResponse);

// Streaming: Messages stream → Responses stream (backward)
for await (const event of converter.convertStream(messagesStream)) {
  // event is a Responses ResponseStreamEvent
}
```

### OpenAI Chat Completions → Google Gemini

```typescript
import { ChatCompletionToGeminiConverter } from "rosetta-ai";
import { GoogleGenAI } from "@google/genai";

const converter = new ChatCompletionToGeminiConverter();
const genai = new GoogleGenAI({ apiKey: "..." });

// Request: CC → Gemini (forward)
const geminiRequest = converter.convertRequest(openaiRequest);

// Call the backend
const geminiResponse = await genai.models.generateContent(geminiRequest);

// Response: Gemini → CC (backward)
const openaiResponse = converter.convertResponse(geminiResponse);

// Streaming: Gemini stream → CC stream (backward)
const stream = await genai.models.generateContentStream(geminiRequest);
for await (const chunk of converter.convertStream(stream)) {
  // chunk is an OpenAI ChatCompletionChunk
}
```

### Anthropic Messages → Google Gemini

```typescript
import { MessagesToGeminiConverter } from "rosetta-ai";

const converter = new MessagesToGeminiConverter();

// Request: Messages → Gemini (forward)
const geminiRequest = converter.convertRequest(anthropicRequest);

// Response: Gemini → Messages (backward)
const anthropicResponse = converter.convertResponse(geminiResponse);

// Streaming: Gemini stream → Messages stream (backward)
for await (const event of converter.convertStream(geminiStream)) {
  // event is an Anthropic RawMessageStreamEvent
}
```

### OpenAI Responses → Google Gemini

```typescript
import { ResponsesToGeminiConverter } from "rosetta-ai";

const converter = new ResponsesToGeminiConverter();

// Request: Responses → Gemini (forward)
const geminiRequest = converter.convertRequest(responsesRequest);

// Response: Gemini → Responses (backward)
const responsesResponse = converter.convertResponse(geminiResponse);

// Streaming: Gemini stream → Responses stream (backward)
for await (const event of converter.convertStream(geminiStream)) {
  // event is a Responses ResponseStreamEvent
}
```

## Supported Conversions

### Chat Completions ↔ Messages

| OpenAI Chat Completions | Anthropic Messages |
|---|---|
| `model` | `model` |
| `messages` (system/developer/user/assistant/tool) | `system` + `messages` (user/assistant with content blocks) |
| `max_tokens` / `max_completion_tokens` | `max_tokens` |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `stop` | `stop_sequences` |
| `tools` (function) | `tools` (custom) |
| `tool_choice` (auto/required/none/named) | `tool_choice` (auto/any/none/tool) |
| `parallel_tool_calls` | `tool_choice.disable_parallel_tool_use` |
| `response_format` (json_schema/json_object) | `output_config.format` |
| `reasoning_effort` | `thinking` (enabled/disabled/adaptive) |
| `user` | `metadata.user_id` |
| `service_tier` | `service_tier` |
| `web_search_options` | `tools` (web_search_20250305) |

### Chat Completions ↔ Responses

| OpenAI Chat Completions | OpenAI Responses |
|---|---|
| `messages` | `input` (ResponseInputItem[]) |
| `system` / `developer` messages | `instructions` |
| `max_completion_tokens` | `max_output_tokens` |
| `tools` (function) | `tools` (function) |
| `tool_choice` | `tool_choice` |
| `response_format` | `text.format` |
| `reasoning_effort` | `reasoning.effort` |
| `web_search_options` | `tools` (web_search) |
| `parallel_tool_calls` | `parallel_tool_calls` |
| `prompt_cache_key` / `prompt_cache_retention` | `prompt_cache_key` / `prompt_cache_retention` |

### Chat Completions → Gemini

| OpenAI Chat Completions | Google Gemini |
|---|---|
| `model` | `model` |
| `messages` (system/developer) | `config.systemInstruction` |
| `messages` (user/assistant/tool) | `contents` (user/model roles) |
| `max_tokens` / `max_completion_tokens` | `config.maxOutputTokens` |
| `temperature` | `config.temperature` |
| `top_p` | `config.topP` |
| `stop` | `config.stopSequences` |
| `tools` (function) | `config.tools[].functionDeclarations` |
| `tool_choice` (auto/required/none/named) | `config.toolConfig.functionCallingConfig` (AUTO/ANY/NONE) |
| `response_format` (json_schema/json_object) | `config.responseMimeType` + `config.responseJsonSchema` |
| `reasoning_effort` | `config.thinkingConfig` |
| `web_search_options` | `config.tools` (googleSearch) |
| `seed` | `config.seed` |
| `frequency_penalty` / `presence_penalty` | `config.frequencyPenalty` / `config.presencePenalty` |
| `logprobs` / `top_logprobs` | `config.responseLogprobs` / `config.logprobs` |
| `tool_calls` on assistant | `functionCall` parts on model content |
| `tool` role messages | `functionResponse` parts on user content |
| `image_url` (base64) | `inlineData` |
| `image_url` (URL) | `fileData` |
| `file` content part | `inlineData` / `fileData` |
| `input_audio` | `inlineData` (audio mime) |

### Gemini Response / Stream

| Google Gemini | OpenAI / Anthropic |
|---|---|
| `candidates[].finishReason` STOP | `finish_reason: "stop"` / `stop_reason: "end_turn"` |
| `candidates[].finishReason` MAX_TOKENS | `finish_reason: "length"` / `stop_reason: "max_tokens"` |
| `candidates[].finishReason` SAFETY | `finish_reason: "content_filter"` / `stop_reason: "refusal"` |
| `functionCall` parts | `tool_calls` / `tool_use` blocks / `function_call` output items |
| `thought` parts | `reasoning` / `thinking` blocks / `reasoning` output items |
| `groundingMetadata.groundingChunks` | `annotations` (url_citation) / `web_search_tool_result` blocks |
| `usageMetadata.promptTokenCount` | `prompt_tokens` / `input_tokens` |
| `usageMetadata.candidatesTokenCount` | `completion_tokens` / `output_tokens` |
| `usageMetadata.cachedContentTokenCount` | `cached_tokens` / `cache_read_input_tokens` |
| `usageMetadata.thoughtsTokenCount` | `reasoning_tokens` |
| `usageMetadata.toolUsePromptTokenCount` | added to `prompt_tokens` / `input_tokens` |

### Content Types

| OpenAI | Anthropic |
|---|---|
| Text content | `text` blocks |
| `image_url` (URL or base64 data URI) | `image` blocks (url or base64 source) |
| `file` content part | `document` blocks (base64/url/text source) |
| `tool_calls` on assistant message | `tool_use` blocks |
| `tool` role messages | `tool_result` blocks in user message |
| `reasoning` / `reasoning_content` | `thinking` blocks |
| `annotations` (url_citation) | `web_search_tool_result` blocks |

### Response / Stream

| OpenAI | Anthropic |
|---|---|
| `finish_reason`: stop/tool_calls/length/content_filter | `stop_reason`: end_turn/tool_use/max_tokens/refusal |
| `usage.prompt_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` | `usage.output_tokens` |
| `prompt_tokens_details.cached_tokens` | `cache_read_input_tokens` |

Streaming includes support for text, tool calls, extended thinking, web search citations, and usage reporting.

## Development

```bash
npm install          # Install dependencies
npm run typecheck    # Type check
npm run lint         # Lint
npm run test         # Run tests
npm run build        # Build
```

## License

MIT
