# rosetta-ai

Universal translator between AI provider protocols.

Convert between OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Google Gemini APIs — with full support for tool calling, multimodal content, extended thinking, web search, grounding, and streaming.

## Install

```bash
npm install @zenmux/rosetta-ai
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
| `GeminiToChatCompletionConverter` | Google Gemini | OpenAI Chat Completions |
| `GeminiToMessagesConverter` | Google Gemini | Anthropic Messages |
| `GeminiToResponsesConverter` | Google Gemini | OpenAI Responses |

## Usage

### OpenAI Chat Completions → Anthropic Messages

```typescript
import { ChatCompletionToMessagesConverter } from "@zenmux/rosetta-ai";

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
import { MessagesToChatCompletionConverter } from "@zenmux/rosetta-ai";

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
import { ChatCompletionToResponsesConverter } from "@zenmux/rosetta-ai";

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
import { ResponsesToChatCompletionConverter } from "@zenmux/rosetta-ai";

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
import { MessagesToResponsesConverter } from "@zenmux/rosetta-ai";

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
import { ResponsesToMessagesConverter } from "@zenmux/rosetta-ai";

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
import { ChatCompletionToGeminiConverter } from "@zenmux/rosetta-ai";
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
import { MessagesToGeminiConverter } from "@zenmux/rosetta-ai";

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
import { ResponsesToGeminiConverter } from "@zenmux/rosetta-ai";

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

### Google Gemini → OpenAI Chat Completions

```typescript
import { GeminiToChatCompletionConverter } from "@zenmux/rosetta-ai";

const converter = new GeminiToChatCompletionConverter();

// Request: Gemini → CC (forward)
const ccRequest = converter.convertRequest(geminiRequest);

// Response: CC → Gemini (backward)
const geminiResponse = converter.convertResponse(ccResponse);

// Streaming: CC stream → Gemini stream (backward)
for await (const chunk of converter.convertStream(ccStream)) {
  // chunk is a Google GenerateContentResponse
}
```

### Google Gemini → Anthropic Messages

```typescript
import { GeminiToMessagesConverter } from "@zenmux/rosetta-ai";

const converter = new GeminiToMessagesConverter();

// Request: Gemini → Messages (forward)
const messagesRequest = converter.convertRequest(geminiRequest);

// Response: Messages → Gemini (backward)
const geminiResponse = converter.convertResponse(anthropicResponse);

// Streaming: Messages stream → Gemini stream (backward)
for await (const chunk of converter.convertStream(messagesStream)) {
  // chunk is a Google GenerateContentResponse
}
```

### Google Gemini → OpenAI Responses

```typescript
import { GeminiToResponsesConverter } from "@zenmux/rosetta-ai";

const converter = new GeminiToResponsesConverter();

// Request: Gemini → Responses (forward)
const responsesRequest = converter.convertRequest(geminiRequest);

// Response: Responses → Gemini (backward)
const geminiResponse = converter.convertResponse(responsesResponse);

// Streaming: Responses stream → Gemini stream (backward)
for await (const chunk of converter.convertStream(responsesStream)) {
  // chunk is a Google GenerateContentResponse
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

### Messages ↔ Responses

| Anthropic Messages | OpenAI Responses |
|---|---|
| `model` | `model` |
| `system` | `instructions` |
| `messages` (user/assistant with content blocks) | `input` (ResponseInputItem[]) |
| `max_tokens` | `max_output_tokens` |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `tools` (custom) | `tools` (function) |
| `tools` (web_search_20250305) | `tools` (web_search) |
| `tool_choice` (auto/any/none/tool) | `tool_choice` (auto/required/none/function) |
| `tool_choice.disable_parallel_tool_use` | `parallel_tool_calls` |
| `thinking` (enabled/disabled/adaptive) | `reasoning.effort` |
| `output_config.format` (json_schema) | `text.format` (json_schema) |
| `metadata.user_id` | `metadata.user_id` |
| `service_tier` | `service_tier` |
| `tool_use` blocks | `function_call` items |
| `tool_result` blocks | `function_call_output` items |
| `image` blocks (base64/url) | `input_image` parts |
| `document` blocks (base64/url/text) | `input_file` / `input_text` parts |

### Messages ↔ Gemini

| Anthropic Messages | Google Gemini |
|---|---|
| `model` | `model` |
| `system` | `config.systemInstruction` |
| `messages` (user/assistant with content blocks) | `contents` (user/model roles) |
| `max_tokens` | `config.maxOutputTokens` |
| `temperature` | `config.temperature` |
| `top_p` | `config.topP` |
| `top_k` | `config.topK` |
| `stop_sequences` | `config.stopSequences` |
| `tools` (custom) | `config.tools[].functionDeclarations` |
| `tools` (web_search_20250305) | `config.tools` (googleSearch) |
| `tool_choice` (auto/any/none/tool) | `config.toolConfig.functionCallingConfig` (AUTO/ANY/NONE) |
| `thinking` (enabled/disabled/adaptive) | `config.thinkingConfig` |
| `output_config.format` (json_schema) | `config.responseMimeType` + `config.responseJsonSchema` |
| `tool_use` blocks | `functionCall` parts |
| `tool_result` blocks | `functionResponse` parts |
| `image` blocks (base64/url) | `inlineData` / `fileData` |
| `document` blocks (base64/url/text) | `inlineData` / `fileData` / text part |
| `thinking` blocks (with signature) | `thought` parts (`thoughtSignature`) |

### Responses ↔ Gemini

| OpenAI Responses | Google Gemini |
|---|---|
| `model` | `model` |
| `instructions` | `config.systemInstruction` |
| `input` (ResponseInputItem[]) | `contents` (user/model roles) |
| `max_output_tokens` | `config.maxOutputTokens` |
| `temperature` | `config.temperature` |
| `top_p` | `config.topP` |
| `tools` (function) | `config.tools[].functionDeclarations` |
| `tools` (web_search) | `config.tools` (googleSearch) |
| `tool_choice` | `config.toolConfig.functionCallingConfig` (AUTO/ANY/NONE) |
| `reasoning.effort` | `config.thinkingConfig` |
| `text.format` (json_schema/json_object) | `config.responseMimeType` + `config.responseJsonSchema` |
| `include` (logprobs) | `config.responseLogprobs` / `config.logprobs` |
| `function_call` items | `functionCall` parts |
| `function_call_output` items | `functionResponse` parts |

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

Apache-2.0
