# rosetta-ai

Universal translator between AI provider protocols.

Convert requests, responses, and streaming chunks between OpenAI Chat Completions and Anthropic Messages APIs — with full support for tool calling, multimodal content, extended thinking, web search, and streaming.

## Install

```bash
npm install rosetta-ai
```

## Usage

### OpenAI Chat Completions → Anthropic Messages

```typescript
import { ChatCompletionToMessagesConverter } from "rosetta-ai";

const converter = new ChatCompletionToMessagesConverter();

// Convert request
const anthropicRequest = converter.convertRequest(openaiRequest);

// Convert response
const anthropicResponse = converter.convertResponse(openaiResponse);

// Convert stream (create new instance per stream)
const streamConverter = new ChatCompletionToMessagesConverter();
for await (const event of streamConverter.convertStream(openaiStream)) {
  // each event is an Anthropic RawMessageStreamEvent
}
```

### Anthropic Messages → OpenAI Chat Completions

```typescript
import { MessagesToChatCompletionConverter } from "rosetta-ai";

const converter = new MessagesToChatCompletionConverter();

// Convert request
const openaiRequest = converter.convertRequest(anthropicRequest);

// Convert response
const openaiResponse = converter.convertResponse(anthropicResponse);

// Convert stream (create new instance per stream)
const streamConverter = new MessagesToChatCompletionConverter();
for await (const chunk of streamConverter.convertStream(anthropicStream)) {
  // chunk is an OpenAI ChatCompletionChunk
}
```

## Supported Conversions

### Request Fields

| OpenAI | Anthropic | Direction |
|---|---|---|
| `model` | `model` | ↔ |
| `messages` (system/developer/user/assistant/tool) | `system` + `messages` (user/assistant with content blocks) | ↔ |
| `max_tokens` / `max_completion_tokens` | `max_tokens` | ↔ |
| `temperature` | `temperature` | ↔ |
| `top_p` | `top_p` | ↔ |
| `stop` | `stop_sequences` | ↔ |
| `stream` | `stream` | ↔ |
| `tools` (function) | `tools` (custom) | ↔ |
| `tool_choice` (auto/required/none/named) | `tool_choice` (auto/any/none/tool) | ↔ |
| `parallel_tool_calls` | `tool_choice.disable_parallel_tool_use` | ↔ |
| `response_format` (json_schema/json_object) | `output_config.format` | ↔ |
| `reasoning_effort` | `thinking` (enabled/disabled/adaptive) | ↔ |
| `user` | `metadata.user_id` | ↔ |
| `service_tier` | `service_tier` | ↔ |
| `web_search_options` | `tools` (web_search_20250305) | ↔ |

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

### Response Fields

| OpenAI | Anthropic |
|---|---|
| `finish_reason`: stop/tool_calls/length/content_filter | `stop_reason`: end_turn/tool_use/max_tokens/refusal |
| `usage.prompt_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` | `usage.output_tokens` |
| `prompt_tokens_details.cached_tokens` | `cache_read_input_tokens` |

### Streaming

Both converters handle chunk-by-chunk streaming conversion:

- **ChatCompletionToMessagesConverter**: OpenAI `ChatCompletionChunk` → Anthropic `RawMessageStreamEvent[]`
- **MessagesToChatCompletionConverter**: Anthropic `RawMessageStreamEvent` → OpenAI `ChatCompletionChunk | null`

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
