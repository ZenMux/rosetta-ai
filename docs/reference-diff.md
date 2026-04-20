# Conversion Differences

## Anthropic Messages -> OpenAI Chat Completions

### Response Conversion

#### `content` field when no text blocks

| | Behavior |
|---|---|
| **Reference** | Always `string` — starts as `""`, concatenated with `+=`, so tool-only responses have `content: ""` |
| **rosetta-ai** | `string \| null` — returns `null` when no text blocks exist |

**Rationale:** OpenAI's spec defines `content` as `string | null`, and the OpenAI API returns `null` for pure tool_call responses. rosetta-ai follows the spec.

#### Custom tool type passthrough

| | Behavior |
|---|---|
| **Reference** | Looks up the original tool definition from request params to decide `type: "function"` vs `type: "custom"` on tool_calls |
| **rosetta-ai** | Always outputs `type: "function"` for all tool_use / server_tool_use blocks |

**Rationale:** rosetta-ai's `convertResponse` is stateless — it doesn't receive the original request params. The reference can do this because it holds both request and response in the same adapter class. For a standalone converter library, always using `type: "function"` is the safe default since most OpenAI clients only expect function tool calls.

### Stream Conversion

#### `message_start` event handling

| | Behavior |
|---|---|
| **Reference** | Extracts usage into context, yields **nothing** (no chunk emitted) |
| **rosetta-ai** | Extracts usage into state, emits a chunk with `delta: { role: "assistant" }` |

**Rationale:** OpenAI's Chat Completions streaming sends an initial chunk with `{ delta: { role: "assistant", content: "" } }` before content starts. rosetta-ai emits this to match OpenAI's actual streaming behavior. The reference skips it, which means clients don't receive the role signal until the first content_block event.

### Request Conversion

#### `cache_control` on content blocks

| | Behavior |
|---|---|
| **Reference** | Passes `cache_control` from OpenAI content parts through to Anthropic blocks via `setCacheControl()` |
| **rosetta-ai** | Drops `cache_control` — no OpenAI equivalent, and Anthropic → OpenAI direction has nothing to preserve |

**Rationale:** `cache_control` is Anthropic-specific. In the Anthropic → OpenAI direction there is no target field. In the OpenAI → Anthropic direction, the OpenAI SDK types don't include `cache_control` on content parts — the reference uses untyped `Object.hasOwn()` checks to pass it through. rosetta-ai does not support untyped field passthrough.

#### `reasoning` field name on request messages

| | Behavior |
|---|---|
| **Reference** | Reads `msg.reasoning \|\| msg.reasoning_content` (supports both field names) for assistant messages |
| **rosetta-ai** | Reads `block.type === 'thinking'` from Anthropic content blocks (converting Anthropic → OpenAI) |

**Rationale:** The reference handles OpenAI → Anthropic direction where `reasoning` is an extension field on OpenAI assistant messages. rosetta-ai converts from Anthropic's typed `thinking` blocks, which is structurally correct.

#### Error handling

| | Behavior |
|---|---|
| **Reference** | Throws `BadRequestError` for unsupported content types and tools |
| **rosetta-ai** | Silently drops unsupported content types, skips unsupported tools |

**Rationale:** The reference is a runtime gateway adapter that validates requests. rosetta-ai is a conversion library — callers are responsible for validation. Throwing from a converter makes error handling harder for consumers.

## OpenAI Chat Completions -> Anthropic Messages

### Response Conversion

#### `thinking` block signature

| | Behavior |
|---|---|
| **Reference** | Creates thinking block with `signature: ""` (empty string) |
| **rosetta-ai** | Same — `signature: ""` |

No difference. Both use empty signature since OpenAI doesn't provide one.

#### `web_search_tool_result` tool_use_id generation

| | Behavior |
|---|---|
| **Reference** | Uses `crypto.randomUUID()` — `websearch_${crypto.randomUUID()}` |
| **rosetta-ai** | Uses `Math.random().toString(36)` — `websearch_${randomId}` |

**Rationale:** rosetta-ai avoids a Node.js `crypto` dependency for a non-critical field. The tool_use_id is synthetic (OpenAI doesn't provide one) and only needs uniqueness, not cryptographic randomness.

#### Content block ordering

| | Behavior |
|---|---|
| **Reference** | Order: thinking → web_search_tool_result → text → tool_use |
| **rosetta-ai** | Same order |

No difference. Both follow the reference ordering.

### Stream Conversion

#### First chunk behavior

| | Behavior |
|---|---|
| **Reference** | First chunk emits `message_start` with skeleton message — no `role: "assistant"` chunk |
| **rosetta-ai** | First chunk emits `message_start` with `role: "assistant"` in the message |

**Rationale:** Anthropic's streaming spec sends `message_start` with a full message object including `role: "assistant"`. rosetta-ai matches this spec. The reference also includes `role: "assistant"` in the message_start message — both behave the same.

#### Text block auto-creation on first chunk

| | Behavior |
|---|---|
| **Reference** | Defers `content_block_start` until the first content delta arrives, tracks `contentStartType` to determine block type |
| **rosetta-ai** | Same — defers `content_block_start`, uses `transitionBlock()` with `currentBlockType` tracking |

No difference. Both defer content block creation.

#### `reasoning_content` field support

| | Behavior |
|---|---|
| **Reference** | Checks both `delta.reasoning` and `delta.reasoning_content` (some providers use alternate field name) |
| **rosetta-ai** | Same — checks both via `hasReasoning()` / `getReasoning()` helpers |

No difference.

### Request Conversion

#### System message content blocks

| | Behavior |
|---|---|
| **Reference** | Keeps system `TextBlockParam[]` as OpenAI content parts array: `{ role: "system", content: [{type: "text", text: "..."}] }` |
| **rosetta-ai** | Joins system `TextBlockParam[]` into single string with `\n`: `{ role: "system", content: "line1\nline2" }` |

**Rationale:** OpenAI's system message `content` supports both `string` and `Array<ContentPartText>`. The reference preserves the array structure. rosetta-ai collapses to a string for broader client compatibility — some OpenAI-compatible providers only accept string content for system messages.

#### `thinking` blocks in assistant request messages

| | Behavior |
|---|---|
| **Reference** | Converts `thinking` blocks to text content parts in the assistant message |
| **rosetta-ai** | Converts `thinking` blocks to `reasoning` + `reasoning_details` extension fields on the assistant message |

**Rationale:** The reference treats thinking as plain text, losing the structured reasoning metadata. rosetta-ai preserves the reasoning structure using the `reasoning`/`reasoning_details` extension fields that OpenAI-compatible reasoning models (o1, o3, deepseek-r1) understand. This enables round-trip fidelity when the downstream provider also supports reasoning.

#### `additionalBody` / `additionalHeaders` passthrough

| | Behavior |
|---|---|
| **Reference** | Supports `additionalBody` and `additionalHeaders` for arbitrary parameter injection |
| **rosetta-ai** | Not supported |

**Rationale:** `additionalBody`/`additionalHeaders` is a gateway-specific escape hatch for provider-specific parameters that don't map to standard fields. A conversion library should not support untyped passthrough — callers can merge additional fields after conversion.
