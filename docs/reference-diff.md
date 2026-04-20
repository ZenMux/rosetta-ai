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
