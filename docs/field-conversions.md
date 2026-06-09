# Field Conversions Reference

Complete mapping of all fields across the 12 pairwise converters between OpenAI Chat Completions (CC), Anthropic Messages, OpenAI Responses, and Google Gemini.

## 1. CC → Messages (ChatCompletionToMessagesConverter)

### Request (CC → Messages)

| CC Field | Messages Field | Notes |
|---|---|---|
| `model` | `model` | |
| `messages` (system/developer) | `system` | Joined as text blocks |
| `messages` (user) | `messages[]` (user) | Text, image blocks (`file` parts → unsupported text placeholder) |
| `messages` (assistant) | `messages[]` (assistant) | Text, tool_use, thinking blocks |
| `messages` (tool) | `messages[]` (user tool_result) | |
| `max_tokens` / `max_completion_tokens` | `max_tokens` | Default 4096 |
| `temperature` | `temperature` | |
| `top_p` | `top_p` | |
| `stop` | `stop_sequences` | String → array |
| `tools` (function) | `tools` (input_schema) | |
| `web_search_options` | `tools` (web_search_20250305) | |
| `tool_choice` | `tool_choice` | auto/required/none → auto/any/none |
| `parallel_tool_calls` | `tool_choice.disable_parallel_tool_use` | Inverted |
| `response_format` (json_schema) | `output_config.format` | |
| `reasoning_effort` | `thinking` | Budget mapped: low→2048, medium→5120, high→10240 |
| `user` | `metadata.user_id` | |
| `service_tier` | `service_tier` | auto/standard_only only |
| `stream` | `stream` | |

**Not converted:** `frequency_penalty`, `presence_penalty`, `seed`, `n`, `logprobs`, `top_logprobs`, `logit_bias`

### Response (Messages → CC)

| Messages Field | CC Field | Notes |
|---|---|---|
| `id` | `id` | |
| `model` | `model` | |
| `content[]` text | `choices[0].message.content` | Joined |
| `content[]` thinking | `choices[0].message.reasoning` + `reasoning_details` | With signature |
| `content[]` tool_use | `choices[0].message.tool_calls` | |
| `content[]` web_search_tool_result | `choices[0].message.annotations` | url_citation |
| `stop_reason` | `finish_reason` | end_turn→stop, tool_use→tool_calls, max_tokens→length, refusal→content_filter |
| `usage.input_tokens` | `usage.prompt_tokens` | |
| `usage.output_tokens` | `usage.completion_tokens` | |
| `usage.cache_read_input_tokens` | `prompt_tokens_details.cached_tokens` | |
| `usage.server_tool_use` | `prompt_tokens_details.web_search` | |

### Stream: text ✅ tool_calls ✅ thinking ✅ annotations ✅ usage ✅

---

## 2. CC → Responses (ChatCompletionToResponsesConverter)

### Request (CC → Responses)

| CC Field | Responses Field | Notes |
|---|---|---|
| `model` | `model` | |
| `messages` | `input` | Converted to ResponseInputItem[] |
| `max_completion_tokens` | `max_output_tokens` | |
| `temperature` | `temperature` | |
| `top_p` | `top_p` | |
| `tools` (function) | `tools` (function) | |
| `web_search_options` | `tools` (web_search) | |
| `tool_choice` | `tool_choice` | |
| `parallel_tool_calls` | `parallel_tool_calls` | |
| `response_format` | `text.format` | |
| `reasoning_effort` | `reasoning.effort` | |
| `top_logprobs` | `include` (logprobs) | |
| `metadata` | `metadata` | |
| `service_tier` | `service_tier` | |
| `prompt_cache_key` | `prompt_cache_key` | |
| `prompt_cache_retention` | `prompt_cache_retention` | |
| `stream` | `stream` + `stream_options` | |

**Not converted:** `frequency_penalty`, `presence_penalty`, `seed`, `n`, `logprobs`, `stop`

### Response (Responses → CC)

| Responses Field | CC Field | Notes |
|---|---|---|
| `id` | `id` | |
| `model` | `model` | |
| `created_at` | `created` | |
| `service_tier` | `service_tier` | |
| `output[]` reasoning | `choices[0].message.reasoning` + `reasoning_details` | With encrypted_content |
| `output[]` function_call | `choices[0].message.tool_calls` | |
| `output[]` message (output_text) | `choices[0].message.content` | |
| `output[]` message (refusal) | `choices[0].message.refusal` | |
| `output[]` message (annotations) | `choices[0].message.annotations` | |
| `output[]` web_search_call | `prompt_tokens_details.web_search` | Count |
| `status` | `finish_reason` | completed→stop, incomplete→length, failed→stop; incomplete + `incomplete_details.reason: content_filter`→content_filter |
| `usage.*` | `usage.*` | cached_tokens, reasoning_tokens mapped |

### Stream: text ✅ tool_calls ✅ reasoning ✅ annotations ✅ usage ✅

---

## 3. CC → Gemini (ChatCompletionToGeminiConverter)

### Request (CC → Gemini)

| CC Field | Gemini Field | Notes |
|---|---|---|
| `model` | `model` | |
| `messages` (system/developer) | `config.systemInstruction` | |
| `messages` (user/assistant/tool) | `contents` | user/model roles |
| `max_tokens` / `max_completion_tokens` | `config.maxOutputTokens` | |
| `temperature` | `config.temperature` | |
| `top_p` | `config.topP` | |
| `stop` | `config.stopSequences` | |
| `seed` | `config.seed` | |
| `frequency_penalty` | `config.frequencyPenalty` | |
| `presence_penalty` | `config.presencePenalty` | |
| `n` | `config.candidateCount` | |
| `logprobs` | `config.responseLogprobs` | |
| `top_logprobs` | `config.logprobs` | |
| `tools` (function) | `config.tools[].functionDeclarations` | parametersJsonSchema |
| `web_search_options` | `config.tools[].googleSearch` | |
| `tool_choice` | `config.toolConfig.functionCallingConfig` | auto/required/none → AUTO/ANY/NONE |
| `response_format` (json_schema) | `config.responseMimeType` + `config.responseJsonSchema` | |
| `response_format` (json_object) | `config.responseMimeType` | application/json |
| `reasoning_effort` | `config.thinkingConfig` | Budget mapped |
| `image_url` (base64) | `inlineData` | |
| `image_url` (URL) | `fileData` | |
| `file` content | `inlineData` / `fileData` | |
| `input_audio` | `inlineData` | Audio mime |

**Not converted:** `logit_bias`, `user`

### Response (Gemini → CC)

| Gemini Field | CC Field | Notes |
|---|---|---|
| `responseId` | `id` | |
| `modelVersion` | `model` | |
| `candidates[0].content.parts[]` text | `choices[0].message.content` | |
| `candidates[0].content.parts[]` thought | `choices[0].message.reasoning` + `reasoning_details` | |
| `candidates[0].content.parts[]` functionCall | `choices[0].message.tool_calls` | Override finish_reason→tool_calls |
| `candidates[0].groundingMetadata` | `choices[0].message.annotations` | url_citation |
| `candidates[0].finishReason` | `finish_reason` | STOP→stop, MAX_TOKENS→length, SAFETY→content_filter |
| `usageMetadata.promptTokenCount` + `toolUsePromptTokenCount` | `usage.prompt_tokens` | Combined |
| `usageMetadata.candidatesTokenCount` + `thoughtsTokenCount` | `usage.completion_tokens` | Combined |
| `usageMetadata.cachedContentTokenCount` | `prompt_tokens_details.cached_tokens` | |
| `usageMetadata.thoughtsTokenCount` | `completion_tokens_details.reasoning_tokens` | |

### Stream: text ✅ tool_calls ✅ reasoning ✅ grounding ✅ usage ✅

---

## 4. Messages → CC (MessagesToChatCompletionConverter)

### Request (Messages → CC)

| Messages Field | CC Field | Notes |
|---|---|---|
| `model` | `model` | |
| `system` | `messages[]` (system) | String or TextBlockParam[], joined as text |
| `messages` (user) | `messages[]` (user) | Text, image, document, tool_result |
| `messages` (assistant) | `messages[]` (assistant) | Text, thinking, tool_use |
| `max_tokens` | `max_completion_tokens` | `max_tokens` deprecated; reasoning models require `max_completion_tokens` |
| `temperature` | `temperature` | |
| `top_p` | `top_p` | |
| `stop_sequences` | `stop` | |
| `tools` (input_schema) | `tools` (function) | |
| `tools` (web_search_20250305 / web_search_20260209) | `web_search_options` | max_uses, user_location mapped |
| `tool_choice` | `tool_choice` + `parallel_tool_calls` | auto/any/none → auto/required/none |
| `output_config` (json_schema) | `response_format` | |
| `thinking` | `reasoning_effort` | disabled→none, enabled→low/medium/high |
| `metadata.user_id` | `user` | |
| `service_tier` | `service_tier` | standard_only→default |
| `stream` | `stream` + `stream_options` | `stream_options.include_usage: true` |

**Document block sources:** base64/url → `file` content part; text → text part; `content` source → nested text/image parts unpacked

**tool_result content blocks:** text kept; `search_result` → inner text flattened; `tool_reference` → tool_name; image/document skipped (unsupported by tool role)

**Not converted:** `top_k`, Anthropic-specific metadata fields

### Response (CC → Messages)

| CC Field | Messages Field | Notes |
|---|---|---|
| `id` | `id` | |
| `model` | `model` | |
| `choices[0].message.content` | `content[]` text | |
| `choices[0].message.reasoning` | `content[]` thinking | With signature="" |
| `choices[0].message.annotations` | `content[]` web_search_tool_result | |
| `choices[0].message.tool_calls` | `content[]` tool_use | |
| `finish_reason` | `stop_reason` | stop→end_turn, tool_calls→tool_use, length→max_tokens, content_filter→refusal |
| `usage.prompt_tokens` | `usage.input_tokens` | |
| `usage.completion_tokens` | `usage.output_tokens` | |
| `prompt_tokens_details.cached_tokens` | `cache_read_input_tokens` | |
| `prompt_tokens_details.web_search` | `server_tool_use.web_search_requests` | |

Usage also re-emits enriched OpenAI-style fields (`prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_tokens_details`, `completion_tokens_details`, audio token fields); origin `*_details` objects are merged in to preserve extra fields. Streaming reuses the same `buildUsage`, so `message_delta` carries real `output_tokens` from the trailing usage chunk.

### Stream: text ✅ tool_calls ✅ thinking ✅ web_search_tool_result ✅ usage ✅

---

## 5. Messages → Responses (MessagesToResponsesConverter)

### Request (Messages → Responses)

| Messages Field | Responses Field | Notes |
|---|---|---|
| `model` | `model` | |
| `system` | `instructions` | Joined text |
| `messages` | `input` | Converted to ResponseInputItem[] |
| `max_tokens` | `max_output_tokens` | |
| `temperature` | `temperature` | |
| `top_p` | `top_p` | |
| `tools` (input_schema) | `tools` (function) | |
| `tools` (web_search) | `tools` (web_search) | |
| `tool_choice` | `tool_choice` + `parallel_tool_calls` | auto/any/none → auto/required/none |
| `thinking` | `reasoning` | Budget→effort mapped |
| `output_config` (json_schema) | `text.format` | |
| `metadata.user_id` | `metadata.user_id` | |
| `service_tier` | `service_tier` | |
| `stream` | `stream` | |

**Not converted:** `stop_sequences`, `top_k`

### Response (Responses → Messages)

| Responses Field | Messages Field | Notes |
|---|---|---|
| `id` | `id` | |
| `model` | `model` | |
| `output[]` reasoning | `content[]` thinking | |
| `output[]` web_search_call | tracked for `server_tool_use` | |
| `output[]` function_call | `content[]` tool_use | |
| `output[]` message | `content[]` text | |
| `status` | `stop_reason` | completed→end_turn/tool_use, incomplete→max_tokens, failed→end_turn |
| `usage.*` | `usage.*` | cached_tokens→cache_read_input_tokens |

### Stream: text ✅ tool_calls ✅ thinking ✅ usage ✅

---

## 6. Messages → Gemini (MessagesToGeminiConverter)

### Request (Messages → Gemini)

| Messages Field | Gemini Field | Notes |
|---|---|---|
| `model` | `model` | |
| `system` | `config.systemInstruction` | String or TextBlockParam[] |
| `messages` (user) | `contents[]` (user) | Text, image→inlineData/fileData, document→inlineData/fileData, tool_result→functionResponse |
| `messages` (assistant) | `contents[]` (model) | Text, tool_use→functionCall |
| `max_tokens` | `config.maxOutputTokens` | |
| `temperature` | `config.temperature` | |
| `top_p` | `config.topP` | |
| `top_k` | `config.topK` | |
| `stop_sequences` | `config.stopSequences` | |
| `tools` (input_schema) | `config.tools[].functionDeclarations` | parametersJsonSchema |
| `tools` (web_search) | `config.tools[].googleSearch` | |
| `tool_choice` | `config.toolConfig.functionCallingConfig` | auto/any/none → AUTO/ANY/NONE |
| `thinking` | `config.thinkingConfig` | disabled→budget:0, enabled→budget passthrough |
| `output_config` (json_schema) | `config.responseMimeType` + `config.responseJsonSchema` | |

**Not converted:** `metadata`

### Response (Gemini → Messages)

| Gemini Field | Messages Field | Notes |
|---|---|---|
| `responseId` | `id` | |
| `modelVersion` | `model` | |
| `candidates[0].content.parts[]` thought | `content[]` thinking | With signature |
| `candidates[0].groundingMetadata` | `content[]` web_search_tool_result | |
| `candidates[0].content.parts[]` text | `content[]` text | |
| `candidates[0].content.parts[]` functionCall | `content[]` tool_use | |
| `candidates[0].finishReason` | `stop_reason` | STOP→end_turn, MAX_TOKENS→max_tokens, SAFETY→refusal |
| `usageMetadata.promptTokenCount` + `toolUsePromptTokenCount` | `usage.input_tokens` | Combined |
| `usageMetadata.candidatesTokenCount` + `thoughtsTokenCount` | `usage.output_tokens` | Combined |
| `usageMetadata.cachedContentTokenCount` | `cache_read_input_tokens` | |

### Stream: text ✅ tool_calls ✅ thinking ✅ web_search_tool_result ✅ usage ✅

---

## 7. Responses → CC (ResponsesToChatCompletionConverter)

### Request (Responses → CC)

| Responses Field | CC Field | Notes |
|---|---|---|
| `model` | `model` | |
| `instructions` | `messages[]` (system) | |
| `input` | `messages` | Converted from ResponseInputItem[] |
| `max_output_tokens` | `max_completion_tokens` | |
| `temperature` | `temperature` | |
| `top_p` | `top_p` | |
| `tools` (function) | `tools` (function) | |
| `tools` (web_search) | `web_search_options` | |
| `tool_choice` | `tool_choice` | |
| `parallel_tool_calls` | `parallel_tool_calls` | |
| `reasoning.effort` | `reasoning_effort` | |
| `text.format` | `response_format` | |
| `text.verbosity` | `verbosity` | |
| `metadata` | `metadata` | |
| `service_tier` | `service_tier` | |
| `prompt_cache_key` | `prompt_cache_key` | |
| `prompt_cache_retention` | `prompt_cache_retention` | |
| `safety_identifier` | `safety_identifier` | |
| `include` (logprobs) | `top_logprobs` | Set to 20 |
| `stream` | `stream` + `stream_options` | |

**Not converted:** `previous_response_id`

### Response (CC → Responses)

| CC Field | Responses Field | Notes |
|---|---|---|
| `id` | `id` | |
| `model` | `model` | |
| `created` | `created_at` | |
| `choices[0].message.reasoning` | `output[]` reasoning | summary_text |
| `choices[0].message.tool_calls` | `output[]` function_call | |
| `choices[0].message.content` | `output[]` message (output_text) | |
| `choices[0].message.refusal` | `output[]` message (refusal) | |
| `choices[0].message.annotations` | `output[]` message annotations | url_citation |
| `finish_reason` | `status` | stop/tool_calls→completed, length→incomplete, content_filter→failed |
| `usage.*` | `usage.*` | cached_tokens, reasoning_tokens mapped |

### Stream: text ✅ tool_calls ✅ reasoning ✅ annotations ✅ usage ✅

---

## 8. Responses → Messages (ResponsesToMessagesConverter)

### Request (Responses → Messages)

| Responses Field | Messages Field | Notes |
|---|---|---|
| `model` | `model` | |
| `instructions` | `system` | |
| `input` | `messages` | Converted from ResponseInputItem[] |
| `max_output_tokens` | `max_tokens` | Default 4096 |
| `temperature` | `temperature` | |
| `top_p` | `top_p` | |
| `tools` (function) | `tools` (input_schema) | |
| `tools` (web_search) | `tools` (web_search_20250305) | |
| `tool_choice` + `parallel_tool_calls` | `tool_choice` | auto/required/none → auto/any/none + disable_parallel_tool_use |
| `reasoning.effort` | `thinking` | Budget mapped |
| `text.format` (json_schema) | `output_config.format` | |
| `metadata.user_id` | `metadata.user_id` | |
| `service_tier` | `service_tier` | auto/standard_only only |
| `stream` | `stream` | |

**Not converted:** `previous_response_id`, `prompt_cache_key`, `prompt_cache_retention`, `safety_identifier`

### Response (Messages → Responses)

| Messages Field | Responses Field | Notes |
|---|---|---|
| `id` | `id` | |
| `model` | `model` | |
| `content[]` thinking | `output[]` reasoning | |
| `content[]` tool_use | `output[]` function_call | |
| `content[]` text | `output[]` message (output_text) | |
| `stop_reason` | `status` | end_turn→completed, max_tokens→incomplete, refusal→failed; **any tool_use/server_tool_use block in content → completed** (overrides stop_reason) |
| `usage.*` | `usage.*` | cache_read_input_tokens→cached_tokens |

### Stream: text ✅ tool_calls ✅ reasoning ✅ usage ✅

---

## 9. Responses → Gemini (ResponsesToGeminiConverter)

### Request (Responses → Gemini)

| Responses Field | Gemini Field | Notes |
|---|---|---|
| `model` | `model` | |
| `instructions` | `config.systemInstruction` | |
| `input` | `contents` | Converted from ResponseInputItem[] |
| `max_output_tokens` | `config.maxOutputTokens` | |
| `temperature` | `config.temperature` | |
| `top_p` | `config.topP` | |
| `tools` (function) | `config.tools[].functionDeclarations` | parametersJsonSchema |
| `tools` (web_search) | `config.tools[].googleSearch` | |
| `tool_choice` | `config.toolConfig.functionCallingConfig` | auto/required/none → AUTO/ANY/NONE |
| `reasoning.effort` | `config.thinkingConfig` | Budget mapped |
| `text.format` | `config.responseMimeType` + `config.responseJsonSchema` | |
| `include` (logprobs) | `config.responseLogprobs` + `config.logprobs` | |

**Not converted:** `metadata`, `previous_response_id`, `parallel_tool_calls`, `prompt_cache_key`, `prompt_cache_retention`, `safety_identifier`

### Response (Gemini → Responses)

| Gemini Field | Responses Field | Notes |
|---|---|---|
| `responseId` | `id` | |
| `modelVersion` | `model` | |
| `candidates[0].content.parts[]` thought | `output[]` reasoning | |
| `candidates[0].content.parts[]` functionCall | `output[]` function_call | |
| `candidates[0].content.parts[]` text | `output[]` message (output_text) | |
| `candidates[0].groundingMetadata` | `output[]` message annotations | url_citation |
| `candidates[0].finishReason` | `status` | STOP→completed, MAX_TOKENS→incomplete, SAFETY→failed; **any function_call → completed** (overrides finishReason) |
| `usageMetadata.*` | `usage.*` | toolUsePromptTokenCount+thoughtsTokenCount combined |

### Stream: text ✅ tool_calls ✅ reasoning ✅ grounding ✅ usage ✅

---

## 10. Gemini → CC (GeminiToChatCompletionConverter)

### Request (Gemini → CC)

| Gemini Field | CC Field | Notes |
|---|---|---|
| `model` | `model` | |
| `config.systemInstruction` | `messages[]` (system) | |
| `contents` | `messages` | user/model → user/assistant |
| `config.maxOutputTokens` | `max_completion_tokens` | |
| `config.temperature` | `temperature` | |
| `config.topP` | `top_p` | |
| `config.stopSequences` | `stop` | |
| `config.seed` | `seed` | |
| `config.frequencyPenalty` | `frequency_penalty` | |
| `config.presencePenalty` | `presence_penalty` | |
| `config.candidateCount` | `n` | |
| `config.responseLogprobs` | `logprobs` | |
| `config.logprobs` | `top_logprobs` | |
| `config.tools[].functionDeclarations` | `tools` (function) | |
| `config.tools[].googleSearch` | `web_search_options` | |
| `config.toolConfig` | `tool_choice` | AUTO/ANY/NONE → auto/required/none |
| `config.responseMimeType` + `responseJsonSchema` | `response_format` | |
| `config.thinkingConfig` | `reasoning_effort` | Budget→low/medium/high |

**Not converted:** `config.safetySettings`, `config.cachedContent`, `config.responseModalities`

### Response (CC → Gemini)

| CC Field | Gemini Field | Notes |
|---|---|---|
| `id` | `responseId` | |
| `model` | `modelVersion` | |
| `choices[0].message.reasoning` | `candidates[0].content.parts[]` thought | |
| `choices[0].message.tool_calls` | `candidates[0].content.parts[]` functionCall | |
| `choices[0].message.content` | `candidates[0].content.parts[]` text | |
| `finish_reason` | `finishReason` | stop/tool_calls→STOP, length→MAX_TOKENS, content_filter→SAFETY |
| `usage.*` | `usageMetadata.*` | cached_tokens→cachedContentTokenCount, reasoning_tokens→thoughtsTokenCount |

### Stream: text ✅ tool_calls ✅ reasoning ✅ usage ✅

---

## 11. Gemini → Messages (GeminiToMessagesConverter)

### Request (Gemini → Messages)

| Gemini Field | Messages Field | Notes |
|---|---|---|
| `model` | `model` | |
| `config.systemInstruction` | `system` | |
| `contents` | `messages` | user/model → user/assistant |
| `config.maxOutputTokens` | `max_tokens` | Default 4096 |
| `config.temperature` | `temperature` | |
| `config.topP` | `top_p` | |
| `config.topK` | `top_k` | |
| `config.stopSequences` | `stop_sequences` | |
| `config.tools[].functionDeclarations` | `tools` (input_schema) | |
| `config.tools[].googleSearch` | `tools` (web_search_20250305) | |
| `config.toolConfig` | `tool_choice` | AUTO/ANY/NONE → auto/any/none |
| `config.thinkingConfig` | `thinking` | Budget passthrough |
| `config.responseMimeType` + `responseJsonSchema` | `output_config` | |

**Not converted:** `config.seed`, `config.frequencyPenalty`, `config.presencePenalty`, `config.candidateCount`, `config.safetySettings`, `config.cachedContent`

### Response (Messages → Gemini)

| Messages Field | Gemini Field | Notes |
|---|---|---|
| `id` | `responseId` | |
| `model` | `modelVersion` | |
| `content[]` thinking | `candidates[0].content.parts[]` thought | With thoughtSignature |
| `content[]` tool_use | `candidates[0].content.parts[]` functionCall | |
| `content[]` text | `candidates[0].content.parts[]` text | |
| `stop_reason` | `finishReason` | end_turn→STOP, max_tokens→MAX_TOKENS, refusal→SAFETY |
| `usage.*` | `usageMetadata.*` | cache_read_input_tokens→cachedContentTokenCount |

### Stream: text ✅ tool_calls ✅ thinking ✅ usage ✅

---

## 12. Gemini → Responses (GeminiToResponsesConverter)

### Request (Gemini → Responses)

| Gemini Field | Responses Field | Notes |
|---|---|---|
| `model` | `model` | |
| `config.systemInstruction` | `instructions` | |
| `contents` | `input` | Converted to ResponseInputItem[] |
| `config.maxOutputTokens` | `max_output_tokens` | |
| `config.temperature` | `temperature` | |
| `config.topP` | `top_p` | |
| `config.tools[].functionDeclarations` | `tools` (function) | |
| `config.tools[].googleSearch` | `tools` (web_search_preview) | |
| `config.toolConfig` | `tool_choice` | AUTO/ANY/NONE → auto/required/none |
| `config.thinkingConfig` | `reasoning` | Budget→effort mapped |
| `config.responseMimeType` + `responseJsonSchema` | `text.format` | |

**Not converted:** `config.seed`, `config.frequencyPenalty`, `config.presencePenalty`, `config.candidateCount`, `config.safetySettings`, `config.cachedContent`

### Response (Responses → Gemini)

| Responses Field | Gemini Field | Notes |
|---|---|---|
| `id` | `responseId` | |
| `model` | `modelVersion` | |
| `output[]` reasoning | `candidates[0].content.parts[]` thought | |
| `output[]` function_call | `candidates[0].content.parts[]` functionCall | |
| `output[]` message (output_text) | `candidates[0].content.parts[]` text | |
| `status` | `finishReason` | completed→STOP, incomplete→MAX_TOKENS, failed→SAFETY |
| `usage.*` | `usageMetadata.*` | cached_tokens→cachedContentTokenCount, reasoning_tokens→thoughtsTokenCount |

### Stream: text ✅ tool_calls ✅ reasoning ✅ usage ✅

---

## Cross-Protocol Field Support Matrix

Which fields are supported when converting **from** a given protocol:

| Field Category | CC | Messages | Responses | Gemini |
|---|---|---|---|---|
| Model | ✅ all | ✅ all | ✅ all | ✅ all |
| System/Instructions | ✅ all | ✅ all | ✅ all | ✅ all |
| Temperature | ✅ all | ✅ all | ✅ all | ✅ all |
| Top P | ✅ all | ✅ all | ✅ all | ✅ all |
| Top K | ✗ | ✅ Gemini | ✗ | ✅ Msgs |
| Max Tokens | ✅ all | ✅ all | ✅ all | ✅ all |
| Stop Sequences | ✅ Msgs/Gemini | ✅ CC/Gemini | ✗ (no field) | ✅ CC/Msgs |
| Tools (function) | ✅ all | ✅ all | ✅ all | ✅ all |
| Tool Choice | ✅ all | ✅ all | ✅ all | ✅ all |
| Web Search | ✅ all | ✅ all | ✅ all | ✅ all |
| Reasoning/Thinking | ✅ all | ✅ all | ✅ all | ✅ all |
| JSON Response Format | ✅ all | ✅ all | ✅ all | ✅ all |
| Multimodal (images) | ✅ Msgs/Gemini | ✅ CC/Gemini | ✅ Gemini | ✅ CC/Msgs |
| Documents/Files | ✅ Msgs/Gemini | ✅ CC | ✗ | ✗ |
| Audio | ✅ Gemini | ✗ | ✗ | ✗ |
| Seed | ✅ Gemini | ✗ | ✗ | ✅ CC |
| Frequency Penalty | ✅ Gemini | ✗ | ✗ | ✅ CC |
| Presence Penalty | ✅ Gemini | ✗ | ✗ | ✅ CC |
| Candidate Count (n) | ✅ Gemini | ✗ | ✗ | ✅ CC |
| Logprobs | ✅ Gemini | ✗ | ✅ Gemini | ✅ CC/Resp |
| Parallel Tool Calls | ✅ Msgs/Resp | ✅ CC/Resp | ✅ CC | ✗ |
| Metadata/User | ✅ Msgs | ✅ CC/Resp | ✅ Msgs | ✗ |
| Service Tier | ✅ Msgs | ✅ CC | ✅ Msgs | ✗ |
| Cache Keys | ✗ | ✗ | ✅ CC | ✗ |
| Stream | ✅ all | ✅ all | ✅ all | ✅ all |
