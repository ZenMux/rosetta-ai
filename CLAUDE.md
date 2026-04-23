# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run build        # Build (tsc)
npm run typecheck    # Type check without emitting
npm run test         # Run all tests
npm run test:verbose # Run tests with verbose output
npm run lint         # Lint
npm run lint:fix     # Lint and auto-fix
npm run format       # Format with Prettier
```

Run a single test file:
```bash
npx jest src/chat-completions/__tests__/gemini.test.ts
```

Pre-commit hook runs `lint-staged` (eslint --fix + prettier --write on staged .ts files).

## Architecture

rosetta-ai converts between four AI provider protocols: **OpenAI Chat Completions (CC)**, **OpenAI Responses**, **Anthropic Messages**, and **Google Gemini**. All 12 pairwise converters are implemented.

### Gateway Pattern

Each converter handles a full round-trip for one gateway route:
- **convertRequest** — forward direction (user protocol → backend protocol)
- **convertResponse** — backward direction (backend → user)
- **convertStream** / **convertStreamChunk** / **convertStreamEvent** — backward streaming (backend stream → user stream events)

The request goes forward, but response and stream go backward. This matches real gateway usage where you present one API to users while calling a different backend.

### Directory Layout

Converters are organized by **source protocol**:
- `src/chat-completions/` — converters FROM Chat Completions (to Messages, Responses, Gemini)
- `src/messages/` — converters FROM Anthropic Messages
- `src/responses/` — converters FROM OpenAI Responses
- `src/gemini/` — converters FROM Google Gemini

Each directory has the converter files and a `__tests__/` subdirectory. Tests are co-located with their source.

### Streaming Differences

Anthropic and OpenAI use incremental delta-based streaming (each chunk has only new content). Gemini uses cumulative streaming (each chunk contains all content so far). Converters between these models must track previous state to extract or accumulate deltas.

Stream converters use a private `StreamState` object initialized in the constructor. Each converter instance handles one stream — create a new instance for each request.

### SDK Type Usage

Converters use SDK types directly (`OpenAI.ChatCompletionCreateParams`, `Anthropic.MessageCreateParams`, `GenerateContentParameters` from `@google/genai`). No custom type wrappers. Stream inputs/outputs are `AsyncIterable<T>` to decouple from SDK-specific stream classes.

### Key Patterns

- `convertTools()` — separates function tools from web search tools (each protocol represents web search differently)
- `convertToolChoice()` — maps between `auto/required/none` (CC/Responses), `auto/any/none` (Messages), `AUTO/ANY/NONE` (Gemini)
- Reasoning/thinking budget mapping: CC `reasoning_effort` (low/medium/high) ↔ Messages `thinking.budget_tokens` (2048/5120/10240) ↔ Gemini `thinkingConfig.thinkingBudget`
- `generateId()` — used for synthetic IDs when converting between protocols that use different ID schemes
- Usage token adjustments: Gemini's `toolUsePromptTokenCount` is added to `prompt_tokens`; `thoughtsTokenCount` is added to `completion_tokens`
