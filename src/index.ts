// rosetta-ai: Universal translator between AI provider protocols

// OpenAI Chat Completion → Anthropic Messages
export { convertOpenAIRequestToAnthropic } from './converters/openai-to-anthropic/request';
export { convertOpenAIResponseToAnthropic } from './converters/openai-to-anthropic/response';
export { createOpenAIToAnthropicStreamConverter } from './converters/openai-to-anthropic/stream';

// Anthropic Messages → OpenAI Chat Completion
export { convertAnthropicRequestToOpenAI } from './converters/anthropic-to-openai/request';
export { convertAnthropicResponseToOpenAI } from './converters/anthropic-to-openai/response';
export { createAnthropicToOpenAIStreamConverter } from './converters/anthropic-to-openai/stream';
