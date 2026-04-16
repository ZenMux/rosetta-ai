import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';

type AnthropicMessage = Anthropic.MessageParam;
type AnthropicContentBlock = Anthropic.ContentBlockParam;

const DEFAULT_MAX_TOKENS = 4096;

export function convertOpenAIRequestToAnthropic(
  params: OpenAI.ChatCompletionCreateParams,
): Anthropic.MessageCreateParams {
  const systemBlocks: Anthropic.TextBlockParam[] = [];
  const messages: AnthropicMessage[] = [];

  for (const msg of params.messages) {
    if (msg.role === 'system' || msg.role === 'developer') {
      systemBlocks.push(...extractSystemText(msg));
    } else if (msg.role === 'user') {
      messages.push({
        role: 'user',
        content: convertUserContent(msg.content),
      });
    } else if (msg.role === 'assistant') {
      messages.push({
        role: 'assistant',
        content: convertAssistantMessage(msg),
      });
    } else if (msg.role === 'tool') {
      appendToolResult(messages, msg);
    }
  }

  const result: Anthropic.MessageCreateParams = {
    model: params.model,
    max_tokens: params.max_tokens ?? params.max_completion_tokens ?? DEFAULT_MAX_TOKENS,
    messages,
  };

  if (systemBlocks.length > 0) {
    result.system = systemBlocks;
  }
  if (params.temperature != null) {
    result.temperature = params.temperature as number;
  }
  if (params.top_p != null) {
    result.top_p = params.top_p as number;
  }
  if (params.stop != null) {
    const stop = params.stop;
    result.stop_sequences = typeof stop === 'string' ? [stop] : stop as string[];
  }
  if (params.tools) {
    result.tools = convertTools(params.tools);
  }
  if (params.tool_choice !== undefined) {
    result.tool_choice = convertToolChoice(params.tool_choice);
  }
  if (params.stream === true) {
    (result as any).stream = true;
  }

  return result;
}

function extractSystemText(
  msg: OpenAI.ChatCompletionSystemMessageParam | OpenAI.ChatCompletionDeveloperMessageParam,
): Anthropic.TextBlockParam[] {
  if (typeof msg.content === 'string') {
    return [{ type: 'text', text: msg.content }];
  }
  return msg.content.map((part) => ({ type: 'text' as const, text: part.text }));
}

function convertUserContent(
  content: OpenAI.ChatCompletionUserMessageParam['content'],
): AnthropicContentBlock[] {
  if (typeof content === 'string') {
    return [{ type: 'text', text: content }];
  }
  return content.map(convertContentPart);
}

function convertContentPart(part: OpenAI.ChatCompletionContentPart): AnthropicContentBlock {
  switch (part.type) {
    case 'text':
      return { type: 'text', text: part.text };
    case 'image_url':
      return convertImageUrl(part);
    default:
      // input_audio, file, etc. - pass as text placeholder
      return { type: 'text', text: `[Unsupported content type: ${(part as any).type}]` };
  }
}

function convertImageUrl(
  part: OpenAI.ChatCompletionContentPartImage,
): Anthropic.ImageBlockParam {
  const url = part.image_url.url;
  const dataUriMatch = url.match(/^data:(image\/[a-z+]+);base64,(.+)$/);

  if (dataUriMatch) {
    return {
      type: 'image',
      source: {
        type: 'base64',
        media_type: dataUriMatch[1] as Anthropic.Base64ImageSource['media_type'],
        data: dataUriMatch[2],
      },
    };
  }

  return {
    type: 'image',
    source: { type: 'url', url },
  };
}

function convertAssistantMessage(
  msg: OpenAI.ChatCompletionAssistantMessageParam,
): AnthropicContentBlock[] {
  const blocks: AnthropicContentBlock[] = [];

  if (msg.content) {
    if (typeof msg.content === 'string') {
      blocks.push({ type: 'text', text: msg.content });
    } else {
      for (const part of msg.content) {
        if (part.type === 'text') {
          blocks.push({ type: 'text', text: part.text });
        }
      }
    }
  }

  if (msg.tool_calls) {
    for (const tc of msg.tool_calls) {
      if (tc.type === 'function') {
        blocks.push({
          type: 'tool_use',
          id: tc.id,
          name: tc.function.name,
          input: JSON.parse(tc.function.arguments || '{}'),
        });
      }
    }
  }

  return blocks;
}

function appendToolResult(
  messages: AnthropicMessage[],
  msg: OpenAI.ChatCompletionToolMessageParam,
): void {
  const toolResult: Anthropic.ToolResultBlockParam = {
    type: 'tool_result',
    tool_use_id: msg.tool_call_id,
    content: typeof msg.content === 'string'
      ? [{ type: 'text', text: msg.content }]
      : msg.content.map((p) => ({ type: 'text' as const, text: p.text })),
  };

  // Merge consecutive tool results into a single user message
  const last = messages[messages.length - 1];
  if (last && last.role === 'user' && Array.isArray(last.content)) {
    const lastContent = last.content as AnthropicContentBlock[];
    if (lastContent.length > 0 && lastContent[0].type === 'tool_result') {
      lastContent.push(toolResult);
      return;
    }
  }

  messages.push({ role: 'user', content: [toolResult] });
}

function convertTools(
  tools: OpenAI.ChatCompletionTool[],
): Anthropic.Tool[] {
  return tools
    .filter((t): t is OpenAI.ChatCompletionFunctionTool => t.type === 'function')
    .map((t) => ({
      name: t.function.name,
      description: t.function.description,
      input_schema: (t.function.parameters ?? { type: 'object' }) as Anthropic.Tool.InputSchema,
    }));
}

function convertToolChoice(
  choice: OpenAI.ChatCompletionToolChoiceOption,
): Anthropic.ToolChoice {
  if (choice === 'auto') return { type: 'auto' };
  if (choice === 'required') return { type: 'any' };
  if (choice === 'none') return { type: 'none' };

  if (typeof choice === 'object' && 'type' in choice) {
    if (choice.type === 'function' && 'function' in choice) {
      return { type: 'tool', name: (choice as OpenAI.ChatCompletionNamedToolChoice).function.name };
    }
  }

  return { type: 'auto' };
}
