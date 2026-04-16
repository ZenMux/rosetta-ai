import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';

type OpenAIMessage = OpenAI.ChatCompletionMessageParam;

export function convertAnthropicRequestToOpenAI(
  params: Anthropic.MessageCreateParams,
): OpenAI.ChatCompletionCreateParams {
  const messages: OpenAIMessage[] = [];

  // Convert system prompt to system message
  if (params.system) {
    if (typeof params.system === 'string') {
      messages.push({ role: 'system', content: params.system });
    } else {
      const text = params.system.map((b) => b.text).join('\n');
      messages.push({ role: 'system', content: text });
    }
  }

  // Convert messages
  for (const msg of params.messages) {
    if (msg.role === 'user') {
      convertUserMessage(messages, msg);
    } else if (msg.role === 'assistant') {
      convertAssistantMessage(messages, msg);
    }
  }

  const result: OpenAI.ChatCompletionCreateParams = {
    model: params.model,
    messages,
  };

  if (params.max_tokens !== undefined) {
    result.max_tokens = params.max_tokens;
  }
  if (params.temperature !== undefined) {
    result.temperature = params.temperature;
  }
  if (params.top_p !== undefined) {
    result.top_p = params.top_p;
  }
  if (params.stop_sequences !== undefined) {
    result.stop = params.stop_sequences;
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

function convertUserMessage(
  messages: OpenAIMessage[],
  msg: Anthropic.MessageParam,
): void {
  if (typeof msg.content === 'string') {
    messages.push({ role: 'user', content: msg.content });
    return;
  }

  const textParts: string[] = [];
  const contentParts: OpenAI.ChatCompletionContentPart[] = [];
  let hasNonText = false;
  const toolResults: Anthropic.ToolResultBlockParam[] = [];

  for (const block of msg.content) {
    switch (block.type) {
      case 'text':
        textParts.push(block.text);
        contentParts.push({ type: 'text', text: block.text });
        break;
      case 'image':
        hasNonText = true;
        contentParts.push(convertImageBlock(block as Anthropic.ImageBlockParam));
        break;
      case 'tool_result':
        toolResults.push(block as Anthropic.ToolResultBlockParam);
        break;
    }
  }

  // If there are tool_results, emit them as separate tool messages
  if (toolResults.length > 0) {
    for (const tr of toolResults) {
      const content = typeof tr.content === 'string'
        ? tr.content
        : tr.content
          ? tr.content
              .filter((b): b is Anthropic.TextBlockParam => b.type === 'text')
              .map((b) => b.text)
              .join('\n')
          : '';
      messages.push({
        role: 'tool',
        tool_call_id: tr.tool_use_id,
        content,
      });
    }
    return;
  }

  // If only text content, use simple string format
  if (!hasNonText && textParts.length === 1) {
    messages.push({ role: 'user', content: textParts[0] });
    return;
  }

  messages.push({ role: 'user', content: contentParts });
}

function convertImageBlock(
  block: Anthropic.ImageBlockParam,
): OpenAI.ChatCompletionContentPartImage {
  const source = block.source;

  if (source.type === 'base64') {
    return {
      type: 'image_url',
      image_url: {
        url: `data:${source.media_type};base64,${source.data}`,
      },
    };
  }

  // URL source
  return {
    type: 'image_url',
    image_url: { url: (source as Anthropic.URLImageSource).url },
  };
}

function convertAssistantMessage(
  messages: OpenAIMessage[],
  msg: Anthropic.MessageParam,
): void {
  if (typeof msg.content === 'string') {
    messages.push({ role: 'assistant', content: msg.content });
    return;
  }

  let textContent: string | undefined;
  const toolCalls: OpenAI.ChatCompletionMessageToolCall[] = [];

  for (const block of msg.content) {
    if (block.type === 'text') {
      textContent = textContent ? textContent + '\n' + block.text : block.text;
    } else if (block.type === 'tool_use') {
      const tu = block as Anthropic.ToolUseBlockParam;
      toolCalls.push({
        id: tu.id,
        type: 'function',
        function: {
          name: tu.name,
          arguments: JSON.stringify(tu.input),
        },
      });
    }
  }

  const assistantMsg: OpenAI.ChatCompletionAssistantMessageParam = {
    role: 'assistant',
  };

  if (textContent !== undefined) {
    assistantMsg.content = textContent;
  }
  if (toolCalls.length > 0) {
    assistantMsg.tool_calls = toolCalls;
  }

  messages.push(assistantMsg);
}

function convertTools(
  tools: Anthropic.ToolUnion[],
): OpenAI.ChatCompletionTool[] {
  return tools
    .filter((t): t is Anthropic.Tool => 'input_schema' in t)
    .map((t) => ({
      type: 'function' as const,
      function: {
        name: t.name,
        description: t.description,
        parameters: t.input_schema as unknown as Record<string, unknown>,
      },
    }));
}

function convertToolChoice(
  choice: Anthropic.ToolChoice,
): OpenAI.ChatCompletionToolChoiceOption {
  switch (choice.type) {
    case 'auto':
      return 'auto';
    case 'any':
      return 'required';
    case 'none':
      return 'none';
    case 'tool':
      return {
        type: 'function',
        function: { name: (choice as Anthropic.ToolChoiceTool).name },
      };
    default:
      return 'auto';
  }
}
