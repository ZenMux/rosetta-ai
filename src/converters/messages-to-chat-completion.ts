import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';

type OpenAIMessage = OpenAI.ChatCompletionMessageParam;

interface StreamState {
  id: string;
  model: string;
  toolCallCounter: number;
}

export class MessagesToChatCompletionConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion ---

  convertRequest(
    params: Anthropic.MessageCreateParams,
  ): OpenAI.ChatCompletionCreateParams {
    const messages: OpenAIMessage[] = [];

    if (params.system) {
      if (typeof params.system === 'string') {
        messages.push({ role: 'system', content: params.system });
      } else {
        const text = params.system.map((b) => b.text).join('\n');
        messages.push({ role: 'system', content: text });
      }
    }

    for (const msg of params.messages) {
      if (msg.role === 'user') {
        this.convertUserMessage(messages, msg);
      } else if (msg.role === 'assistant') {
        this.convertAssistantMessage(messages, msg);
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
      result.tools = this.convertTools(params.tools);
    }
    if (params.tool_choice !== undefined) {
      result.tool_choice = this.convertToolChoice(params.tool_choice);
    }
    if (params.stream === true) {
      (result as any).stream = true;
    }

    return result;
  }

  // --- Response conversion ---

  convertResponse(
    message: Anthropic.Message,
  ): OpenAI.ChatCompletion {
    const textParts: string[] = [];
    const toolCalls: OpenAI.ChatCompletionMessageToolCall[] = [];

    for (const block of message.content) {
      if (block.type === 'text') {
        textParts.push(block.text);
      } else if (block.type === 'tool_use') {
        toolCalls.push({
          id: block.id,
          type: 'function',
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input),
          },
        });
      }
    }

    const assistantMessage: OpenAI.ChatCompletionMessage = {
      role: 'assistant',
      content: textParts.length > 0 ? textParts.join('\n') : null,
      refusal: null,
    };

    if (toolCalls.length > 0) {
      assistantMessage.tool_calls = toolCalls;
    }

    const promptTokens = message.usage.input_tokens;
    const completionTokens = message.usage.output_tokens;

    return {
      id: message.id,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: message.model,
      choices: [
        {
          index: 0,
          message: assistantMessage,
          finish_reason: this.mapStopReasonToFinishReason(message.stop_reason),
          logprobs: null,
        },
      ],
      usage: {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: promptTokens + completionTokens,
      },
    };
  }

  // --- Stream conversion ---

  convertStream(
    event: Anthropic.RawMessageStreamEvent,
  ): OpenAI.ChatCompletionChunk | null {
    switch (event.type) {
      case 'message_start':
        return this.handleMessageStart(event);
      case 'content_block_start':
        return this.handleContentBlockStart(event);
      case 'content_block_delta':
        return this.handleContentBlockDelta(event);
      case 'content_block_stop':
        return null;
      case 'message_delta':
        return this.handleMessageDelta(event);
      case 'message_stop':
        return null;
      default:
        return null;
    }
  }

  resetStreamState(): void {
    this.streamState = this.createStreamState();
  }

  // --- Private helpers ---

  private createStreamState(): StreamState {
    return { id: '', model: '', toolCallCounter: 0 };
  }

  private makeChunk(
    delta: OpenAI.ChatCompletionChunk.Choice.Delta,
    finish_reason: OpenAI.ChatCompletionChunk.Choice['finish_reason'] = null,
  ): OpenAI.ChatCompletionChunk {
    return {
      id: this.streamState.id,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: this.streamState.model,
      choices: [{ index: 0, delta, finish_reason }],
    };
  }

  private handleMessageStart(
    event: Anthropic.RawMessageStartEvent,
  ): OpenAI.ChatCompletionChunk {
    this.streamState.id = event.message.id;
    this.streamState.model = event.message.model;
    return this.makeChunk({ role: 'assistant' });
  }

  private handleContentBlockStart(
    event: Anthropic.RawContentBlockStartEvent,
  ): OpenAI.ChatCompletionChunk | null {
    const block = event.content_block;

    if (block.type === 'tool_use') {
      const index = this.streamState.toolCallCounter++;
      return this.makeChunk({
        tool_calls: [
          {
            index,
            id: block.id,
            type: 'function',
            function: { name: block.name, arguments: '' },
          },
        ],
      });
    }

    return null;
  }

  private handleContentBlockDelta(
    event: Anthropic.RawContentBlockDeltaEvent,
  ): OpenAI.ChatCompletionChunk | null {
    const delta = event.delta;

    if (delta.type === 'text_delta') {
      return this.makeChunk({ content: delta.text });
    }

    if (delta.type === 'input_json_delta') {
      const toolIndex = this.streamState.toolCallCounter - 1;
      return this.makeChunk({
        tool_calls: [
          { index: toolIndex, function: { arguments: delta.partial_json } },
        ],
      });
    }

    return null;
  }

  private handleMessageDelta(
    event: Anthropic.RawMessageDeltaEvent,
  ): OpenAI.ChatCompletionChunk {
    return this.makeChunk(
      {},
      this.mapStopReasonToFinishReason(event.delta.stop_reason),
    );
  }

  private convertUserMessage(
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
          contentParts.push(this.convertImageBlock(block as Anthropic.ImageBlockParam));
          break;
        case 'tool_result':
          toolResults.push(block as Anthropic.ToolResultBlockParam);
          break;
      }
    }

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

    if (!hasNonText && textParts.length === 1) {
      messages.push({ role: 'user', content: textParts[0] });
      return;
    }

    messages.push({ role: 'user', content: contentParts });
  }

  private convertImageBlock(
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

    return {
      type: 'image_url',
      image_url: { url: (source as Anthropic.URLImageSource).url },
    };
  }

  private convertAssistantMessage(
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

  private convertTools(
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

  private convertToolChoice(
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

  private mapStopReasonToFinishReason(
    reason: Anthropic.Message['stop_reason'] | Anthropic.StopReason | null,
  ): OpenAI.ChatCompletion.Choice['finish_reason'] {
    switch (reason) {
      case 'end_turn':
      case 'stop_sequence':
        return 'stop';
      case 'tool_use':
        return 'tool_calls';
      case 'max_tokens':
        return 'length';
      default:
        return 'stop';
    }
  }
}
