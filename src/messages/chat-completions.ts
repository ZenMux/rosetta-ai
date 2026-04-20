import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';

type OpenAIMessage = OpenAI.ChatCompletionMessageParam;

interface StreamState {
  id: string;
  model: string;
  toolCallCounter: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreationInputTokens: number;
  ephemeral5mTokens: number;
  ephemeral1hTokens: number;
  webSearchRequests: number;
  annotations: OpenAI.ChatCompletionMessage.Annotation[];
}

export class MessagesToChatCompletionConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion ---

  convertRequest(params: Anthropic.MessageCreateParams): OpenAI.ChatCompletionCreateParams {
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
      const { tools, webSearchOptions } = this.convertTools(params.tools);
      if (tools.length > 0) {
        result.tools = tools;
      }
      if (webSearchOptions) {
        (result as any).web_search_options = webSearchOptions;
      }
    }
    if (params.tool_choice !== undefined) {
      const { toolChoice, parallelToolCalls } = this.convertToolChoice(params.tool_choice);
      result.tool_choice = toolChoice;
      if (parallelToolCalls !== undefined) {
        (result as any).parallel_tool_calls = parallelToolCalls;
      }
    }
    if (params.output_config) {
      result.response_format = this.convertOutputConfig(params.output_config);
    }
    if (params.thinking) {
      (result as any).reasoning_effort = this.convertThinking(params.thinking);
    }
    if (params.metadata?.user_id) {
      result.user = params.metadata.user_id;
    }
    if (params.service_tier != null) {
      (result as any).service_tier =
        params.service_tier === 'standard_only' ? 'default' : params.service_tier;
    }
    if (params.stream === true) {
      (result as any).stream = true;
    }

    return result;
  }

  // --- Response conversion ---

  convertResponse(message: Anthropic.Message): OpenAI.ChatCompletion {
    const textParts: string[] = [];
    const thinkingParts: string[] = [];
    const reasoningDetails: Array<Record<string, unknown>> = [];
    const toolCalls: OpenAI.ChatCompletionMessageToolCall[] = [];
    const annotations: OpenAI.ChatCompletionMessage.Annotation[] = [];

    for (const block of message.content) {
      if (block.type === 'text') {
        textParts.push(block.text);
      } else if (block.type === 'thinking') {
        thinkingParts.push(block.thinking);
        reasoningDetails.push({
          type: 'reasoning.text',
          text: block.thinking,
          signature: block.signature,
          format: 'anthropic-claude-v1',
          index: 0,
        });
      } else if (block.type === 'tool_use' || block.type === 'server_tool_use') {
        toolCalls.push({
          id: block.id,
          type: 'function',
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input),
          },
        });
      } else if (block.type === 'web_search_tool_result') {
        const content = block.content;
        if (Array.isArray(content)) {
          for (const result of content) {
            annotations.push({
              type: 'url_citation',
              url_citation: {
                title: result.title,
                url: result.url,
                start_index: 0,
                end_index: 0,
              },
            });
          }
        }
      }
    }

    const assistantMessage: OpenAI.ChatCompletionMessage = {
      role: 'assistant',
      content: textParts.length > 0 ? textParts.join('') : null,
      refusal: null,
    };

    if (thinkingParts.length > 0) {
      Object.assign(assistantMessage, {
        reasoning: thinkingParts.join(''),
        reasoning_details: reasoningDetails,
      });
    }
    if (toolCalls.length > 0) {
      assistantMessage.tool_calls = toolCalls;
    }
    if (annotations.length > 0) {
      assistantMessage.annotations = annotations;
    }

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
      usage: this.convertUsage(message.usage),
    };
  }

  private convertUsage(usage: Anthropic.Usage): OpenAI.CompletionUsage {
    const cacheRead = usage.cache_read_input_tokens ?? 0;
    const ephemeral5m = usage.cache_creation?.ephemeral_5m_input_tokens ?? 0;
    const ephemeral1h = usage.cache_creation?.ephemeral_1h_input_tokens ?? 0;
    const webSearch = usage.server_tool_use?.web_search_requests ?? 0;
    const cacheCreation = usage.cache_creation_input_tokens ?? 0;

    return {
      prompt_tokens: usage.input_tokens,
      completion_tokens: usage.output_tokens,
      total_tokens: usage.input_tokens + usage.output_tokens,
      prompt_tokens_details: {
        cached_tokens: cacheRead,
        ...({
          ephemeral_5m_input_tokens: ephemeral5m,
          ephemeral_1h_input_tokens: ephemeral1h,
          web_search: webSearch,
          cache_creation_input_tokens: cacheCreation,
        } as any),
      },
    };
  }

  // --- Stream conversion ---

  convertStream(event: Anthropic.RawMessageStreamEvent): OpenAI.ChatCompletionChunk | null {
    switch (event.type) {
      case 'message_start':
        return this.handleMessageStart(event);
      case 'content_block_start':
        return this.handleContentBlockStart(event);
      case 'content_block_delta':
        return this.handleContentBlockDelta(event);
      case 'content_block_stop':
        return this.handleContentBlockStop(event);
      case 'message_delta':
        return this.handleMessageDelta(event);
      case 'message_stop':
        return this.handleMessageStop();
      default:
        return null;
    }
  }

  // --- Private helpers ---

  private createStreamState(): StreamState {
    return {
      id: '',
      model: '',
      toolCallCounter: -1,
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      cacheCreationInputTokens: 0,
      ephemeral5mTokens: 0,
      ephemeral1hTokens: 0,
      webSearchRequests: 0,
      annotations: [],
    };
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

  private handleMessageStart(event: Anthropic.RawMessageStartEvent): OpenAI.ChatCompletionChunk {
    const state = this.streamState;
    state.id = event.message.id;
    state.model = event.message.model;

    const usage = event.message.usage;
    state.inputTokens = usage.input_tokens;
    state.cacheReadTokens = usage.cache_read_input_tokens ?? 0;
    state.cacheCreationInputTokens = usage.cache_creation_input_tokens ?? 0;
    state.ephemeral5mTokens = usage.cache_creation?.ephemeral_5m_input_tokens ?? 0;
    state.ephemeral1hTokens = usage.cache_creation?.ephemeral_1h_input_tokens ?? 0;

    return this.makeChunk({ role: 'assistant' });
  }

  private handleContentBlockStart(
    event: Anthropic.RawContentBlockStartEvent,
  ): OpenAI.ChatCompletionChunk | null {
    const block = event.content_block;

    if (block.type === 'text') {
      if (block.text && block.text.trim()) {
        return this.makeChunk({ content: block.text });
      }
      return null;
    }

    if (block.type === 'tool_use' || block.type === 'server_tool_use') {
      this.streamState.toolCallCounter++;
      const index = this.streamState.toolCallCounter;
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

    if (block.type === 'web_search_tool_result') {
      const content = block.content;
      if (Array.isArray(content)) {
        for (const result of content) {
          this.streamState.annotations.push({
            type: 'url_citation',
            url_citation: {
              title: result.title,
              url: result.url,
              start_index: 0,
              end_index: 0,
            },
          });
        }
      }
      return null;
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
      const toolIndex = this.streamState.toolCallCounter;
      return this.makeChunk({
        content: '',
        tool_calls: [
          { index: toolIndex, type: 'function', function: { arguments: delta.partial_json } },
        ],
      });
    }

    if (delta.type === 'thinking_delta') {
      return this.makeChunk({
        content: '',
        ...({
          reasoning: delta.thinking,
          reasoning_details: [
            {
              type: 'reasoning.text',
              text: delta.thinking,
              format: 'anthropic-claude-v1',
              index: 0,
            },
          ],
        } as any),
      });
    }

    if (delta.type === 'signature_delta') {
      return this.makeChunk({
        content: '',
        ...({
          reasoning: null,
          reasoning_details: [
            {
              type: 'reasoning.text',
              signature: delta.signature,
              format: 'anthropic-claude-v1',
              index: 0,
            },
          ],
        } as any),
      });
    }

    return null;
  }

  private handleContentBlockStop(
    _event: Anthropic.RawContentBlockStopEvent,
  ): OpenAI.ChatCompletionChunk {
    return this.makeChunk({ content: '' });
  }

  private handleMessageDelta(event: Anthropic.RawMessageDeltaEvent): OpenAI.ChatCompletionChunk {
    const state = this.streamState;
    const usage = event.usage;
    state.outputTokens = usage.output_tokens;
    if (usage.input_tokens != null) {
      state.inputTokens = usage.input_tokens;
    }
    state.webSearchRequests = usage.server_tool_use?.web_search_requests ?? state.webSearchRequests;

    const delta: OpenAI.ChatCompletionChunk.Choice.Delta = { content: '' };
    if (state.annotations.length > 0) {
      (delta as any).annotations = state.annotations;
    }

    return this.makeChunk(delta, this.mapStopReasonToFinishReason(event.delta.stop_reason));
  }

  private handleMessageStop(): OpenAI.ChatCompletionChunk {
    const state = this.streamState;
    const usage: OpenAI.CompletionUsage = {
      prompt_tokens: state.inputTokens,
      completion_tokens: state.outputTokens,
      total_tokens: state.inputTokens + state.outputTokens,
      prompt_tokens_details: {
        cached_tokens: state.cacheReadTokens,
        ...({
          ephemeral_5m_input_tokens: state.ephemeral5mTokens,
          ephemeral_1h_input_tokens: state.ephemeral1hTokens,
          web_search: state.webSearchRequests,
          cache_creation_input_tokens: state.cacheCreationInputTokens,
        } as any),
      },
      completion_tokens_details: {
        reasoning_tokens: 0,
      },
    };

    return {
      id: state.id,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: state.model,
      choices: [],
      usage,
    };
  }

  private convertUserMessage(messages: OpenAIMessage[], msg: Anthropic.MessageParam): void {
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
        case 'document':
          hasNonText = true;
          contentParts.push(this.convertDocumentBlock(block as Anthropic.DocumentBlockParam));
          break;
        case 'tool_result':
          toolResults.push(block as Anthropic.ToolResultBlockParam);
          break;
      }
    }

    if (toolResults.length > 0) {
      for (const tr of toolResults) {
        const content =
          typeof tr.content === 'string'
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

  private convertDocumentBlock(
    block: Anthropic.DocumentBlockParam,
  ): OpenAI.ChatCompletionContentPart {
    const source = block.source;

    if (source.type === 'base64') {
      const src = source as Anthropic.Base64PDFSource;
      return {
        type: 'file',
        file: {
          file_data: `data:${src.media_type};base64,${src.data}`,
          filename: block.title ?? undefined,
        },
      };
    }

    if (source.type === 'url') {
      const src = source as Anthropic.URLPDFSource;
      return {
        type: 'file',
        file: {
          file_data: src.url,
          filename: block.title ?? undefined,
        },
      };
    }

    // plain text source — encode as text content
    if (source.type === 'text') {
      const src = source as Anthropic.PlainTextSource;
      return { type: 'text', text: src.data };
    }

    return { type: 'text', text: '[Unsupported document source]' };
  }

  private convertAssistantMessage(messages: OpenAIMessage[], msg: Anthropic.MessageParam): void {
    if (typeof msg.content === 'string') {
      messages.push({ role: 'assistant', content: msg.content });
      return;
    }

    let textContent: string | undefined;
    const toolCalls: OpenAI.ChatCompletionMessageToolCall[] = [];

    let thinkingContent: string | undefined;
    const reasoningDetails: Array<Record<string, unknown>> = [];

    for (const block of msg.content) {
      if (block.type === 'text') {
        textContent = textContent ? textContent + block.text : block.text;
      } else if (block.type === 'thinking') {
        const tb = block as Anthropic.ThinkingBlockParam;
        thinkingContent = thinkingContent ? thinkingContent + tb.thinking : tb.thinking;
        reasoningDetails.push({
          type: 'reasoning.text',
          text: tb.thinking,
          signature: tb.signature,
          format: 'anthropic-claude-v1',
          index: 0,
        });
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

    if (thinkingContent !== undefined) {
      Object.assign(assistantMsg, {
        reasoning: thinkingContent,
        reasoning_details: reasoningDetails,
      });
    }
    if (textContent !== undefined) {
      assistantMsg.content = textContent;
    }
    if (toolCalls.length > 0) {
      assistantMsg.tool_calls = toolCalls;
    }

    messages.push(assistantMsg);
  }

  private convertTools(tools: Anthropic.ToolUnion[]): {
    tools: OpenAI.ChatCompletionTool[];
    webSearchOptions?: Record<string, unknown>;
  } {
    const functionTools: OpenAI.ChatCompletionTool[] = [];
    let webSearchOptions: Record<string, unknown> | undefined;

    for (const t of tools) {
      if ('input_schema' in t) {
        functionTools.push({
          type: 'function',
          function: {
            name: (t as Anthropic.Tool).name,
            description: (t as Anthropic.Tool).description,
            parameters: (t as Anthropic.Tool).input_schema as unknown as Record<string, unknown>,
          },
        });
      } else if (
        'type' in t &&
        (t.type === 'web_search_20250305' || t.type === 'web_search_20260209')
      ) {
        const ws = t as Anthropic.WebSearchTool20250305;
        webSearchOptions = {};
        if (ws.max_uses != null) {
          webSearchOptions['max_uses'] = ws.max_uses;
        }
        if (ws.user_location != null) {
          webSearchOptions['user_location'] = ws.user_location;
        }
      }
    }

    return { tools: functionTools, webSearchOptions };
  }

  private convertToolChoice(choice: Anthropic.ToolChoice): {
    toolChoice: OpenAI.ChatCompletionToolChoiceOption;
    parallelToolCalls?: boolean;
  } {
    const parallelToolCalls =
      'disable_parallel_tool_use' in choice && choice.disable_parallel_tool_use === true
        ? false
        : undefined;

    switch (choice.type) {
      case 'auto':
        return { toolChoice: 'auto', parallelToolCalls };
      case 'any':
        return { toolChoice: 'required', parallelToolCalls };
      case 'none':
        return { toolChoice: 'none' };
      case 'tool':
        return {
          toolChoice: {
            type: 'function',
            function: { name: (choice as Anthropic.ToolChoiceTool).name },
          },
          parallelToolCalls,
        };
      default:
        return { toolChoice: 'auto' };
    }
  }

  private convertOutputConfig(
    config: Anthropic.OutputConfig,
  ): OpenAI.ChatCompletionCreateParams['response_format'] {
    if (config.format?.type === 'json_schema') {
      return {
        type: 'json_schema',
        json_schema: {
          name: 'response',
          schema: config.format.schema,
          strict: true,
        },
      };
    }
    return { type: 'text' };
  }

  private convertThinking(thinking: Anthropic.ThinkingConfigParam): string | null {
    if (thinking.type === 'disabled') return 'none';
    if (thinking.type === 'enabled') {
      const budget = (thinking as Anthropic.ThinkingConfigEnabled).budget_tokens;
      if (budget <= 2048) return 'low';
      if (budget <= 5120) return 'medium';
      if (budget <= 10240) return 'high';
      return 'high';
    }
    // adaptive
    return 'medium';
  }

  private mapStopReasonToFinishReason(
    reason: Anthropic.Message['stop_reason'] | Anthropic.StopReason | null | string,
  ): OpenAI.ChatCompletion.Choice['finish_reason'] {
    switch (reason) {
      case 'end_turn':
      case 'stop_sequence':
      case 'pause_turn':
        return 'stop';
      case 'tool_use':
        return 'tool_calls';
      case 'max_tokens':
      case 'model_context_window_exceeded':
        return 'length';
      case 'refusal':
        return 'content_filter';
      default:
        return 'stop';
    }
  }
}
