import type OpenAI from 'openai';
import type Anthropic from '@anthropic-ai/sdk';

type AnthropicMessage = Anthropic.MessageParam;
type AnthropicContentBlock = Anthropic.ContentBlockParam;

const DEFAULT_MAX_TOKENS = 4096;

interface StreamState {
  messageStarted: boolean;
  currentBlockIndex: number;
  currentBlockType: 'text' | 'tool_use' | null;
  id: string;
  model: string;
  toolCallIndexMap: Map<number, number>;
}

export class ChatCompletionToMessagesConverter {
  private streamState: StreamState;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion ---

  convertRequest(
    params: OpenAI.ChatCompletionCreateParams,
  ): Anthropic.MessageCreateParams {
    const systemBlocks: Anthropic.TextBlockParam[] = [];
    const messages: AnthropicMessage[] = [];

    for (const msg of params.messages) {
      if (msg.role === 'system' || msg.role === 'developer') {
        systemBlocks.push(...this.extractSystemText(msg));
      } else if (msg.role === 'user') {
        messages.push({
          role: 'user',
          content: this.convertUserContent(msg.content),
        });
      } else if (msg.role === 'assistant') {
        messages.push({
          role: 'assistant',
          content: this.convertAssistantMessage(msg),
        });
      } else if (msg.role === 'tool') {
        this.appendToolResult(messages, msg);
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
      result.tools = this.convertTools(params.tools);
    }
    if (params.tool_choice !== undefined) {
      result.tool_choice = this.convertToolChoice(
        params.tool_choice,
        (params as any).parallel_tool_calls,
      );
    }
    if (params.response_format) {
      result.output_config = this.convertResponseFormat(params.response_format);
    }
    if ((params as any).reasoning_effort != null) {
      result.thinking = this.convertReasoningEffort((params as any).reasoning_effort);
    }
    if (params.user) {
      result.metadata = { user_id: params.user };
    }
    if ((params as any).service_tier != null) {
      const tier = (params as any).service_tier;
      if (tier === 'auto' || tier === 'standard_only') {
        result.service_tier = tier;
      }
    }
    if (params.stream === true) {
      (result as any).stream = true;
    }

    return result;
  }

  // --- Response conversion ---

  convertResponse(
    response: OpenAI.ChatCompletion,
  ): Anthropic.Message {
    const choice = response.choices[0];
    const msg = choice?.message;

    const content: Anthropic.ContentBlock[] = [];

    if (msg?.content) {
      content.push({ type: 'text', text: msg.content, citations: null });
    }

    if (msg?.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (tc.type === 'function') {
          content.push({
            type: 'tool_use',
            id: tc.id,
            name: tc.function.name,
            input: JSON.parse(tc.function.arguments || '{}'),
            caller: { type: 'direct' },
          });
        }
      }
    }

    if (content.length === 0) {
      content.push({ type: 'text', text: '', citations: null });
    }

    return {
      id: response.id,
      type: 'message',
      role: 'assistant',
      model: response.model as Anthropic.Model,
      content,
      stop_reason: this.mapFinishReasonToStopReason(choice?.finish_reason),
      stop_sequence: null,
      usage: {
        input_tokens: response.usage?.prompt_tokens ?? 0,
        output_tokens: response.usage?.completion_tokens ?? 0,
        cache_creation_input_tokens: null,
        cache_read_input_tokens: null,
        cache_creation: null,
        inference_geo: null,
        server_tool_use: null,
        service_tier: null,
      },
      container: null,
    };
  }

  // --- Stream conversion ---

  convertStream(
    chunk: OpenAI.ChatCompletionChunk,
  ): Anthropic.RawMessageStreamEvent[] {
    const state = this.streamState;
    const events: Anthropic.RawMessageStreamEvent[] = [];
    const choice = chunk.choices[0];
    if (!choice) return events;

    state.id = chunk.id;
    state.model = chunk.model;

    const delta = choice.delta;

    // First chunk with role - emit message_start
    if (!state.messageStarted && delta.role === 'assistant') {
      state.messageStarted = true;
      events.push({
        type: 'message_start',
        message: {
          id: chunk.id,
          type: 'message',
          role: 'assistant',
          model: chunk.model as Anthropic.Model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            cache_creation: null,
            inference_geo: null,
            server_tool_use: null,
            service_tier: null,
          },
          container: null,
        },
      });

      state.currentBlockIndex = 0;
      state.currentBlockType = 'text';
      events.push({
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '', citations: null },
      });
    }

    // Text content delta
    if (delta.content) {
      events.push({
        type: 'content_block_delta',
        index: state.currentBlockIndex,
        delta: { type: 'text_delta', text: delta.content },
      });
    }

    // Tool call deltas
    if (delta.tool_calls) {
      for (const tc of delta.tool_calls) {
        if (tc.id && tc.function?.name) {
          if (state.currentBlockType !== null) {
            events.push({
              type: 'content_block_stop',
              index: state.currentBlockIndex,
            });
          }

          state.currentBlockIndex++;
          state.currentBlockType = 'tool_use';
          state.toolCallIndexMap.set(tc.index, state.currentBlockIndex);

          events.push({
            type: 'content_block_start',
            index: state.currentBlockIndex,
            content_block: {
              type: 'tool_use',
              id: tc.id,
              name: tc.function.name,
              input: {},
              caller: { type: 'direct' },
            },
          });

          if (tc.function.arguments) {
            events.push({
              type: 'content_block_delta',
              index: state.currentBlockIndex,
              delta: {
                type: 'input_json_delta',
                partial_json: tc.function.arguments,
              },
            });
          }
        } else if (tc.function?.arguments) {
          const blockIndex = state.toolCallIndexMap.get(tc.index) ?? state.currentBlockIndex;
          events.push({
            type: 'content_block_delta',
            index: blockIndex,
            delta: {
              type: 'input_json_delta',
              partial_json: tc.function.arguments,
            },
          });
        }
      }
    }

    // Finish reason - emit stop events
    if (choice.finish_reason) {
      if (state.currentBlockType !== null) {
        events.push({
          type: 'content_block_stop',
          index: state.currentBlockIndex,
        });
      }

      events.push({
        type: 'message_delta',
        delta: {
          stop_reason: this.mapFinishReasonToStopReason(choice.finish_reason),
          stop_sequence: null,
          container: null,
        },
        usage: {
          output_tokens: 0,
          input_tokens: null,
          cache_creation_input_tokens: null,
          cache_read_input_tokens: null,
          server_tool_use: null,
        },
      });

      events.push({ type: 'message_stop' });
    }

    return events;
  }

  // --- Private helpers ---

  private createStreamState(): StreamState {
    return {
      messageStarted: false,
      currentBlockIndex: -1,
      currentBlockType: null,
      id: '',
      model: '',
      toolCallIndexMap: new Map(),
    };
  }

  private extractSystemText(
    msg: OpenAI.ChatCompletionSystemMessageParam | OpenAI.ChatCompletionDeveloperMessageParam,
  ): Anthropic.TextBlockParam[] {
    if (typeof msg.content === 'string') {
      return [{ type: 'text', text: msg.content }];
    }
    return msg.content.map((part) => ({ type: 'text' as const, text: part.text }));
  }

  private convertUserContent(
    content: OpenAI.ChatCompletionUserMessageParam['content'],
  ): AnthropicContentBlock[] {
    if (typeof content === 'string') {
      return [{ type: 'text', text: content }];
    }
    return content.map((part) => this.convertContentPart(part));
  }

  private convertContentPart(part: OpenAI.ChatCompletionContentPart): AnthropicContentBlock {
    switch (part.type) {
      case 'text':
        return { type: 'text', text: part.text };
      case 'image_url':
        return this.convertImageUrl(part);
      default:
        return { type: 'text', text: `[Unsupported content type: ${(part as any).type}]` };
    }
  }

  private convertImageUrl(
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

  private convertAssistantMessage(
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

  private appendToolResult(
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

  private convertTools(
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

  private convertToolChoice(
    choice: OpenAI.ChatCompletionToolChoiceOption,
    parallelToolCalls?: boolean,
  ): Anthropic.ToolChoice {
    const disableParallel = parallelToolCalls === false ? true : undefined;

    if (choice === 'auto') {
      return disableParallel ? { type: 'auto', disable_parallel_tool_use: true } : { type: 'auto' };
    }
    if (choice === 'required') {
      return disableParallel ? { type: 'any', disable_parallel_tool_use: true } : { type: 'any' };
    }
    if (choice === 'none') return { type: 'none' };

    if (typeof choice === 'object' && 'type' in choice) {
      if (choice.type === 'function' && 'function' in choice) {
        const name = (choice as OpenAI.ChatCompletionNamedToolChoice).function.name;
        return disableParallel
          ? { type: 'tool', name, disable_parallel_tool_use: true }
          : { type: 'tool', name };
      }
    }

    return { type: 'auto' };
  }

  private convertResponseFormat(
    format: NonNullable<OpenAI.ChatCompletionCreateParams['response_format']>,
  ): Anthropic.OutputConfig {
    if ('json_schema' in format && format.type === 'json_schema') {
      return {
        format: {
          type: 'json_schema',
          schema: format.json_schema.schema ?? {},
        },
      };
    }
    if (format.type === 'json_object') {
      return {
        format: {
          type: 'json_schema',
          schema: { type: 'object' },
        },
      };
    }
    // text format — no output_config needed, return empty
    return {};
  }

  private convertReasoningEffort(
    effort: string | null,
  ): Anthropic.ThinkingConfigParam {
    if (!effort || effort === 'none' || effort === 'minimal') {
      return { type: 'disabled' };
    }
    // Map OpenAI effort levels to Anthropic budget_tokens
    const budgetMap: Record<string, number> = {
      low: 2048,
      medium: 5120,
      high: 10240,
      xhigh: 20480,
    };
    return {
      type: 'enabled',
      budget_tokens: budgetMap[effort] ?? 10240,
    };
  }

  private mapFinishReasonToStopReason(
    reason: OpenAI.ChatCompletion.Choice['finish_reason'] | string | undefined,
  ): Anthropic.Message['stop_reason'] {
    switch (reason) {
      case 'stop':
        return 'end_turn';
      case 'tool_calls':
        return 'tool_use';
      case 'length':
        return 'max_tokens';
      case 'content_filter':
      case 'function_call':
      default:
        return 'end_turn';
    }
  }
}
