import type OpenAI from "openai";
import type Anthropic from "@anthropic-ai/sdk";

type AnthropicMessage = Anthropic.MessageParam;
type AnthropicContentBlock = Anthropic.ContentBlockParam;

const DEFAULT_MAX_TOKENS = 4096;

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

export class ChatCompletionToMessagesConverter {
  private streamState: StreamState;
  private customToolNames = new Set<string>();

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (CC → Messages, forward) ---

  convertRequest(params: OpenAI.ChatCompletionCreateParams): Anthropic.MessageCreateParams {
    this.customToolNames.clear();
    const systemBlocks: Anthropic.TextBlockParam[] = [];
    const messages: AnthropicMessage[] = [];

    for (const msg of params.messages) {
      if (msg.role === "system" || msg.role === "developer") {
        systemBlocks.push(...this.extractSystemText(msg));
      } else if (msg.role === "user") {
        messages.push({
          role: "user",
          content: this.convertUserContent(msg.content),
        });
      } else if (msg.role === "assistant") {
        messages.push({
          role: "assistant",
          content: this.convertAssistantMessage(msg),
        });
      } else if (msg.role === "tool") {
        this.appendToolResult(messages, msg);
      }
    }

    const result: Anthropic.MessageCreateParams = {
      model: params.model,
      max_tokens: params.max_completion_tokens ?? params.max_tokens ?? DEFAULT_MAX_TOKENS,
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
      result.stop_sequences = typeof stop === "string" ? [stop] : (stop as string[]);
    }
    const tools = this.convertTools(params.tools ?? []);
    const webSearchTool = this.convertWebSearchOptions((params as any).web_search_options);
    if (webSearchTool) {
      tools.push(webSearchTool);
    }
    if (tools.length > 0) {
      result.tools = tools;
    }
    if (params.tool_choice !== undefined) {
      result.tool_choice = this.convertToolChoice(params.tool_choice, params.parallel_tool_calls);
    } else if (params.parallel_tool_calls === false) {
      result.tool_choice = { type: "auto", disable_parallel_tool_use: true };
    }
    if (params.response_format) {
      result.output_config = this.convertResponseFormat(params.response_format);
    }
    if (params.reasoning_effort != null) {
      result.thinking = this.convertReasoningEffort(params.reasoning_effort as string);
    }
    const metadataUserId = (params.metadata as Record<string, string> | null | undefined)?.user_id;
    if (metadataUserId != null || params.user != null) {
      result.metadata = { user_id: metadataUserId ?? params.user! };
    }
    if (params.service_tier != null) {
      const tier = params.service_tier as string;
      if (tier === "auto" || tier === "standard_only") {
        result.service_tier = tier;
      }
    }
    if (params.stream != null) {
      (result as any).stream = params.stream;
    }

    return result;
  }

  // --- Response conversion (Messages → CC, backward) ---

  convertResponse(message: Anthropic.Message): OpenAI.ChatCompletion {
    const textParts: string[] = [];
    const thinkingParts: string[] = [];
    const reasoningDetails: Array<Record<string, unknown>> = [];
    const toolCalls: OpenAI.ChatCompletionMessageToolCall[] = [];
    const annotations: OpenAI.ChatCompletionMessage.Annotation[] = [];

    for (const block of message.content) {
      if (block.type === "text") {
        textParts.push(block.text);
      } else if (block.type === "thinking") {
        thinkingParts.push(block.thinking);
        reasoningDetails.push({
          type: "reasoning.text",
          text: block.thinking,
          signature: block.signature,
          format: "anthropic-claude-v1",
          index: 0,
        });
      } else if (block.type === "tool_use" || block.type === "server_tool_use") {
        if (this.customToolNames.has(block.name)) {
          toolCalls.push({
            id: block.id,
            type: "custom",
            custom: {
              name: block.name,
              input: JSON.stringify(block.input),
            },
          });
        } else {
          toolCalls.push({
            id: block.id,
            type: "function",
            function: {
              name: block.name,
              arguments: JSON.stringify(block.input),
            },
          });
        }
      } else if (block.type === "web_search_tool_result") {
        const content = block.content;
        if (Array.isArray(content)) {
          for (const result of content) {
            annotations.push({
              type: "url_citation",
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
      role: "assistant",
      content: textParts.join(""),
      refusal: null,
    };

    if (thinkingParts.length > 0) {
      Object.assign(assistantMessage, {
        reasoning: thinkingParts.join(""),
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
      object: "chat.completion",
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
        ...{
          ephemeral_5m_input_tokens: ephemeral5m,
          ephemeral_1h_input_tokens: ephemeral1h,
          web_search: webSearch,
          cache_creation_input_tokens: cacheCreation,
        },
      },
    };
  }

  // --- Stream conversion (Messages → CC, backward) ---

  async *convertStream(
    stream: AsyncIterable<Anthropic.RawMessageStreamEvent>
  ): AsyncIterable<OpenAI.ChatCompletionChunk> {
    for await (const event of stream) {
      const chunk = this.convertStreamEvent(event);
      if (chunk) {
        yield chunk;
      }
    }
  }

  convertStreamEvent(event: Anthropic.RawMessageStreamEvent): OpenAI.ChatCompletionChunk | null {
    switch (event.type) {
      case "message_start":
        return this.handleMessageStart(event);
      case "content_block_start":
        return this.handleContentBlockStart(event);
      case "content_block_delta":
        return this.handleContentBlockDelta(event);
      case "content_block_stop":
        return this.handleContentBlockStop(event);
      case "message_delta":
        return this.handleMessageDelta(event);
      case "message_stop":
        return this.handleMessageStop();
      default:
        return null;
    }
  }

  // --- Stream event handlers ---

  private createStreamState(): StreamState {
    return {
      id: "",
      model: "",
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
    finish_reason: OpenAI.ChatCompletionChunk.Choice["finish_reason"] = null
  ): OpenAI.ChatCompletionChunk {
    return {
      id: this.streamState.id,
      object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000),
      model: this.streamState.model,
      service_tier: null,
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

    return this.makeChunk({ role: "assistant" });
  }

  private handleContentBlockStart(
    event: Anthropic.RawContentBlockStartEvent
  ): OpenAI.ChatCompletionChunk | null {
    const block = event.content_block;

    if (block.type === "text") {
      if (block.text && block.text.trim()) {
        return this.makeChunk({ content: block.text });
      }
      return null;
    }

    if (block.type === "tool_use" || block.type === "server_tool_use") {
      this.streamState.toolCallCounter++;
      const index = this.streamState.toolCallCounter;
      return this.makeChunk({
        tool_calls: [
          {
            index,
            id: block.id,
            type: "function",
            function: { name: block.name, arguments: "" },
          },
        ],
      });
    }

    if (block.type === "web_search_tool_result") {
      const content = block.content;
      if (Array.isArray(content)) {
        for (const result of content) {
          this.streamState.annotations.push({
            type: "url_citation",
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
    event: Anthropic.RawContentBlockDeltaEvent
  ): OpenAI.ChatCompletionChunk | null {
    const delta = event.delta;

    if (delta.type === "text_delta") {
      return this.makeChunk({ content: delta.text });
    }

    if (delta.type === "input_json_delta") {
      const toolIndex = this.streamState.toolCallCounter;
      return this.makeChunk({
        content: "",
        tool_calls: [
          { index: toolIndex, type: "function", function: { arguments: delta.partial_json } },
        ],
      });
    }

    if (delta.type === "thinking_delta") {
      return this.makeChunk({
        content: "",
        ...{
          reasoning: delta.thinking,
          reasoning_details: [
            {
              type: "reasoning.text",
              text: delta.thinking,
              format: "anthropic-claude-v1",
              index: 0,
            },
          ],
        },
      });
    }

    if (delta.type === "signature_delta") {
      return this.makeChunk({
        content: "",
        ...{
          reasoning: null,
          reasoning_details: [
            {
              type: "reasoning.text",
              signature: delta.signature,
              format: "anthropic-claude-v1",
              index: 0,
            },
          ],
        },
      });
    }

    return null;
  }

  private handleContentBlockStop(
    _event: Anthropic.RawContentBlockStopEvent
  ): OpenAI.ChatCompletionChunk {
    return this.makeChunk({ content: "" });
  }

  private handleMessageDelta(event: Anthropic.RawMessageDeltaEvent): OpenAI.ChatCompletionChunk {
    const state = this.streamState;
    const usage = event.usage;
    state.outputTokens = usage.output_tokens;
    if (usage.input_tokens != null) {
      state.inputTokens = usage.input_tokens;
    }
    state.webSearchRequests = usage.server_tool_use?.web_search_requests ?? state.webSearchRequests;

    const delta: OpenAI.ChatCompletionChunk.Choice.Delta = { content: "" };
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
        ...{
          ephemeral_5m_input_tokens: state.ephemeral5mTokens,
          ephemeral_1h_input_tokens: state.ephemeral1hTokens,
          web_search: state.webSearchRequests,
          cache_creation_input_tokens: state.cacheCreationInputTokens,
        },
      },
      completion_tokens_details: {
        reasoning_tokens: 0,
      },
    };

    return {
      id: state.id,
      object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000),
      model: state.model,
      service_tier: null,
      choices: [],
      usage,
    };
  }

  private mapStopReasonToFinishReason(
    reason: Anthropic.Message["stop_reason"] | Anthropic.StopReason | null | string
  ): OpenAI.ChatCompletion.Choice["finish_reason"] {
    switch (reason) {
      case "end_turn":
      case "stop_sequence":
      case "pause_turn":
        return "stop";
      case "tool_use":
        return "tool_calls";
      case "max_tokens":
      case "model_context_window_exceeded":
        return "length";
      case "refusal":
        return "content_filter";
      default:
        return "stop";
    }
  }

  // --- Request conversion helpers ---

  private extractSystemText(
    msg: OpenAI.ChatCompletionSystemMessageParam | OpenAI.ChatCompletionDeveloperMessageParam
  ): Anthropic.TextBlockParam[] {
    if (typeof msg.content === "string") {
      return [{ type: "text", text: msg.content }];
    }
    return msg.content.map(part => ({ type: "text" as const, text: part.text }));
  }

  private convertUserContent(
    content: OpenAI.ChatCompletionUserMessageParam["content"]
  ): AnthropicContentBlock[] {
    if (typeof content === "string") {
      return [{ type: "text", text: content }];
    }
    return content.map(part => this.convertContentPart(part));
  }

  private convertContentPart(part: OpenAI.ChatCompletionContentPart): AnthropicContentBlock {
    switch (part.type) {
      case "text":
        return { type: "text", text: part.text };
      case "image_url":
        return this.convertImageUrl(part);
      case "file":
        return this.convertFile(part);
      default:
        return { type: "text", text: `[Unsupported content type: ${part.type}]` };
    }
  }

  private convertImageUrl(part: OpenAI.ChatCompletionContentPartImage): Anthropic.ImageBlockParam {
    const url = part.image_url.url;
    const dataUriMatch = url.match(/^data:(image\/[a-z+]+);base64,(.+)$/);

    if (dataUriMatch) {
      return {
        type: "image",
        source: {
          type: "base64",
          media_type: dataUriMatch[1] as Anthropic.Base64ImageSource["media_type"],
          data: dataUriMatch[2],
        },
      };
    }

    return {
      type: "image",
      source: { type: "url", url },
    };
  }

  private convertAssistantMessage(
    msg: OpenAI.ChatCompletionAssistantMessageParam
  ): AnthropicContentBlock[] {
    const blocks: AnthropicContentBlock[] = [];

    if (msg.content) {
      if (typeof msg.content === "string") {
        blocks.push({ type: "text", text: msg.content });
      } else {
        for (const part of msg.content) {
          if (part.type === "text") {
            blocks.push({ type: "text", text: part.text });
          }
        }
      }
    }

    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (tc.type === "function") {
          blocks.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: this.safeParseObject(tc.function.arguments),
          });
        } else if (tc.type === "custom") {
          blocks.push({
            type: "tool_use",
            id: tc.id,
            name: tc.custom.name,
            input: this.safeParseObject(tc.custom.input),
          });
        }
      }
    }

    return blocks;
  }

  private appendToolResult(
    messages: AnthropicMessage[],
    msg: OpenAI.ChatCompletionToolMessageParam
  ): void {
    const toolResult: Anthropic.ToolResultBlockParam = {
      type: "tool_result",
      tool_use_id: msg.tool_call_id,
      content:
        typeof msg.content === "string"
          ? [{ type: "text", text: msg.content }]
          : msg.content.map(p => ({ type: "text" as const, text: p.text })),
    };

    const last = messages[messages.length - 1];
    if (last && last.role === "user" && Array.isArray(last.content)) {
      const lastContent = last.content as AnthropicContentBlock[];
      if (lastContent.length > 0 && lastContent[0].type === "tool_result") {
        lastContent.push(toolResult);
        return;
      }
    }

    messages.push({ role: "user", content: [toolResult] });
  }

  private convertTools(tools: OpenAI.ChatCompletionTool[]): Anthropic.ToolUnion[] {
    return tools.map(tool => {
      if (tool.type === "function") {
        return {
          type: "custom",
          name: tool.function.name,
          description: tool.function.description,
          input_schema: (tool.function.parameters ?? {
            type: "object",
          }) as Anthropic.Tool.InputSchema,
        };
      }

      this.customToolNames.add(tool.custom.name);
      return {
        type: "custom",
        name: tool.custom.name,
        description: tool.custom.description ?? "",
        input_schema: this.customToolInputSchema(tool.custom.format),
      };
    });
  }

  private customToolInputSchema(
    format: OpenAI.ChatCompletionCustomTool.Custom["format"]
  ): Anthropic.Tool.InputSchema {
    if (format?.type === "grammar") {
      return {
        type: "object",
        properties: {
          content: {
            type: "string",
            description: `Content must follow grammar (${format.grammar.syntax}): ${format.grammar.definition}`,
          },
        },
        required: ["content"],
      };
    }

    return {
      type: "object",
      properties: { content: { type: "string" } },
      required: ["content"],
    };
  }

  private convertWebSearchOptions(options: unknown): Anthropic.WebSearchTool20250305 | null {
    if (options == null) return null;

    return {
      name: "web_search",
      type: "web_search_20250305",
    };
  }

  private convertFile(part: OpenAI.ChatCompletionContentPart.File): Anthropic.DocumentBlockParam {
    const fileData = part.file.file_data;
    if (!fileData) {
      throw new Error("no file data found");
    }

    const dataUriMatch = fileData.match(/^data:([^;]+);base64,(.+)$/);
    if (!dataUriMatch) {
      throw new Error("invalid base64 file data");
    }

    return {
      type: "document",
      source: {
        type: "base64",
        media_type: dataUriMatch[1] as "application/pdf",
        data: dataUriMatch[2],
      },
    };
  }

  private safeParseObject(value: string | null | undefined): Record<string, unknown> {
    if (!value) return {};
    try {
      return JSON.parse(value) as Record<string, unknown>;
    } catch {
      return {};
    }
  }

  private convertToolChoice(
    choice: OpenAI.ChatCompletionToolChoiceOption,
    parallelToolCalls?: boolean
  ): Anthropic.ToolChoice {
    const disableParallel = parallelToolCalls === false ? true : undefined;

    if (choice === "auto") {
      return disableParallel ? { type: "auto", disable_parallel_tool_use: true } : { type: "auto" };
    }
    if (choice === "required") {
      return disableParallel ? { type: "any", disable_parallel_tool_use: true } : { type: "any" };
    }
    if (choice === "none") return { type: "none" };

    if (typeof choice === "object" && "type" in choice) {
      if (choice.type === "function" && "function" in choice) {
        const name = (choice as OpenAI.ChatCompletionNamedToolChoice).function.name;
        return disableParallel
          ? { type: "tool", name, disable_parallel_tool_use: true }
          : { type: "tool", name };
      }
    }

    return { type: "auto" };
  }

  private convertResponseFormat(
    format: NonNullable<OpenAI.ChatCompletionCreateParams["response_format"]>
  ): Anthropic.OutputConfig {
    if ("json_schema" in format && format.type === "json_schema") {
      return {
        format: {
          type: "json_schema",
          schema: format.json_schema.schema ?? {},
        },
      };
    }
    if (format.type === "json_object") {
      return {
        format: {
          type: "json_schema",
          schema: { type: "object" },
        },
      };
    }
    // text format — no output_config needed, return empty
    return {};
  }

  private convertReasoningEffort(effort: string | null): Anthropic.ThinkingConfigParam {
    if (!effort || effort === "none" || effort === "minimal") {
      return { type: "disabled" };
    }
    // Map OpenAI effort levels to Anthropic budget_tokens
    const budgetMap: Record<string, number> = {
      low: 2048,
      medium: 5120,
      high: 10240,
      xhigh: 20480,
    };
    return {
      type: "enabled",
      budget_tokens: budgetMap[effort] ?? 10240,
    };
  }
}
