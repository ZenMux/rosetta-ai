import type OpenAI from "openai";
import type Anthropic from "@anthropic-ai/sdk";
import { ReasoningEffort } from "openai/resources.js";

type OpenAIMessage = OpenAI.ChatCompletionMessageParam;

type BlockType = "text" | "tool_use" | "thinking" | "web_search_tool_result" | null;

interface StreamState {
  messageStarted: boolean;
  currentBlockIndex: number;
  currentBlockType: BlockType;
  id: string;
  model: string;
  toolCallIndexMap: Map<number, number>;
  stopReason: Anthropic.Message["stop_reason"];
  finished: boolean;
}

export class MessagesToChatCompletionConverter {
  private streamState: StreamState;
  // Set when the forward request enables stream_options.include_usage, so the
  // backward stream knows usage arrives in a trailing usage-only chunk and can
  // defer terminal events until then.
  private streamUsageExpected = false;

  constructor() {
    this.streamState = this.createStreamState();
  }

  // --- Request conversion (Messages → CC, forward) ---

  convertRequest(params: Anthropic.MessageCreateParams): OpenAI.ChatCompletionCreateParams {
    const messages: OpenAIMessage[] = [];

    if (params.system) {
      if (typeof params.system === "string") {
        messages.push({ role: "system", content: params.system });
      } else {
        const text = params.system.map(b => b.text).join("\n");
        messages.push({ role: "system", content: text });
      }
    }

    for (const msg of params.messages) {
      if (msg.role === "user") {
        this.convertUserMessage(messages, msg);
      } else if (msg.role === "assistant") {
        this.convertAssistantMessage(messages, msg);
      }
    }

    const result: OpenAI.ChatCompletionCreateParams = {
      model: params.model,
      messages,
    };

    if (params.max_tokens !== undefined) {
      // Use max_completion_tokens: max_tokens is deprecated and rejected by
      // o-series / reasoning models.
      result.max_completion_tokens = params.max_tokens;
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
        result.web_search_options = webSearchOptions;
      }
    }
    if (params.tool_choice !== undefined) {
      const { toolChoice, parallelToolCalls } = this.convertToolChoice(params.tool_choice);
      result.tool_choice = toolChoice;
      if (parallelToolCalls !== undefined) {
        result.parallel_tool_calls = parallelToolCalls;
      }
    }
    if (params.output_config) {
      result.response_format = this.convertOutputConfig(params.output_config);
    }
    if (params.thinking) {
      result.reasoning_effort = this.convertThinking(params.thinking);
    }
    if (params.metadata?.user_id) {
      result.user = params.metadata.user_id;
    }
    if (params.service_tier != null) {
      result.service_tier =
        params.service_tier === "standard_only" ? "default" : params.service_tier;
    }
    if (params.stream === true) {
      (result as any).stream = true;
      (result as any).stream_options = { include_usage: true };
      this.streamUsageExpected = true;
    }

    return result;
  }

  // --- Response conversion (CC → Messages, backward) ---

  convertResponse(response: OpenAI.ChatCompletion): Anthropic.Message {
    const choice = response.choices[0];
    const msg = choice?.message;

    const content: Anthropic.ContentBlock[] = [];

    // reasoning -> thinking block
    const reasoning = (msg as any)?.reasoning || (msg as any)?.reasoning_content;
    if (reasoning) {
      content.push({
        type: "thinking",
        thinking: reasoning,
        signature: "",
      });
    }

    // annotations -> web_search_tool_result
    if (msg?.annotations && msg.annotations.length > 0) {
      const searchResults = msg.annotations.map(a => ({
        type: "web_search_result" as const,
        title: a.url_citation.title,
        url: a.url_citation.url,
        encrypted_content: "",
        page_age: null,
      }));
      content.push({
        type: "web_search_tool_result",
        content: searchResults,
        tool_use_id: `websearch_${this.generateId()}`,
        caller: { type: "direct" },
      });
    }

    if (msg?.content) {
      content.push({ type: "text", text: msg.content, citations: null });
    }

    if (msg?.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (tc.type === "function") {
          content.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: JSON.parse(tc.function.arguments || "{}"),
            caller: { type: "direct" },
          });
        }
      }
    }

    if (content.length === 0) {
      content.push({ type: "text", text: "", citations: null });
    }

    return {
      id: response.id,
      type: "message",
      role: "assistant",
      model: response.model as Anthropic.Model,
      content,
      stop_reason: this.mapFinishReasonToStopReason(choice?.finish_reason),
      stop_sequence: null,
      usage: this.buildUsage(response.usage),
      container: null,
    };
  }

  private buildUsage(usage?: OpenAI.CompletionUsage): any {
    const u = usage ?? ({} as OpenAI.CompletionUsage);
    const inputTokens = u.prompt_tokens ?? 0;
    const outputTokens = u.completion_tokens ?? 0;
    const totalTokens = u.total_tokens ?? inputTokens + outputTokens;
    const cachedTokens = u.prompt_tokens_details?.cached_tokens ?? 0;
    const reasoningTokens = u.completion_tokens_details?.reasoning_tokens ?? 0;
    const webSearch: number = (u.prompt_tokens_details as any)?.web_search ?? 0;

    return {
      completion_tokens: outputTokens,
      prompt_tokens: inputTokens,
      total_tokens: totalTokens,
      completion_tokens_details: {
        ...u.completion_tokens_details,
        reasoning_tokens: reasoningTokens,
      },
      prompt_tokens_details: {
        ...u.prompt_tokens_details,
        cached_tokens: cachedTokens,
        ...(webSearch > 0 ? { web_search: webSearch } : {}),
      },
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      cache_read_input_tokens: cachedTokens,
      server_tool_use:
        webSearch > 0
          ? {
              web_fetch_requests: 0,
              web_search_requests: webSearch,
            }
          : null,
      cache_creation_input_tokens: 0,
      service_tier: "standard",
      audio_input_tokens: u.prompt_tokens_details?.audio_tokens ?? 0,
      audio_cache_read_tokens: (u.prompt_tokens_details as any)?.audio_cached_tokens ?? 0,
    };
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  // --- Stream conversion (CC → Messages, backward) ---

  async *convertStream(
    stream: AsyncIterable<OpenAI.ChatCompletionChunk>
  ): AsyncIterable<Anthropic.RawMessageStreamEvent> {
    for await (const chunk of stream) {
      const events = this.convertStreamChunk(chunk);
      for (const event of events) {
        yield event;
      }
    }

    // Safety flush: if the backend signalled finish but never sent the trailing
    // usage chunk, emit terminal events so the stream is well-formed.
    for (const event of this.flushTerminalEvents()) {
      yield event;
    }
  }

  /**
   * Emit the trailing terminal events (message_delta + message_stop) when the
   * backend signalled a finish reason but never sent the trailing usage-only
   * chunk that normally triggers them. Returns an empty array when the stream
   * already terminated cleanly.
   *
   * Callers that drive the stream chunk-by-chunk via convertStreamChunk (instead
   * of convertStream) must invoke this once the upstream stream is exhausted, or
   * a backend that omits the usage chunk will leave the converted stream without
   * a message_stop.
   */
  flushTerminalEvents(): Anthropic.RawMessageStreamEvent[] {
    const events: Anthropic.RawMessageStreamEvent[] = [];
    if (this.streamState.stopReason !== null && !this.streamState.finished) {
      this.emitTerminalEvents(this.streamState, events, undefined);
    }
    return events;
  }

  convertStreamChunk(chunk: OpenAI.ChatCompletionChunk): Anthropic.RawMessageStreamEvent[] {
    const state = this.streamState;
    const events: Anthropic.RawMessageStreamEvent[] = [];
    const choice = chunk.choices[0];

    if (!choice) {
      // Usage-only chunk (empty choices) emitted last when stream_options.include_usage is set.
      if (chunk.usage) {
        this.emitTerminalEvents(state, events, chunk.usage);
      }
      return events;
    }

    state.id = chunk.id;
    state.model = chunk.model;

    const delta = choice.delta;

    // First chunk with role - emit message_start
    if (!state.messageStarted && delta.role === "assistant") {
      state.messageStarted = true;
      events.push({
        type: "message_start",
        message: {
          id: chunk.id,
          type: "message",
          role: "assistant",
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
    }

    // Tool call deltas
    if (delta.tool_calls && delta.tool_calls.length) {
      for (const tc of delta.tool_calls) {
        if (tc.id && tc.function?.name) {
          this.transitionBlock(state, events, "tool_use");
          state.toolCallIndexMap.set(tc.index, state.currentBlockIndex);

          events.push({
            type: "content_block_start",
            index: state.currentBlockIndex,
            content_block: {
              type: "tool_use",
              id: tc.id,
              name: tc.function.name,
              input: {},
              caller: { type: "direct" },
            },
          });

          if (tc.function.arguments) {
            events.push({
              type: "content_block_delta",
              index: state.currentBlockIndex,
              delta: {
                type: "input_json_delta",
                partial_json: tc.function.arguments,
              },
            });
          }
        } else if (tc.function?.arguments) {
          const blockIndex = state.toolCallIndexMap.get(tc.index) ?? state.currentBlockIndex;
          events.push({
            type: "content_block_delta",
            index: blockIndex,
            delta: {
              type: "input_json_delta",
              partial_json: tc.function.arguments,
            },
          });
        }
      }
    } else if (this.hasReasoning(delta)) {
      // Reasoning delta -> thinking block
      if (state.currentBlockType !== "thinking") {
        this.transitionBlock(state, events, "thinking");
        events.push({
          type: "content_block_start",
          index: state.currentBlockIndex,
          content_block: { type: "thinking", thinking: "", signature: "" },
        });
      }
      events.push({
        type: "content_block_delta",
        index: state.currentBlockIndex,
        delta: { type: "thinking_delta", thinking: this.getReasoning(delta) },
      });
    } else if (delta.content) {
      // Text content delta
      if (state.currentBlockType !== "text") {
        this.transitionBlock(state, events, "text");
        events.push({
          type: "content_block_start",
          index: state.currentBlockIndex,
          content_block: { type: "text", text: "", citations: null },
        });
      }
      events.push({
        type: "content_block_delta",
        index: state.currentBlockIndex,
        delta: { type: "text_delta", text: delta.content },
      });
    } else {
      // Annotations delta -> web_search_tool_result
      const annotations = (delta as any).annotations as
        | OpenAI.ChatCompletionMessage.Annotation[]
        | undefined;
      if (annotations && annotations.length > 0) {
        this.transitionBlock(state, events, "web_search_tool_result");
        const searchResults = annotations.map(a => ({
          type: "web_search_result" as const,
          title: a.url_citation.title,
          url: a.url_citation.url,
          encrypted_content: "",
          page_age: null,
        }));
        events.push({
          type: "content_block_start",
          index: state.currentBlockIndex,
          content_block: {
            type: "web_search_tool_result",
            content: searchResults,
            tool_use_id: `websearch_${this.generateId()}`,
            caller: { type: "direct" },
          },
        });
      }
    }

    // Finish reason - close the open block, then emit terminal events.
    // When stream_options.include_usage is set, usage arrives in a trailing
    // usage-only chunk; defer message_delta/message_stop until then so we can
    // report real output_tokens. Otherwise (usage already on this chunk, or no
    // usage at all) emit immediately.
    if (choice.finish_reason) {
      state.stopReason = this.mapFinishReasonToStopReason(choice.finish_reason);

      if (state.currentBlockType !== null) {
        events.push({
          type: "content_block_stop",
          index: state.currentBlockIndex,
        });
        state.currentBlockType = null;
      }

      if (chunk.usage) {
        this.emitTerminalEvents(state, events, chunk.usage);
      } else if (!this.streamUsageExpected) {
        this.emitTerminalEvents(state, events, undefined);
      }
    }

    return events;
  }

  private emitTerminalEvents(
    state: StreamState,
    events: Anthropic.RawMessageStreamEvent[],
    usage: OpenAI.CompletionUsage | undefined
  ): void {
    if (state.finished) return;
    state.finished = true;

    events.push({
      type: "message_delta",
      delta: {
        stop_reason: state.stopReason,
        stop_sequence: null,
        container: null,
      },
      usage: this.buildUsage(usage),
    } as any);

    events.push({ type: "message_stop" });
  }

  // --- Stream helpers ---

  private createStreamState(): StreamState {
    return {
      messageStarted: false,
      currentBlockIndex: -1,
      currentBlockType: null,
      id: "",
      model: "",
      toolCallIndexMap: new Map(),
      stopReason: null,
      finished: false,
    };
  }

  private transitionBlock(
    state: StreamState,
    events: Anthropic.RawMessageStreamEvent[],
    newType: BlockType
  ): void {
    if (state.currentBlockType !== null) {
      events.push({ type: "content_block_stop", index: state.currentBlockIndex });
    }
    state.currentBlockIndex++;
    state.currentBlockType = newType;
  }

  private hasReasoning(delta: Record<string, any>): boolean {
    return !!delta["reasoning"] || !!delta["reasoning_content"];
  }

  private getReasoning(delta: Record<string, any>): string {
    return delta["reasoning"] || delta["reasoning_content"] || "";
  }

  private mapFinishReasonToStopReason(
    reason: OpenAI.ChatCompletion.Choice["finish_reason"] | string | undefined
  ): Anthropic.Message["stop_reason"] {
    switch (reason) {
      case "stop":
        return "end_turn";
      case "tool_calls":
      case "function_call":
        return "tool_use";
      case "length":
        return "max_tokens";
      case "content_filter":
        return "refusal";
      default:
        return "end_turn";
    }
  }

  // --- Request conversion helpers ---

  private convertUserMessage(messages: OpenAIMessage[], msg: Anthropic.MessageParam): void {
    if (typeof msg.content === "string") {
      messages.push({ role: "user", content: msg.content });
      return;
    }

    const textParts: string[] = [];
    const contentParts: OpenAI.ChatCompletionContentPart[] = [];
    let hasNonText = false;
    const toolResults: Anthropic.ToolResultBlockParam[] = [];

    for (const block of msg.content) {
      switch (block.type) {
        case "text":
          textParts.push(block.text);
          contentParts.push({ type: "text", text: block.text });
          break;
        case "image":
          hasNonText = true;
          contentParts.push(this.convertImageBlock(block as Anthropic.ImageBlockParam));
          break;
        case "document":
          hasNonText = true;
          contentParts.push(...this.convertDocumentBlock(block as Anthropic.DocumentBlockParam));
          break;
        case "tool_result":
          toolResults.push(block as Anthropic.ToolResultBlockParam);
          break;
      }
    }

    if (toolResults.length > 0) {
      for (const tr of toolResults) {
        messages.push({
          role: "tool",
          tool_call_id: tr.tool_use_id,
          content: this.convertToolResultContent(tr.content),
        });
      }
      return;
    }

    if (!hasNonText && textParts.length === 1) {
      messages.push({ role: "user", content: textParts[0] });
      return;
    }

    messages.push({ role: "user", content: contentParts });
  }

  private convertToolResultContent(content: Anthropic.ToolResultBlockParam["content"]): string {
    if (typeof content === "string") {
      return content;
    }
    if (!content) {
      return "";
    }

    const texts: string[] = [];
    for (const block of content) {
      if (block.type === "text") {
        texts.push(block.text);
      } else if (block.type === "search_result") {
        // OpenAI tool role has no search_result type; keep the text portions
        const sr = block as Anthropic.SearchResultBlockParam;
        for (const inner of sr.content) {
          texts.push(inner.text);
        }
      } else if (block.type === "tool_reference") {
        // keep only the referenced tool name
        texts.push((block as Anthropic.ToolReferenceBlockParam).tool_name);
      }
      // image/document blocks are unsupported by OpenAI's tool role — skip
    }
    return texts.join("\n");
  }

  private convertImageBlock(
    block: Anthropic.ImageBlockParam
  ): OpenAI.ChatCompletionContentPartImage {
    const source = block.source;

    if (source.type === "base64") {
      return {
        type: "image_url",
        image_url: {
          url: `data:${source.media_type};base64,${source.data}`,
        },
      };
    }

    return {
      type: "image_url",
      image_url: { url: (source as Anthropic.URLImageSource).url },
    };
  }

  private convertDocumentBlock(
    block: Anthropic.DocumentBlockParam
  ): OpenAI.ChatCompletionContentPart[] {
    const source = block.source;

    if (source.type === "base64") {
      const src = source as Anthropic.Base64PDFSource;
      return [
        {
          type: "file",
          file: {
            file_data: `data:${src.media_type};base64,${src.data}`,
            filename: block.title ?? undefined,
          },
        },
      ];
    }

    if (source.type === "url") {
      const src = source as Anthropic.URLPDFSource;
      return [
        {
          type: "file",
          file: {
            file_data: src.url,
            filename: block.title ?? undefined,
          },
        },
      ];
    }

    // plain text source — encode as text content
    if (source.type === "text") {
      const src = source as Anthropic.PlainTextSource;
      return [{ type: "text", text: src.data }];
    }

    // content source — unpack nested text/image blocks into content parts
    if (source.type === "content") {
      const src = source as Anthropic.ContentBlockSource;
      if (typeof src.content === "string") {
        return [{ type: "text", text: src.content }];
      }
      const parts: OpenAI.ChatCompletionContentPart[] = [];
      for (const nested of src.content) {
        if (nested.type === "text") {
          parts.push({ type: "text", text: nested.text });
        } else if (nested.type === "image") {
          parts.push(this.convertImageBlock(nested as Anthropic.ImageBlockParam));
        }
      }
      return parts;
    }

    return [{ type: "text", text: "[Unsupported document source]" }];
  }

  private convertAssistantMessage(messages: OpenAIMessage[], msg: Anthropic.MessageParam): void {
    if (typeof msg.content === "string") {
      messages.push({ role: "assistant", content: msg.content });
      return;
    }

    let textContent: string | undefined;
    const toolCalls: OpenAI.ChatCompletionMessageToolCall[] = [];

    let thinkingContent: string | undefined;
    const reasoningDetails: Array<Record<string, unknown>> = [];

    for (const block of msg.content) {
      if (block.type === "text") {
        textContent = textContent ? textContent + block.text : block.text;
      } else if (block.type === "thinking") {
        const tb = block as Anthropic.ThinkingBlockParam;
        thinkingContent = thinkingContent ? thinkingContent + tb.thinking : tb.thinking;
        reasoningDetails.push({
          type: "reasoning.text",
          text: tb.thinking,
          signature: tb.signature,
          format: "anthropic-claude-v1",
          index: 0,
        });
      } else if (block.type === "tool_use") {
        const tu = block as Anthropic.ToolUseBlockParam;
        toolCalls.push({
          id: tu.id,
          type: "function",
          function: {
            name: tu.name,
            arguments: JSON.stringify(tu.input),
          },
        });
      }
    }

    const assistantMsg: OpenAI.ChatCompletionAssistantMessageParam = {
      role: "assistant",
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
      if ("input_schema" in t) {
        functionTools.push({
          type: "function",
          function: {
            name: (t as Anthropic.Tool).name,
            description: (t as Anthropic.Tool).description,
            parameters: (t as Anthropic.Tool).input_schema as unknown as Record<string, unknown>,
          },
        });
      } else if (
        "type" in t &&
        (t.type === "web_search_20250305" || t.type === "web_search_20260209")
      ) {
        const ws = t as Anthropic.WebSearchTool20250305;
        webSearchOptions = {};
        if (ws.max_uses != null) {
          webSearchOptions["max_uses"] = ws.max_uses;
        }
        if (ws.user_location != null) {
          webSearchOptions["user_location"] = ws.user_location;
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
      "disable_parallel_tool_use" in choice && choice.disable_parallel_tool_use === true
        ? false
        : undefined;

    switch (choice.type) {
      case "auto":
        return { toolChoice: "auto", parallelToolCalls };
      case "any":
        return { toolChoice: "required", parallelToolCalls };
      case "none":
        return { toolChoice: "none" };
      case "tool":
        return {
          toolChoice: {
            type: "function",
            function: { name: (choice as Anthropic.ToolChoiceTool).name },
          },
          parallelToolCalls,
        };
      default:
        return { toolChoice: "auto" };
    }
  }

  private convertOutputConfig(
    config: Anthropic.OutputConfig
  ): OpenAI.ChatCompletionCreateParams["response_format"] {
    if (config.format?.type === "json_schema") {
      return {
        type: "json_schema",
        json_schema: {
          name: "response",
          schema: config.format.schema,
          strict: true,
        },
      };
    }
    return { type: "text" };
  }

  private convertThinking(thinking: Anthropic.ThinkingConfigParam): ReasoningEffort | null {
    if (thinking.type === "disabled") return "none";
    if (thinking.type === "enabled") {
      const budget = (thinking as Anthropic.ThinkingConfigEnabled).budget_tokens;
      if (budget <= 2048) return "low";
      if (budget <= 5120) return "medium";
      if (budget <= 10240) return "high";
      return "high";
    }
    // adaptive
    return "medium";
  }
}
