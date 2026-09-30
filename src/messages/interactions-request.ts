import type Anthropic from "@anthropic-ai/sdk";
import {
  InteractionsRequestError,
  type InteractionContent,
  type InteractionCreateParams,
  type InteractionGenerationConfig,
  type InteractionStep,
} from "../interactions/types";

type RecordValue = Record<string, any>;
export function validateMessagesInteractionsParameters(
  params: Anthropic.MessageCreateParams
): void {
  const p = params as unknown as RecordValue;
  const known = new Set([
    "model",
    "messages",
    "max_tokens",
    "system",
    "stream",
    "tools",
    "tool_choice",
    "stop_sequences",
    "thinking",
    "output_config",
  ]);
  for (const [key, value] of Object.entries(p))
    if (value != null && !known.has(key))
      fail(`${key} has no equivalent in Messages to Interactions V1`);
  if (p.max_tokens != null && (!Number.isSafeInteger(p.max_tokens) || p.max_tokens <= 0))
    fail("max_tokens must be a positive integer");
  if (p.stream != null && typeof p.stream !== "boolean") fail("stream must be a boolean");
  if (p.thinking != null) {
    const thinking = object(p.thinking, "thinking");
    if (thinking.type !== "adaptive")
      fail(
        "Interactions V1 supports adaptive thinking; disabling thinking or setting a token budget has no equivalent"
      );
    if (
      Object.entries(thinking).some(
        ([key, value]) => value != null && !["type", "display"].includes(key)
      )
    )
      fail("Unsupported thinking option for Interactions V1");
    if (thinking.display != null && !["summarized", "omitted"].includes(thinking.display))
      fail("Unsupported thinking.display");
  }
  if (p.output_config != null) {
    const config = object(p.output_config, "output_config");
    if (
      Object.entries(config).some(
        ([key, value]) => value != null && !["effort", "format"].includes(key)
      )
    )
      fail("Unsupported output_config option");
    if (config.effort != null && !["low", "medium", "high"].includes(config.effort))
      fail("Unsupported output_config.effort for Interactions V1");
    if (config.format != null) {
      const format = object(config.format, "output_config.format");
      if (format.type !== "json_schema" || format.strict === true)
        fail("Unsupported output_config.format or strict guarantee");
      object(format.schema, "output_config.format.schema");
    }
  }
  if (
    p.stop_sequences != null &&
    (!Array.isArray(p.stop_sequences) ||
      p.stop_sequences.some((s: unknown) => typeof s !== "string"))
  )
    fail("stop_sequences must be an array of strings");
  if (p.tools != null && !Array.isArray(p.tools)) fail("tools must be an array");
  for (const raw of p.tools ?? []) {
    const tool = object(raw, "tool");
    if (tool.type != null && tool.type !== "custom")
      fail("Only custom function tools are supported by this converter");
    if (tool.strict === true) fail("Interactions V1 cannot guarantee tool strict=true");
    for (const key of [
      "cache_control",
      "defer_loading",
      "allowed_callers",
      "input_examples",
      "eager_input_streaming",
    ])
      if (tool[key] != null) fail(`tool.${key} has no equivalent in Interactions V1`);
  }
  if (p.tool_choice != null) {
    const choice = object(p.tool_choice, "tool_choice");
    if (!["auto", "any", "none", "tool"].includes(choice.type))
      fail("Unsupported tool_choice.type");
    if (choice.disable_parallel_tool_use === true)
      fail("Interactions V1 cannot enforce disable_parallel_tool_use=true");
  }
}

export function convertMessagesInteractionRequest(
  params: Anthropic.MessageCreateParams,
  onMessage: (index: number, start: number, end: number) => void
): InteractionCreateParams {
  validateMessagesInteractionsParameters(params);
  const p = params as unknown as RecordValue;
  const model = nonempty(p.model, "model");
  if (!Array.isArray(p.messages) || !p.messages.length) fail("messages must be a nonempty array");
  const input: InteractionStep[] = [];
  const calls = new Map<string, string>();
  const results = new Set<string>();
  for (const [index, raw] of p.messages.entries()) {
    const message = object(raw, "message");
    if (!["user", "assistant"].includes(message.role))
      fail("Messages role must be user or assistant");
    if (message.interactions != null)
      fail("Native history requires the gateway Interactions extension");
    const start = input.length;
    const blocks =
      typeof message.content === "string"
        ? [{ type: "text", text: message.content }]
        : array(message.content, "content");
    let content: InteractionContent[] = [];
    const flush = () => {
      if (content.length)
        input.push({ type: message.role === "user" ? "user_input" : "model_output", content });
      content = [];
    };
    for (const rawBlock of blocks) {
      const block = object(rawBlock, "content block");
      if (block.interactions != null)
        fail("Native history requires the gateway Interactions extension");
      if (block.cache_control != null) fail("cache_control has no equivalent in Interactions V1");
      if (block.type === "tool_use") {
        flush();
        if (message.role !== "assistant") fail("tool_use requires assistant role");
        const id = nonempty(block.id, "tool_use.id"),
          name = nonempty(block.name, "tool_use.name");
        if (calls.has(id)) fail("Duplicate tool_use.id");
        if (block.caller != null && block.caller.type !== "direct") fail("Unsupported tool caller");
        calls.set(id, name);
        input.push({
          type: "function_call",
          id,
          name,
          arguments: structuredClone(object(block.input, "tool_use.input")),
        });
      } else if (block.type === "tool_result") {
        flush();
        if (message.role !== "user") fail("tool_result requires user role");
        const id = nonempty(block.tool_use_id, "tool_result.tool_use_id");
        if (!calls.has(id) || results.has(id))
          fail("tool_result must match one preceding tool_use");
        if (block.is_error != null && typeof block.is_error !== "boolean")
          fail("tool_result.is_error must be a boolean");
        results.add(id);
        input.push({
          type: "function_result",
          call_id: id,
          name: calls.get(id),
          ...(block.is_error != null && { is_error: block.is_error }),
          result:
            typeof block.content === "string"
              ? block.content
              : array(block.content ?? [], "tool_result.content").map(convertContent),
        });
      } else if (block.type === "thinking") {
        flush();
        if (message.role !== "assistant") fail("thinking requires assistant role");
        if (block.signature)
          fail("Thinking signatures require a supported gateway Interactions carrier");
        input.push({
          type: "thought",
          summary: [{ type: "text", text: string(block.thinking, "thinking") }],
        });
      } else {
        content.push(convertContent(block));
      }
    }
    flush();
    onMessage(index, start, input.length);
  }
  const config: InteractionGenerationConfig = {};
  if (p.max_tokens != null) config.max_output_tokens = p.max_tokens;
  if (p.stop_sequences != null) config.stop_sequences = [...p.stop_sequences];
  if (p.output_config?.effort != null) config.thinking_level = p.output_config.effort;
  if (p.thinking != null)
    config.thinking_summaries = p.thinking.display === "omitted" ? "none" : "auto";
  const tools: NonNullable<InteractionCreateParams["tools"]> = [];
  const names = new Set<string>();
  for (const tool of p.tools ?? []) {
    const name = nonempty(tool.name, "tool.name");
    if (names.has(name)) fail("Duplicate tool name");
    names.add(name);
    tools.push({
      type: "function",
      name,
      parameters: structuredClone(object(tool.input_schema, "tool.input_schema")),
      ...(tool.description != null && {
        description: string(tool.description, "tool.description"),
      }),
    });
  }
  if (p.tool_choice != null) {
    const choice = p.tool_choice;
    if (choice.type === "tool") {
      if (!names.has(choice.name)) fail("tool_choice must refer to a declared function");
      config.tool_choice = { allowed_tools: { mode: "any", tools: [choice.name] } };
    } else config.tool_choice = choice.type;
    if (choice.type === "any" && !tools.length) fail("tool_choice=any requires tools");
  }
  const result: InteractionCreateParams = { model, input, store: false, stream: p.stream ?? false };
  if (p.system != null)
    result.system_instruction =
      typeof p.system === "string"
        ? p.system
        : array(p.system, "system")
            .map(block => {
              if (block.type !== "text" || block.cache_control != null)
                fail("system supports plain text only");
              return string(block.text, "system.text");
            })
            .join("\n");
  if (Object.keys(config).length) result.generation_config = config;
  if (tools.length) result.tools = tools;
  if (p.output_config?.format)
    result.response_format = {
      type: "text",
      mime_type: "application/json",
      schema: structuredClone(p.output_config.format.schema),
    };
  return result;
}

function convertContent(raw: unknown): InteractionContent {
  const block = object(raw, "content block");
  if (block.cache_control != null) fail("cache_control has no equivalent in Interactions V1");
  if (block.type === "text") {
    if (block.citations != null && (!Array.isArray(block.citations) || block.citations.length))
      fail("Citation history requires the gateway Interactions extension");
    return { type: "text", text: string(block.text, "text") };
  }
  if (block.type === "image" || block.type === "document") {
    if (block.title != null || block.context != null || block.citations != null)
      fail("Document metadata/citations have no equivalent in Interactions V1");
    const source = object(block.source, "source");
    if (source.type === "base64")
      return {
        type: block.type,
        mime_type: nonempty(source.media_type, "source.media_type"),
        data: nonempty(source.data, "source.data"),
      };
    if (source.type === "url") return { type: block.type, uri: nonempty(source.url, "source.url") };
    fail("Only base64 or URL media sources are supported");
  }
  return fail(`Unsupported Messages content block: ${block.type}`);
}
function array(value: unknown, name: string): RecordValue[] {
  if (!Array.isArray(value)) fail(`${name} must be an array`);
  return value.map(item => object(item, name));
}
function object(value: unknown, name: string): RecordValue {
  if (!value || typeof value !== "object" || Array.isArray(value))
    fail(`${name} must be an object`);
  return value as RecordValue;
}
function string(value: unknown, name: string): string {
  if (typeof value !== "string") fail(`${name} must be a string`);
  return value;
}
function nonempty(value: unknown, name: string): string {
  const text = string(value, name);
  if (!text.trim()) fail(`${name} must not be empty`);
  return text;
}
function fail(message: string): never {
  throw new InteractionsRequestError(message);
}
