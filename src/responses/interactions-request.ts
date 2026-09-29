import type OpenAI from "openai";
import {
  InteractionsRequestError,
  type InteractionContent,
  type InteractionCreateParams,
  type InteractionGenerationConfig,
  type InteractionStep,
} from "../interactions/types";

type Params = OpenAI.Responses.ResponseCreateParams;

export function validateResponsesInteractionsParameters(params: Params): void {
  if (params.include != null && !Array.isArray(params.include)) fail("include must be an array");
  if (params.tools != null && !Array.isArray(params.tools)) fail("tools must be an array");
  if (params.reasoning != null) {
    const reasoning = object(params.reasoning, "reasoning");
    for (const [key, value] of Object.entries(reasoning))
      if (value != null && !["effort", "summary"].includes(key))
        fail(`Unsupported reasoning.${key} for Interactions V1`);
  }
  if (params.text?.format?.type === "json_schema" && params.text.format.strict === true)
    fail("Interactions V1 cannot guarantee text.format.strict=true");
  if (params.tools?.some(tool => tool?.type === "function" && tool.strict === true))
    fail("Interactions V1 cannot guarantee function strict=true");
  for (const key of [
    "temperature",
    "top_p",
    "top_logprobs",
    "max_tool_calls",
    "prompt_cache_key",
    "prompt_cache_retention",
    "safety_identifier",
    "user",
    "prompt",
    "conversation",
  ] as const) {
    if ((params as unknown as Record<string, unknown>)[key] != null)
      fail(`${key} has no equivalent in Interactions V1`);
  }
  if (params.background === true)
    fail("background=true is not supported; use synchronous requests");
  if (params.previous_response_id)
    fail("previous_response_id is not supported; send complete input history");
  if (params.parallel_tool_calls === false)
    fail("Interactions V1 cannot enforce parallel_tool_calls=false");
  if (params.truncation != null && params.truncation !== "disabled")
    fail("truncation=auto has no equivalent in Interactions V1");
  if (params.service_tier != null && params.service_tier !== "auto")
    fail("service_tier has no equivalent in Interactions V1");
  if (params.stream_options != null && Object.keys(params.stream_options).length)
    fail("stream_options has no equivalent in Interactions V1");
  if (params.text?.verbosity != null) fail("text.verbosity has no equivalent in Interactions V1");
  if (
    params.reasoning?.effort != null &&
    !["minimal", "low", "medium", "high"].includes(params.reasoning.effort)
  )
    fail("Unsupported reasoning.effort for Interactions V1");
  if (params.reasoning?.summary != null && params.reasoning.summary !== "auto")
    fail("Only reasoning.summary=auto has an equivalent in Interactions V1");
  if (params.include?.some(value => value !== "reasoning.encrypted_content"))
    fail("Unsupported include for Interactions V1");
  const known = new Set([
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "text",
    "reasoning",
    "include",
    "max_output_tokens",
    "metadata",
    "store",
    "stream",
    "background",
    "previous_response_id",
    "truncation",
    "service_tier",
    "stream_options",
    "temperature",
    "top_p",
    "top_logprobs",
    "max_tool_calls",
    "prompt_cache_key",
    "prompt_cache_retention",
    "safety_identifier",
    "user",
    "prompt",
    "conversation",
  ]);
  for (const [key, value] of Object.entries(params)) {
    if (value != null && !known.has(key)) fail(`Unsupported Responses parameter: ${key}`);
  }
  for (const key of ["store", "stream", "background", "parallel_tool_calls"] as const) {
    if (params[key] != null && typeof params[key] !== "boolean") fail(`${key} must be a boolean`);
  }
  if (
    params.max_output_tokens != null &&
    (!Number.isSafeInteger(params.max_output_tokens) || params.max_output_tokens <= 0)
  )
    fail("max_output_tokens must be a positive integer");
}

/** Input ranges let a gateway restore native history/media without reimplementing conversion. */
export function convertResponsesInteractionRequest(
  params: Params,
  onItem: (index: number, start: number, end: number) => void
): InteractionCreateParams {
  validateResponsesInteractionsParameters(params);
  if (typeof params.model !== "string" || !params.model) fail("model is required");
  const input: InteractionStep[] = [];
  const system: string[] = [];
  if (params.instructions != null) system.push(string(params.instructions, "instructions"));
  const names = new Map<string, string>();
  const results = new Set<string>();
  if (typeof params.input === "string")
    input.push({ type: "user_input", content: [{ type: "text", text: params.input }] });
  else if (Array.isArray(params.input)) {
    for (const [index, raw] of params.input.entries()) {
      const item = object(raw, "input item");
      const start = input.length;
      if (item.interactions != null)
        fail("Native input history requires the gateway Interactions extension");
      if (item.type === "function_call") {
        const id = nonempty(item.call_id, "function_call.call_id");
        const name = nonempty(item.name, "function_call.name");
        if (names.has(id)) fail("Duplicate function_call.call_id");
        names.set(id, name);
        input.push({ type: "function_call", id, name, arguments: parseArguments(item.arguments) });
      } else if (item.type === "function_call_output") {
        const id = nonempty(item.call_id, "function_call_output.call_id");
        if (!names.has(id) || results.has(id))
          fail("Function result must match one preceding call");
        results.add(id);
        const result =
          typeof item.output === "string" ? item.output : parts(item.output).map(convertContent);
        input.push({ type: "function_result", call_id: id, name: names.get(id), result });
      } else if (item.type === "reasoning") {
        if (item.encrypted_content != null)
          fail("Reasoning encrypted_content requires a supported gateway signature carrier");
        const summary = parts(item.summary).map(part => {
          if (part.type !== "summary_text") return fail("Unsupported reasoning summary part");
          return { type: "text", text: string(part.text, "summary.text") };
        });
        if (summary.length) input.push({ type: "thought", summary });
      } else if (item.type == null || item.type === "message") {
        const role = item.role;
        const content =
          typeof item.content === "string"
            ? [{ type: "text", text: item.content }]
            : parts(item.content).map(convertContent);
        if (role === "system" || role === "developer") {
          if (content.some(part => part.type !== "text"))
            fail("System/developer messages must contain text");
          system.push(content.map(part => part.text).join(""));
        } else if (role === "user" || role === "assistant") {
          input.push({ type: role === "user" ? "user_input" : "model_output", content });
        } else fail("Unsupported Responses message role");
      } else fail(`Unsupported Responses input item: ${item.type}`);
      onItem(index, start, input.length);
    }
  } else fail("input must be a string or an array of items");

  const config: InteractionGenerationConfig = {};
  if (params.max_output_tokens != null) config.max_output_tokens = params.max_output_tokens;
  if (params.reasoning?.effort != null)
    config.thinking_level = params.reasoning
      .effort as InteractionGenerationConfig["thinking_level"];
  if (params.reasoning?.summary === "auto") config.thinking_summaries = "auto";
  const tools: NonNullable<InteractionCreateParams["tools"]> = [];
  const toolNames = new Set<string>();
  for (const raw of params.tools ?? []) {
    const tool = object(raw, "tool");
    if (tool.type === "function") {
      const name = nonempty(tool.name, "function.name");
      if (toolNames.has(name)) fail("Duplicate function tool name");
      toolNames.add(name);
      if (tool.strict === true) fail("Interactions V1 cannot guarantee function strict=true");
      if (tool.namespace != null) fail("Namespaced tools are not supported by Interactions V1");
      tools.push({
        type: "function",
        name,
        ...(tool.description != null && { description: string(tool.description, "description") }),
        ...(tool.parameters != null && { parameters: object(tool.parameters, "parameters") }),
      });
    } else if (tool.type === "web_search" || tool.type === "web_search_preview") {
      if (Object.entries(tool).some(([key, value]) => key !== "type" && value != null))
        fail("Web search options have no equivalent in Interactions V1");
      tools.push({ type: "google_search" });
    } else fail(`Unsupported Interactions tool: ${tool.type}`);
  }
  if (params.tool_choice != null) {
    const choice = params.tool_choice;
    if (typeof choice === "string") {
      if (!["auto", "none", "required"].includes(choice)) fail("Unsupported tool_choice");
      config.tool_choice = choice === "required" ? "any" : choice;
    } else {
      const value = object(choice, "tool_choice");
      const chosen =
        value.type === "function"
          ? [value]
          : value.type === "allowed_tools"
            ? parts(value.tools)
            : fail("Unsupported tool_choice type");
      const mode = value.type === "function" ? "required" : value.mode;
      if (mode !== "auto" && mode !== "required") fail("Unsupported allowed_tools mode");
      if (!chosen.length) fail("allowed_tools must not be empty");
      config.tool_choice = {
        allowed_tools: {
          mode: mode === "required" ? "any" : "auto",
          tools: chosen.map(tool => {
            if (tool.type !== "function" || !toolNames.has(tool.name))
              fail("tool_choice must refer to a declared function");
            return tool.name as string;
          }),
        },
      };
    }
    if (config.tool_choice === "any" && !tools.length) fail("required tool_choice requires tools");
  }
  const result: InteractionCreateParams = {
    model: params.model,
    input,
    store: params.store ?? true,
    stream: params.stream ?? false,
    ...(system.length && { system_instruction: system.join("\n") }),
    ...(tools.length && { tools }),
    ...(Object.keys(config).length && { generation_config: config }),
  };
  const format = params.text?.format;
  if (format != null) {
    if (format.type === "text") result.response_format = { type: "text", mime_type: "text/plain" };
    else if (format.type === "json_object")
      result.response_format = { type: "text", mime_type: "application/json" };
    else if (format.type === "json_schema") {
      if (format.strict === true) fail("Interactions V1 cannot guarantee text.format.strict=true");
      result.response_format = {
        type: "text",
        mime_type: "application/json",
        schema: structuredClone(object(format.schema, "text.format.schema")),
      };
    } else fail("Unsupported text.format");
  }
  if (params.metadata != null) {
    const metadata = object(params.metadata, "metadata");
    if (Object.values(metadata).some(value => typeof value !== "string"))
      fail("metadata values must be strings");
    result.labels = { ...metadata };
  }
  return result;
}

function convertContent(raw: unknown): InteractionContent {
  const part = object(raw, "content part");
  if (part.type === "input_text" || part.type === "output_text")
    return { type: "text", text: string(part.text, "text") };
  if (part.type === "refusal") return { type: "text", text: string(part.refusal, "refusal") };
  if (part.type === "input_image") {
    if (part.file_id != null) fail("OpenAI file_id cannot be resolved by Interactions");
    if (part.detail != null && part.detail !== "auto")
      fail("Image detail has no equivalent in Interactions V1");
    return media("image", part.image_url);
  }
  if (part.type === "input_file") {
    if (part.file_id != null) fail("OpenAI file_id cannot be resolved by Interactions");
    if (part.file_url != null && part.file_data != null) fail("Use either file_url or file_data");
    if (part.file_url != null) return media("document", part.file_url);
    const data = nonempty(part.file_data, "file_data");
    if (data.startsWith("data:")) return media("document", data);
    const mime = part.filename?.endsWith(".pdf")
      ? "application/pdf"
      : part.filename?.endsWith(".csv")
        ? "text/csv"
        : undefined;
    if (!mime) fail("file_data requires a MIME data URL or .pdf/.csv filename");
    return { type: "document", data, mime_type: mime };
  }
  if (part.type === "input_audio") {
    const audio = object(part.input_audio, "input_audio");
    if (!["mp3", "wav"].includes(audio.format)) fail("Unsupported input_audio.format");
    return {
      type: "audio",
      data: nonempty(audio.data, "input_audio.data"),
      mime_type: audio.format === "mp3" ? "audio/mp3" : "audio/wav",
    };
  }
  return fail(`Unsupported Responses content part: ${part.type}`);
}

function media(type: string, raw: unknown): InteractionContent {
  const value = nonempty(raw, "media URL");
  if (!value.startsWith("data:")) return { type, uri: value };
  const match = /^data:([^;,]+);base64,([\s\S]+)$/.exec(value);
  if (!match) fail("Expected a base64 media data URL");
  return { type, mime_type: match[1], data: match[2] };
}
function parts(value: unknown): Record<string, any>[] {
  if (!Array.isArray(value)) return fail("Expected an array of content/items");
  return value.map(part => object(part, "part"));
}
function object(value: unknown, name: string): Record<string, any> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    return fail(`${name} must be an object`);
  return value as Record<string, any>;
}
function string(value: unknown, name: string): string {
  if (typeof value !== "string") return fail(`${name} must be a string`);
  return value;
}
function nonempty(value: unknown, name: string): string {
  const result = string(value, name);
  if (!result.trim()) fail(`${name} must not be empty`);
  return result;
}
function parseArguments(value: unknown): Record<string, unknown> {
  try {
    return object(JSON.parse(string(value, "arguments")), "arguments");
  } catch {
    return fail("Function arguments must be a valid JSON object");
  }
}
function fail(message: string): never {
  throw new InteractionsRequestError(message);
}
