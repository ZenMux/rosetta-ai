import type OpenAI from "openai";
import {
  InteractionsRequestError,
  type InteractionContent,
  type InteractionCreateParams,
  type InteractionGenerationConfig,
  type InteractionStep,
} from "../interactions/types";

/** Validate parameter semantics before gateways filter unsupported model fields. */
export function validateChatCompletionInteractionsParameters(
  params: OpenAI.ChatCompletionCreateParams
): void {
  // These fields have no corresponding contract in the V1 generation config.
  // Reject explicit requests instead of silently changing sampling/format semantics.
  for (const key of [
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "prediction",
    "audio",
  ] as const) {
    if (params[key] != null) fail(`${key} has no equivalent in Interactions V1`);
  }
  if (params.n != null && params.n !== 1) fail("Interactions V1 supports only n=1");
  if (params.parallel_tool_calls === false)
    fail("Interactions V1 cannot enforce parallel_tool_calls=false");
  if (params.modalities?.some(modality => modality !== "text")) {
    fail("Non-text output requires an explicit Interactions media output contract");
  }

  if (
    params.reasoning_effort != null &&
    !["minimal", "low", "medium", "high"].includes(params.reasoning_effort)
  ) {
    fail(`reasoning_effort=${params.reasoning_effort} has no equivalent in Interactions V1`);
  }
}

export function convertChatCompletionRequest(
  params: OpenAI.ChatCompletionCreateParams,
  onMessage?: (index: number, start: number, end: number) => void
): InteractionCreateParams {
  validateChatCompletionInteractionsParameters(params);
  const input: InteractionStep[] = [];
  const system: string[] = [];
  const names = new Map<string, string>();
  if (!Array.isArray(params.messages)) fail("messages must be an array");
  for (const [messageIndex, message] of params.messages.entries()) {
    const start = input.length;
    if (!message || typeof message !== "object") fail("messages must contain objects");
    switch (message.role) {
      case "system":
      case "developer":
        system.push(textContent(message.content));
        break;
      case "user": {
        const content =
          typeof message.content === "string"
            ? [{ type: "text", text: message.content }]
            : message.content.map(convertContent);
        input.push({ type: "user_input", content });
        break;
      }
      case "assistant": {
        if (message.audio)
          fail("assistant.audio history requires an inline Interactions audio input");
        if (message.content != null) {
          const text = textContent(message.content);
          if (text) input.push({ type: "model_output", content: [{ type: "text", text }] });
        }
        for (const call of message.tool_calls ?? []) {
          if (call.type !== "function")
            fail("custom tool calls are not supported by Interactions V1");
          names.set(call.id, call.function.name);
          input.push({
            type: "function_call",
            id: call.id,
            name: call.function.name,
            arguments: parseArguments(call.function.arguments),
          });
        }
        if (message.function_call) {
          const id = `call_${input.length}`;
          names.set(message.function_call.name, id);
          input.push({
            type: "function_call",
            id,
            name: message.function_call.name,
            arguments: parseArguments(message.function_call.arguments),
          });
        }
        break;
      }
      case "tool":
        input.push({
          type: "function_result",
          call_id: message.tool_call_id,
          ...(names.has(message.tool_call_id) && { name: names.get(message.tool_call_id) }),
          result: textContent(message.content),
        });
        break;
      case "function": {
        const callId = names.get(message.name);
        if (!callId) fail(`No preceding function call for ${message.name}`);
        input.push({
          type: "function_result",
          call_id: callId,
          name: message.name,
          result: message.content ?? "",
        });
        break;
      }
      default:
        fail("Unsupported message role");
    }
    onMessage?.(messageIndex, start, input.length);
  }

  const config: InteractionGenerationConfig = {};
  const maxTokens = params.max_completion_tokens ?? params.max_tokens;
  if (maxTokens != null) config.max_output_tokens = maxTokens;
  if (params.seed != null) config.seed = params.seed;
  if (params.stop != null)
    config.stop_sequences = typeof params.stop === "string" ? [params.stop] : [...params.stop];
  if (params.reasoning_effort != null) {
    config.thinking_level =
      params.reasoning_effort as InteractionGenerationConfig["thinking_level"];
  }

  const tools: NonNullable<InteractionCreateParams["tools"]> = [];
  for (const tool of params.tools ?? []) {
    if (tool.type !== "function") fail("custom tools are not supported by Interactions V1");
    tools.push({
      type: "function",
      name: tool.function.name,
      description: tool.function.description,
      parameters: tool.function.parameters,
    });
  }
  if (!params.tools && params.functions) {
    tools.push(
      ...params.functions.map(fn => ({
        type: "function",
        name: fn.name,
        description: fn.description,
        parameters: fn.parameters,
      }))
    );
  }
  if (params.web_search_options != null) tools.push({ type: "google_search" });
  if (params.tool_choice != null) config.tool_choice = convertToolChoice(params.tool_choice);
  else if (params.function_call != null) {
    config.tool_choice =
      typeof params.function_call === "string"
        ? params.function_call
        : { allowed_tools: { mode: "any", tools: [params.function_call.name] } };
  }

  const result: InteractionCreateParams = {
    model: params.model,
    input,
    // CC defaults to false, unlike Interactions' stored-session workflow.
    store: params.store ?? false,
    stream: params.stream ?? false,
    ...(system.length > 0 && { system_instruction: system.join("\n") }),
    ...(Object.keys(config).length > 0 && { generation_config: config }),
    ...(tools.length > 0 && { tools }),
  };
  if (params.metadata != null) {
    if (Object.values(params.metadata).some(value => typeof value !== "string"))
      fail("metadata values must be strings");
    result.labels = { ...params.metadata } as Record<string, string>;
  }
  if (params.response_format?.type === "text")
    result.response_format = { type: "text", mime_type: "text/plain" };
  else if (params.response_format?.type === "json_object")
    result.response_format = { type: "text", mime_type: "application/json" };
  else if (params.response_format?.type === "json_schema") {
    result.response_format = {
      type: "text",
      mime_type: "application/json",
      schema: params.response_format.json_schema.schema,
    };
  }
  return result;
}

function convertToolChoice(
  choice: OpenAI.ChatCompletionToolChoiceOption
): InteractionGenerationConfig["tool_choice"] {
  if (typeof choice === "string") return choice === "required" ? "any" : choice;
  if (choice.type === "function")
    return { allowed_tools: { mode: "any", tools: [choice.function.name] } };
  if (choice.type === "allowed_tools") {
    return {
      allowed_tools: {
        mode: choice.allowed_tools.mode === "required" ? "any" : "auto",
        tools: choice.allowed_tools.tools.map(tool => {
          if (
            tool.type !== "function" ||
            typeof tool.function !== "object" ||
            tool.function == null ||
            !("name" in tool.function) ||
            typeof tool.function.name !== "string"
          )
            fail("allowed_tools must contain named functions");
          return tool.function.name;
        }),
      },
    };
  }
  return fail("Unsupported Interactions tool_choice");
}

function textContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content))
    return fail("Message content must be text or an array of text parts");
  return content
    .map(part => {
      if (part?.type === "text" && typeof part.text === "string") return part.text;
      if (part?.type === "refusal" && typeof part.refusal === "string") return part.refusal;
      return fail("Expected a text message part");
    })
    .join("");
}

function convertContent(part: OpenAI.ChatCompletionContentPart): InteractionContent {
  if (!part) return fail("Message content parts must be objects");
  if (part.type === "text") return { type: "text", text: part.text };
  if (part.type === "image_url") return mediaContent("image", part.image_url.url);
  if (part.type === "input_audio")
    return {
      type: "audio",
      data: part.input_audio.data,
      mime_type: part.input_audio.format === "mp3" ? "audio/mp3" : "audio/wav",
    };
  if (part.type === "file") {
    if (part.file.file_id)
      return fail("OpenAI file_id cannot be resolved by Interactions; use file_data");
    if (!part.file.file_data) return fail("file.file_data is required");
    const mime = part.file.filename?.endsWith(".csv")
      ? "text/csv"
      : part.file.filename?.endsWith(".pdf")
        ? "application/pdf"
        : undefined;
    if (!part.file.file_data.startsWith("data:") && !mime)
      return fail("file_data requires a MIME data URL or a .pdf/.csv filename");
    return part.file.file_data.startsWith("data:")
      ? mediaContent("document", part.file.file_data)
      : { type: "document", data: part.file.file_data, mime_type: mime };
  }
  return fail("Unsupported CC content part");
}

function mediaContent(type: string, value: string): InteractionContent {
  if (typeof value !== "string") return fail("Media URL must be a string");
  if (!value.startsWith("data:")) return { type, uri: value };
  const match = /^data:([^;,]+);base64,([\s\S]*)$/.exec(value);
  if (!match) return fail("Expected a base64 media data URL");
  return { type, mime_type: match[1], data: match[2] };
}

function parseArguments(value: string): Record<string, unknown> {
  try {
    const result: unknown = JSON.parse(value);
    if (!result || typeof result !== "object" || Array.isArray(result))
      return fail("Function arguments must be a JSON object");
    return result as Record<string, unknown>;
  } catch (error) {
    if (error instanceof InteractionsRequestError) throw error;
    return fail("Function arguments must be valid JSON");
  }
}

function fail(message: string): never {
  throw new InteractionsRequestError(message);
}
