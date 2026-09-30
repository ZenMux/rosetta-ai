import { MessagesToInteractionsConverter } from "../interactions";
import { convertInteractionMessageUsage } from "../../interactions/usage";
import type { Interaction, InteractionEvent, InteractionUsage } from "../../interactions/types";

const params = {
  model: "omni",
  max_tokens: 1024,
  messages: [{ role: "user" as const, content: "hello" }],
};
const usage: InteractionUsage = {
  total_input_tokens: 100,
  total_cached_tokens: 30,
  total_output_tokens: 20,
  total_thought_tokens: 7,
  total_tokens: 127,
  total_tool_use_tokens: 12,
  future_usage: { count: 9 },
  input_tokens_by_modality: [
    { modality: "image", tokens: 90 },
    { modality: "text", tokens: 10 },
  ],
  output_tokens_by_modality: [{ modality: "text", tokens: 20 }],
  cached_tokens_by_modality: [{ modality: "image", tokens: 30 }],
  grounding_tool_count: [{ type: "google_search", count: 2 }],
};
function response(): Interaction {
  return {
    id: "native_1",
    model: "omni",
    status: "completed",
    usage: structuredClone(usage),
    steps: [
      {
        type: "thought",
        summary: [{ type: "text", text: "想一想" }],
        signature: "native-signature",
      },
      { type: "model_output", content: [{ type: "text", text: "hello" }] },
    ],
  };
}
function converter() {
  const c = new MessagesToInteractionsConverter({ allowUnmappedContent: true });
  c.convertRequest(params);
  return c;
}
function events(raw = response()): InteractionEvent[] {
  const result: InteractionEvent[] = [
    {
      event_type: "interaction.created",
      interaction: { id: raw.id, status: "in_progress", model: raw.model },
    },
  ];
  raw.steps?.forEach((step, index) => {
    result.push({
      event_type: "step.start",
      index,
      step: { type: step.type, ...(step.id && { id: step.id, name: step.name }) },
    });
    if (step.type === "thought") {
      result.push({
        event_type: "step.delta",
        index,
        delta: { type: "thought_summary", content: { type: "text", text: "想一想" } },
      });
      result.push({
        event_type: "step.delta",
        index,
        delta: { type: "thought_signature", signature: step.signature },
      });
    } else if (step.type === "function_call") {
      result.push({
        event_type: "step.delta",
        index,
        delta: { type: "arguments_delta", arguments: '{"city":' },
      });
      result.push({
        event_type: "step.delta",
        index,
        delta: { type: "arguments_delta", arguments: '"杭州"}' },
      });
    } else
      for (const part of step.content ?? [])
        result.push({ event_type: "step.delta", index, delta: part });
    result.push({ event_type: "step.stop", index, step_usage: { total_input_tokens: 99999 } });
  });
  result.push({ event_type: "interaction.completed", interaction: raw });
  return result;
}

describe("Messages Interactions usage", () => {
  test("preserves every native field, separates cached input, and adds thought once", () => {
    const actual = convertInteractionMessageUsage(usage)!;
    expect(actual).toEqual({
      input_tokens: 70,
      cache_read_input_tokens: 30,
      output_tokens: 27,
      interactions: usage,
    });
    expect(actual.interactions).not.toBe(usage);
    expect(actual.input_tokens! + actual.cache_read_input_tokens! + actual.output_tokens!).toBe(
      127
    );
    expect(actual).not.toHaveProperty("cache_creation_input_tokens");
  });
  test.each([
    [undefined, undefined],
    [{}, { interactions: {} }],
    [
      { total_input_tokens: 0, total_output_tokens: 0, total_thought_tokens: 0, total_tokens: 0 },
      {
        input_tokens: 0,
        output_tokens: 0,
        interactions: {
          total_input_tokens: 0,
          total_output_tokens: 0,
          total_thought_tokens: 0,
          total_tokens: 0,
        },
      },
    ],
    [{ total_output_tokens: 20 }, { interactions: { total_output_tokens: 20 } }],
    [{ total_thought_tokens: 7 }, { interactions: { total_thought_tokens: 7 } }],
    [
      { total_cached_tokens: 5 },
      { cache_read_input_tokens: 5, interactions: { total_cached_tokens: 5 } },
    ],
  ])("does not invent missing counters %#", (raw, expected) =>
    expect(convertInteractionMessageUsage(raw as InteractionUsage)).toEqual(expected)
  );
  test("preserves contradictory counters for diagnostics", () => {
    const raw = {
      total_input_tokens: 2,
      total_cached_tokens: 4,
      total_output_tokens: 1,
      total_thought_tokens: 1,
      total_tokens: 999,
    };
    expect(convertInteractionMessageUsage(raw)).toEqual({
      input_tokens: -2,
      cache_read_input_tokens: 4,
      output_tokens: 2,
      interactions: raw,
    });
  });
  test.each([
    [10, 115, 259, 384, 374],
    [16, 5793, 241, 6050, 6034],
  ])("maps the user-provided billing sample %#", (input, output, thought, total, expected) => {
    const raw = {
      total_input_tokens: input,
      total_output_tokens: output,
      total_thought_tokens: thought,
      total_tokens: total,
    };
    expect(convertInteractionMessageUsage(raw)).toEqual({
      input_tokens: input,
      output_tokens: expected,
      interactions: raw,
    });
  });
});

describe("Messages Interactions requests", () => {
  test("maps text, system, schema, adaptive thinking and limits directly", () => {
    const c = converter();
    const result = c.convertRequest({
      ...params,
      system: [
        { type: "text", text: "one" },
        { type: "text", text: "two" },
      ],
      thinking: { type: "adaptive" },
      output_config: { effort: "low", format: { type: "json_schema", schema: { type: "object" } } },
      stop_sequences: ["END"],
    });
    expect(result).toEqual({
      model: "omni",
      input: [{ type: "user_input", content: [{ type: "text", text: "hello" }] }],
      store: false,
      stream: false,
      system_instruction: "one\ntwo",
      generation_config: {
        max_output_tokens: 1024,
        stop_sequences: ["END"],
        thinking_level: "low",
        thinking_summaries: "auto",
      },
      response_format: { type: "text", mime_type: "application/json", schema: { type: "object" } },
    });
  });
  test("preserves mixed media order and URL/base64 bytes", () => {
    const p = {
      ...params,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "before" },
            { type: "image", source: { type: "base64", media_type: "image/png", data: "YWJj" } },
            { type: "text", text: "after" },
            { type: "document", source: { type: "url", url: "https://example.org/file.pdf" } },
          ],
        },
      ],
    };
    expect(converter().convertRequest(p as any).input).toEqual([
      {
        type: "user_input",
        content: [
          { type: "text", text: "before" },
          { type: "image", mime_type: "image/png", data: "YWJj" },
          { type: "text", text: "after" },
          { type: "document", uri: "https://example.org/file.pdf" },
        ],
      },
    ]);
  });
  test("maps tool calls/results and names without duplicating history", () => {
    const c = converter();
    const result = c.convertRequest({
      ...params,
      tools: [{ name: "weather", input_schema: { type: "object" } }],
      tool_choice: { type: "tool", name: "weather" },
      messages: [
        ...params.messages,
        {
          role: "assistant",
          content: [
            { type: "text", text: "checking" },
            { type: "tool_use", id: "call_1", name: "weather", input: { city: "杭州" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "call_1", content: "sunny" },
            { type: "text", text: "continue" },
          ],
        },
      ],
    });
    expect(result.input).toEqual([
      { type: "user_input", content: [{ type: "text", text: "hello" }] },
      { type: "model_output", content: [{ type: "text", text: "checking" }] },
      { type: "function_call", id: "call_1", name: "weather", arguments: { city: "杭州" } },
      { type: "function_result", call_id: "call_1", name: "weather", result: "sunny" },
      { type: "user_input", content: [{ type: "text", text: "continue" }] },
    ]);
    expect(c.getInputStepRanges()).toEqual([
      { start: 0, end: 1 },
      { start: 1, end: 3 },
      { start: 3, end: 5 },
    ]);
  });
  test.each([
    { temperature: 0.2 },
    { top_p: 0.5 },
    { top_k: 2 },
    { service_tier: "auto" },
    { metadata: { user_id: "u" } },
    { cache_control: { type: "ephemeral" } },
    { thinking: { type: "enabled", budget_tokens: 1024 } },
    { thinking: { type: "disabled" } },
    { thinking: { type: "adaptive", budget_tokens: 1024 } },
    { output_config: { effort: "max" } },
    { output_config: { format: { type: "json_schema", schema: {}, strict: true } } },
    { tool_choice: { type: "auto", disable_parallel_tool_use: true } },
    { tools: [{ name: "f", input_schema: {}, strict: true }] },
    { tools: [{ type: "web_search_20250305", name: "web_search" }] },
    { max_tokens: 0 },
    {
      messages: [
        { role: "user", content: [{ type: "tool_result", tool_use_id: "missing", content: "x" }] },
      ],
    },
    {
      messages: [
        { role: "assistant", content: [{ type: "thinking", thinking: "x", signature: "foreign" }] },
      ],
    },
    {
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "x", cache_control: { type: "ephemeral" } }],
        },
      ],
    },
    { messages: [{ role: "system", content: "x" }] },
    {
      messages: [
        {
          role: "user",
          content: [{ type: "document", source: { type: "file", file_id: "foreign" } }],
        },
      ],
    },
  ])("rejects unsupported intent instead of dropping it %#", extra =>
    expect(() => converter().convertRequest({ ...params, ...extra } as any)).toThrow()
  );
});

describe("Messages Interactions JSON/SSE", () => {
  test("accepts empty stateless upstream IDs in JSON and SSE", () => {
    const raw = response();
    raw.id = "";
    expect(converter().convertResponse(raw).id).toMatch(/^msg_/);
    const c = converter();
    const output = events(raw).flatMap(e => c.convertStreamEvent(e));
    output.push(...c.finishStream());
    expect(output[0].type === "message_start" && output[0].message.id).toMatch(/^msg_/);
    expect(output.at(-1)?.type).toBe("message_stop");
  });
  test("does not invent a max_tokens reason for unspecified incomplete results", () => {
    const raw = response();
    raw.status = "incomplete";
    expect(converter().convertResponse(raw).stop_reason).toBeNull();
  });
  test("maps text, thinking, signature and usage in a JSON response", () => {
    const c = converter(),
      raw = response(),
      msg = c.convertResponse(raw);
    expect(msg.content).toEqual([
      { type: "thinking", thinking: "想一想", signature: "native-signature" },
      { type: "text", text: "hello", citations: null },
    ]);
    expect(msg.usage?.interactions).toEqual(raw.usage);
    expect(msg.stop_reason).toBe("end_turn");
  });
  test("emits each block once and terminal usage after late cumulative metadata", () => {
    const c = converter(),
      out = events().flatMap(e => c.convertStreamEvent(e));
    expect(out.some(e => e.type === "message_stop")).toBe(false);
    c.convertStreamEvent({
      event_type: "metadata",
      metadata: {
        total_usage: { total_cached_tokens: 0, total_output_tokens: 21, total_tokens: 128 },
      },
    });
    out.push(...c.finishStream());
    expect(out.filter(e => e.type === "content_block_start")).toHaveLength(2);
    expect(out.filter(e => e.type === "content_block_stop")).toHaveLength(2);
    expect(out.at(-1)?.type).toBe("message_stop");
    const terminal = out.find(e => e.type === "message_delta")!;
    expect(terminal.type === "message_delta" && terminal.usage).toEqual({
      input_tokens: 100,
      cache_read_input_tokens: 0,
      output_tokens: 28,
      interactions: {
        ...usage,
        total_cached_tokens: 0,
        total_output_tokens: 21,
        total_tokens: 128,
      },
    });
    expect(c.finishStream()).toEqual([]);
  });
  test("duplicate event IDs and per-step usage cannot inflate totals", () => {
    const c = converter();
    for (const e of events()) {
      e.event_id = String(Math.random());
      c.convertStreamEvent(e);
      expect(c.convertStreamEvent(e)).toEqual([]);
    }
    c.finishStream();
    expect(c.getUsage()).toEqual(usage);
  });
  test("maps function argument deltas and stop_reason", () => {
    const raw = response();
    raw.status = "requires_action";
    raw.steps!.push({
      type: "function_call",
      id: "call_1",
      name: "weather",
      arguments: { city: "杭州" },
    });
    const c = converter(),
      out = events(raw).flatMap(e => c.convertStreamEvent(e));
    out.push(...c.finishStream());
    expect(
      out
        .filter(e => e.type === "content_block_delta" && e.delta.type === "input_json_delta")
        .map(e =>
          e.type === "content_block_delta" && e.delta.type === "input_json_delta"
            ? e.delta.partial_json
            : ""
        )
        .join("")
    ).toBe('{"city":"杭州"}');
    const end = out.find(e => e.type === "message_delta");
    expect(end?.type === "message_delta" && end.delta.stop_reason).toBe("tool_use");
  });
  test("preserves media for the gateway and rejects it without extension support", () => {
    const raw = response();
    raw.steps!.push({
      type: "model_output",
      content: [{ type: "video", data: "YWJj", mime_type: "video/mp4", future: "x" }],
    });
    const c = converter();
    c.convertResponse(raw);
    expect(c.getOutputStepEntries().map(e => e.step)).toEqual(raw.steps);
    expect(() => new MessagesToInteractionsConverter().convertResponse(raw)).toThrow(/extension/);
  });
  test("missing usage stays absent without fake zero counters", () => {
    const raw = response();
    delete raw.usage;
    expect(converter().convertResponse(raw)).not.toHaveProperty("usage");
    const c = converter();
    const out = events(raw).flatMap(e => c.convertStreamEvent(e));
    out.push(...c.finishStream());
    const start = out[0];
    expect(start.type === "message_start" && start.message.usage).toEqual({});
    const end = out.find(e => e.type === "message_delta");
    expect(end?.type === "message_delta" && end.usage).toEqual({});
  });
  test.each([
    "missing-terminal",
    "error",
    "after-terminal",
    "mismatched-text",
    "mismatched-signature",
    "invalid-arguments",
  ])("does not complete invalid stream %s", scenario => {
    const c = converter(),
      ev = events();
    if (scenario === "missing-terminal") ev.pop();
    if (scenario === "error")
      ev.splice(2, 0, { event_type: "error", error: { code: 400, message: "bad" } });
    if (scenario === "after-terminal")
      ev.push({ event_type: "step.start", index: 9, step: { type: "model_output" } });
    if (scenario === "mismatched-text")
      ev.at(-1)!.interaction!.steps![1].content = [{ type: "text", text: "different" }];
    if (scenario === "mismatched-signature")
      ev.at(-1)!.interaction!.steps![0].signature = "different";
    if (scenario === "invalid-arguments") {
      const raw = response();
      raw.steps = [{ type: "function_call", id: "call_1", name: "f", arguments: {} }];
      ev.splice(0, ev.length, ...events(raw));
      ev.find(e => e.delta?.type === "arguments_delta")!.delta!.arguments = "broken";
    }
    expect(() => {
      ev.forEach(e => c.convertStreamEvent(e));
      c.finishStream();
    }).toThrow();
  });
});
