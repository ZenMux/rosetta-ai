import OpenAI from "openai";
import {
  ResponsesToInteractionsConverter,
  convertInteractionResponseUsage,
  type Interaction,
  type InteractionEvent,
  type InteractionUsage,
} from "../../index";

const usage: InteractionUsage = {
  total_input_tokens: 7,
  total_output_tokens: 20,
  total_thought_tokens: 22,
  total_tokens: 49,
  total_cached_tokens: 3,
  total_tool_use_tokens: 10,
  input_tokens_by_modality: [
    { modality: "text", tokens: 4 },
    { modality: "video", tokens: 3 },
  ],
  output_tokens_by_modality: [{ modality: "audio", tokens: 20 }],
  cached_tokens_by_modality: [{ modality: "text", tokens: 3 }],
  tool_use_tokens_by_modality: [{ modality: "text", tokens: 10 }],
  grounding_tool_count: [{ type: "google_search", count: 2 }],
  future_metric: { count: 9 },
};
const response: Interaction = {
  id: "interaction_1",
  model: "omni",
  status: "completed",
  steps: [{ type: "model_output", content: [{ type: "text", text: "你好" }] }],
  usage,
};
const params = (extra: Record<string, unknown> = {}) =>
  ({ model: "omni", input: "hello", ...extra }) as OpenAI.Responses.ResponseCreateParams;
function converter(extra: Record<string, unknown> = {}, media = false) {
  const value = new ResponsesToInteractionsConverter({ allowUnmappedContent: media });
  value.convertRequest(params(extra));
  return value;
}
function events(): InteractionEvent[] {
  return [
    {
      event_type: "interaction.created",
      interaction: { id: "interaction_1", status: "in_progress", model: "omni" },
    },
    { event_type: "step.start", index: 0, step: { type: "model_output" } },
    { event_type: "step.delta", event_id: "a", index: 0, delta: { type: "text", text: "你" } },
    { event_type: "step.delta", event_id: "b", index: 0, delta: { type: "text", text: "好" } },
    { event_type: "step.stop", index: 0, usage: { total_input_tokens: 7 } },
    { event_type: "interaction.completed", interaction: response },
  ];
}
function convertAll(source: InteractionEvent[], value = converter()) {
  const stream = source.flatMap(event => value.convertStreamEvent(structuredClone(event)));
  stream.push(...value.finishStream());
  return { stream, final: (stream.at(-1) as any).response, value };
}

describe("Responses → Interactions request", () => {
  test.each([undefined, true, false, null])("preserves store intent %s", store => {
    const value = converter({ store });
    expect(value.convertRequest(params({ store })).store).toBe(store ?? true);
  });
  test("maps ordered history, instructions, flat tools and call_id independently of item.id", () => {
    const result = converter().convertRequest(
      params({
        instructions: "first",
        input: [
          { role: "developer", content: "second" },
          { role: "user", content: [{ type: "input_text", text: "question" }] },
          {
            type: "function_call",
            id: "fc_different",
            call_id: "call_1",
            name: "weather",
            arguments: '{"city":"杭州"}',
          },
          {
            type: "function_call_output",
            call_id: "call_1",
            output: [{ type: "input_text", text: "sunny" }],
          },
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "answer" }],
          },
        ],
        tools: [{ type: "function", name: "weather", parameters: { type: "object" } }],
        tool_choice: {
          type: "allowed_tools",
          mode: "required",
          tools: [{ type: "function", name: "weather" }],
        },
        reasoning: { effort: "high", summary: "auto" },
        max_output_tokens: 1024,
        text: { format: { type: "json_schema", name: "answer", schema: { type: "object" } } },
      })
    );
    expect(result.system_instruction).toBe("first\nsecond");
    expect((result.input as any[]).map(item => item.type)).toEqual([
      "user_input",
      "function_call",
      "function_result",
      "model_output",
    ]);
    expect(result.input[1]).toMatchObject({ id: "call_1", arguments: { city: "杭州" } });
    expect(result.input[2]).toMatchObject({
      call_id: "call_1",
      result: [{ type: "text", text: "sunny" }],
    });
    expect(result.generation_config).toEqual({
      max_output_tokens: 1024,
      thinking_level: "high",
      thinking_summaries: "auto",
      tool_choice: { allowed_tools: { mode: "any", tools: ["weather"] } },
    });
    expect(result.response_format).toEqual({
      type: "text",
      mime_type: "application/json",
      schema: { type: "object" },
    });
  });
  test("maps mixed image/audio/file input in order without mutation", () => {
    const request = params({
      input: [
        {
          role: "user",
          content: [
            { type: "input_image", image_url: "data:image/png;base64,YQ==" },
            { type: "input_audio", input_audio: { data: "Yg==", format: "wav" } },
            { type: "input_file", file_url: "https://example.com/file.pdf" },
            { type: "input_text", text: "inspect" },
          ],
        },
      ],
    });
    const before = structuredClone(request);
    expect(converter().convertRequest(request).input[0]).toEqual({
      type: "user_input",
      content: [
        { type: "image", mime_type: "image/png", data: "YQ==" },
        { type: "audio", mime_type: "audio/wav", data: "Yg==" },
        { type: "document", uri: "https://example.com/file.pdf" },
        { type: "text", text: "inspect" },
      ],
    });
    expect(request).toEqual(before);
  });
  test.each([
    { temperature: 0 },
    { top_p: 1 },
    { background: true },
    { previous_response_id: "resp_1" },
    { conversation: "conv_1" },
    { prompt: { id: "prompt_1" } },
    { parallel_tool_calls: false },
    { reasoning: { effort: "none" } },
    { reasoning: { summary: "detailed" } },
    { include: ["web_search_call.action.sources"] },
    { truncation: "auto" },
    { max_tool_calls: 1 },
    { tools: [{ type: "computer" }] },
    { unknown_option: 1 },
    { text: { format: { type: "json_schema", name: "x", schema: {}, strict: true } } },
    { tools: [{ type: "function", name: "fn", strict: true }] },
    { tools: [{ type: "web_search", filters: { allowed_domains: ["example.com"] } }] },
    { tool_choice: { type: "function", name: "missing" } },
    { input: [{ type: "item_reference", id: "msg_1" }] },
    { input: [{ role: "user", content: [{ type: "input_image", file_id: "file_1" }] }] },
    { input: [{ type: "function_call_output", call_id: "missing", output: "result" }] },
    { input: [{ type: "function_call", call_id: "c", name: "f", arguments: "[]" }] },
    { input: [{ type: "reasoning", summary: [], encrypted_content: "foreign" }] },
  ])("rejects unsupported semantics: %j", input => {
    expect(() => converter().convertRequest(params(input))).toThrow();
  });
});

describe("Responses usage", () => {
  test("preserves all raw fields and counts thinking once", () => {
    const result = convertInteractionResponseUsage(usage)!;
    expect(result).toEqual({
      input_tokens: 7,
      output_tokens: 42,
      total_tokens: 49,
      input_tokens_details: { cached_tokens: 3 },
      output_tokens_details: { reasoning_tokens: 22 },
      interactions: usage,
    });
    expect(result.interactions).not.toBe(usage);
  });
  test("distinguishes absent usage, partial counts and actual zero", () => {
    expect(convertInteractionResponseUsage(undefined)).toBeNull();
    expect(convertInteractionResponseUsage({ total_input_tokens: 0 })).toEqual({
      input_tokens: 0,
      interactions: { total_input_tokens: 0 },
    });
    expect(
      convertInteractionResponseUsage({ total_output_tokens: 4 })?.output_tokens
    ).toBeUndefined();
  });
});

describe("Interactions → Responses", () => {
  test("JSON and streaming produce identical text and same-source usage", () => {
    const json = converter().convertResponse(response);
    const { stream, final } = convertAll(events());
    expect(final.output_text).toBe(json.output_text);
    expect(final.usage).toEqual(json.usage);
    expect(stream[0].type).toBe("response.created");
    expect(stream.map(event => event.sequence_number)).toEqual(stream.map((_, i) => i));
    expect(stream.filter(e => e.type === "response.output_item.done")).toHaveLength(1);
    expect(final.output[0].content[0].text).toBe("你好");
  });
  test("does not double count duplicate IDs or cumulative usage; waits for late usage", () => {
    const source = events();
    source.splice(3, 0, structuredClone(source[2]));
    source.push({
      event_type: "metadata",
      metadata: { total_usage: { ...usage, total_cached_tokens: 4 } },
    });
    const { final } = convertAll(source);
    expect(final.output_text).toBe("你好");
    expect(final.usage).toMatchObject({
      input_tokens: 7,
      output_tokens: 42,
      input_tokens_details: { cached_tokens: 4 },
    });
  });
  test("retains media, thought and signatures through the native step interface", () => {
    const { final, value } = convertAll(
      [
        { event_type: "step.start", index: 0, step: { type: "thought" } },
        {
          event_type: "step.delta",
          index: 0,
          delta: { type: "thought_summary", content: { type: "text", text: "summary" } },
        },
        {
          event_type: "step.delta",
          index: 0,
          delta: { type: "thought_signature", signature: "test-signature" },
        },
        { event_type: "step.stop", index: 0 },
        { event_type: "step.start", index: 1, step: { type: "model_output" } },
        {
          event_type: "step.delta",
          index: 1,
          delta: { type: "video", data: "YWJj", mime_type: "video/mp4" },
        },
        { event_type: "step.stop", index: 1 },
        {
          event_type: "interaction.completed",
          interaction: { id: "", status: "completed", usage },
        },
      ],
      converter({}, true)
    );
    expect(final.output.map((item: any) => item.type)).toEqual(["reasoning", "message"]);
    expect(final.output[1].content).toEqual([]);
    expect(value.getOutputStepEntries().map(entry => entry.step)).toEqual([
      {
        type: "thought",
        summary: [{ type: "text", text: "summary" }],
        signature: "test-signature",
      },
      { type: "model_output", content: [{ type: "video", data: "YWJj", mime_type: "video/mp4" }] },
    ]);
    expect(final.id).toMatch(/^resp_/);
  });
  test("tool arguments remain one JSON string with stable call identity", () => {
    const { stream, final } = convertAll([
      {
        event_type: "step.start",
        index: 0,
        step: { type: "function_call", id: "call_1", name: "weather" },
      },
      {
        event_type: "step.delta",
        index: 0,
        delta: { type: "arguments_delta", arguments: '{"city":' },
      },
      {
        event_type: "step.delta",
        index: 0,
        delta: { type: "arguments_delta", arguments: '"杭州"}' },
      },
      { event_type: "step.stop", index: 0 },
      {
        event_type: "interaction.completed",
        interaction: { id: "r", status: "requires_action", usage },
      },
    ]);
    expect(final.output[0]).toMatchObject({
      type: "function_call",
      call_id: "call_1",
      arguments: '{"city":"杭州"}',
    });
    expect(final.output[0].id).not.toBe("call_1");
    expect(stream.filter(e => e.type === "response.function_call_arguments.done")).toHaveLength(1);
  });
  test("incomplete does not invent max_output_tokens as the cause", () => {
    const final = converter().convertResponse({ ...response, status: "incomplete" });
    expect(final.status).toBe("incomplete");
    expect(final.incomplete_details).toBeNull();
  });
  test.each(
    [
      [{ event_type: "step.delta", index: 0, delta: { type: "text", text: "bad" } }],
      [{ event_type: "step.start", index: 0, step: { type: "model_output" } }],
      [{ event_type: "interaction.completed", interaction: { ...response, status: "cancelled" } }],
      [{ event_type: "error", error: { code: "invalid_request", message: "bad" } }],
      [...events(), { event_type: "error", error: { message: "late failure" } }],
      [...events(), events().at(-1)!],
    ].map(source => ({ source }))
  )("rejects truncation, errors or malformed lifecycle", ({ source }) => {
    expect(() => convertAll(source as InteractionEvent[])).toThrow();
  });
  test("standard-only conversion refuses to drop media or signatures", () => {
    expect(() =>
      converter().convertResponse({
        ...response,
        steps: [{ type: "model_output", content: [{ type: "image", data: "YQ==" }] }],
      })
    ).toThrow(/extension/);
    expect(() =>
      converter().convertResponse({
        ...response,
        steps: [{ type: "thought", signature: "signature" }],
      })
    ).toThrow(/extension/);
  });
  test.each([{ reasoning: { budget_tokens: 1024 } }, { reasoning: { enabled: true } }])(
    "rejects unsupported reasoning controls %j",
    extra => {
      expect(() => converter(extra)).toThrow(/reasoning/);
    }
  );
  test("echoes effective native configuration after overrides", () => {
    const value = converter({
      reasoning: { effort: "high" },
      instructions: "old",
      tools: [{ type: "function", name: "old", parameters: {} }],
    });
    value.setEffectiveRequest({
      model: "omni",
      input: [],
      store: false,
      system_instruction: "new",
      labels: { purpose: "test" },
      generation_config: { max_output_tokens: 32 },
      tools: [{ type: "function", name: "new", parameters: {} }],
      response_format: { type: "video", mime_type: "video/mp4" },
    });
    const result = value.convertResponse(response);
    expect(result).toMatchObject({
      instructions: "new",
      reasoning: null,
      store: false,
      max_output_tokens: 32,
      tools: [{ name: "new" }],
      metadata: { purpose: "test" },
    });
    expect(result.text).toBeUndefined();
  });
  test.each([
    { type: "model_output", content: [{ type: "text", text: "different" }] },
    { type: "thought", summary: [{ type: "text", text: "different" }] },
  ])("rejects conflicting final text/type %j", step => {
    const source = events();
    source[source.length - 1] = {
      event_type: "interaction.completed",
      interaction: { ...response, steps: [step] },
    };
    expect(() => convertAll(source)).toThrow(/disagree/);
  });
  test.each(["signature", "media"])("rejects conflicting final %s", kind => {
    const step =
      kind === "signature"
        ? { type: "thought", signature: "first" }
        : { type: "model_output", content: [{ type: "video", data: "first" }] };
    const final =
      kind === "signature"
        ? { ...step, signature: "second" }
        : { ...step, content: [{ type: "video", data: "second" }] };
    expect(() =>
      convertAll(
        [
          { event_type: "step.start", index: 0, step },
          { event_type: "step.stop", index: 0 },
          { event_type: "interaction.completed", interaction: { ...response, steps: [final] } },
        ],
        converter({}, true)
      )
    ).toThrow(/disagree/);
  });
  test.each([false, true])(
    "validates final arguments independently of property order: conflict=%s",
    conflict => {
      const value = converter();
      const source: InteractionEvent[] = [
        {
          event_type: "step.start",
          index: 0,
          step: { type: "function_call", id: "c", name: "f", arguments: {} },
        },
        {
          event_type: "step.delta",
          index: 0,
          delta: { type: "arguments_delta", arguments: '{"a":1,"b":2}' },
        },
        { event_type: "step.stop", index: 0 },
        {
          event_type: "interaction.completed",
          interaction: {
            ...response,
            steps: [
              {
                type: "function_call",
                id: "c",
                name: "f",
                arguments: { b: 2, a: conflict ? 3 : 1 },
              },
            ],
          },
        },
      ];
      if (conflict) expect(() => convertAll(source, value)).toThrow(/arguments disagree/);
      else expect(convertAll(source, value).final.output[0].arguments).toBe('{"a":1,"b":2}');
    }
  );
  test("OpenAI SDK streams and finalResponse consume the generated lifecycle", async () => {
    const { stream } = convertAll(events());
    const client = new OpenAI({
      apiKey: "fixture",
      fetch: async () =>
        new Response(
          stream.map(event => `data: ${JSON.stringify(event)}\n\n`).join("") + "data: [DONE]\n\n",
          { headers: { "content-type": "text/event-stream" } }
        ),
    });
    const result = await client.responses.stream({ model: "omni", input: "hello" }).finalResponse();
    expect(result.output_text).toBe("你好");
    expect(result.usage).toMatchObject({ input_tokens: 7, output_tokens: 42, total_tokens: 49 });
  });
});
