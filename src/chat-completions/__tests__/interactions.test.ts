import type OpenAI from "openai";
import { ChatCompletionToInteractionsConverter } from "../interactions";
import { validateChatCompletionInteractionsParameters } from "../interactions-request";
import { convertInteractionUsage } from "../../interactions/usage";
import {
  Interaction,
  InteractionEvent,
  InteractionUsage,
  InteractionsRequestError,
} from "../../interactions/types";

// Official V1 example: visible output and thought tokens are separate.
const usage: InteractionUsage = {
  total_input_tokens: 7,
  total_output_tokens: 20,
  total_thought_tokens: 22,
  total_tool_use_tokens: 0,
  total_cached_tokens: 0,
  total_tokens: 49,
};
const response: Interaction = {
  id: "interaction_1",
  model: "gemini-omni",
  created: "2026-09-22T00:00:00Z",
  status: "completed",
  steps: [{ type: "model_output", content: [{ type: "text", text: "你好🙂" }] }],
  usage,
};
const created: InteractionEvent = {
  event_type: "interaction.created",
  interaction: { ...response, status: "in_progress", steps: undefined, usage: undefined },
};
const completed: InteractionEvent = {
  event_type: "interaction.completed",
  interaction: { ...response, steps: undefined },
};
const request = (extra: object = {}) =>
  ({
    model: "gemini-omni",
    messages: [{ role: "user", content: "hello" }],
    ...extra,
  }) as OpenAI.ChatCompletionCreateParams;

describe("CC to Interactions V1", () => {
  test("converts full history directly to steps, preserving parallel call IDs/results and source store default", () => {
    const converter = new ChatCompletionToInteractionsConverter();
    const result = converter.convertRequest(
      request({
        messages: [
          { role: "system", content: "system" },
          { role: "developer", content: "developer" },
          { role: "user", content: "weather" },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "a",
                type: "function",
                function: { name: "weather", arguments: '{"city":"北京"}' },
              },
              {
                id: "b",
                type: "function",
                function: { name: "weather", arguments: '{"city":"上海"}' },
              },
            ],
          },
          { role: "tool", tool_call_id: "b", content: "rain" },
          { role: "tool", tool_call_id: "a", content: "sun" },
        ],
      })
    );
    expect(result).toEqual({
      model: "gemini-omni",
      stream: false,
      store: false,
      system_instruction: "system\ndeveloper",
      input: [
        { type: "user_input", content: [{ type: "text", text: "weather" }] },
        { type: "function_call", id: "a", name: "weather", arguments: { city: "北京" } },
        { type: "function_call", id: "b", name: "weather", arguments: { city: "上海" } },
        { type: "function_result", call_id: "b", name: "weather", result: "rain" },
        { type: "function_result", call_id: "a", name: "weather", result: "sun" },
      ],
    });
    expect(converter.getMessageStepRanges()[3]).toEqual({ start: 1, end: 3 });
  });

  test("maps generation fields, tool selection and JSON schema without mutating input", () => {
    const params = request({
      store: true,
      max_tokens: 30,
      max_completion_tokens: 50,
      seed: 0,
      stop: ["STOP"],
      reasoning_effort: "high",
      tools: [{ type: "function", function: { name: "f", parameters: { type: "object" } } }],
      tool_choice: { type: "function", function: { name: "f" } },
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "answer",
          schema: { type: "object", properties: { answer: { type: "string" } } },
        },
      },
    });
    const original = structuredClone(params);
    const result = new ChatCompletionToInteractionsConverter().convertRequest(params);
    expect(result.generation_config).toEqual({
      max_output_tokens: 50,
      seed: 0,
      stop_sequences: ["STOP"],
      thinking_level: "high",
      tool_choice: { allowed_tools: { mode: "any", tools: ["f"] } },
    });
    expect(result.response_format).toEqual({
      type: "text",
      mime_type: "application/json",
      schema: { type: "object", properties: { answer: { type: "string" } } },
    });
    expect(result.store).toBe(true);
    expect(params).toEqual(original);
  });

  test("preserves inline media bytes, remote URI and MIME on standard CC input", () => {
    const result = new ChatCompletionToInteractionsConverter().convertRequest(
      request({
        messages: [
          {
            role: "user",
            content: [
              { type: "image_url", image_url: { url: "data:image/png;base64,YWJj" } },
              { type: "image_url", image_url: { url: "https://example.com/image.png" } },
              { type: "input_audio", input_audio: { data: "YWJj", format: "mp3" } },
              { type: "file", file: { file_data: "data:application/pdf;base64,YWJj" } },
            ],
          },
        ],
      })
    );
    expect(result.input).toEqual([
      {
        type: "user_input",
        content: [
          { type: "image", data: "YWJj", mime_type: "image/png" },
          { type: "image", uri: "https://example.com/image.png" },
          { type: "audio", data: "YWJj", mime_type: "audio/mp3" },
          { type: "document", data: "YWJj", mime_type: "application/pdf" },
        ],
      },
    ]);
  });

  test.each([
    { temperature: 0 },
    { top_p: 0.8 },
    { n: 2 },
    { reasoning_effort: "none" },
    { parallel_tool_calls: false },
  ])("rejects unsupported semantics before upstream: %j", extra => {
    expect(() => validateChatCompletionInteractionsParameters(request(extra))).toThrow(
      InteractionsRequestError
    );
    expect(() =>
      new ChatCompletionToInteractionsConverter().convertRequest(request(extra))
    ).toThrow(InteractionsRequestError);
  });

  test("parameter guard accepts nullish/default semantics without converting history", () => {
    const params = request({
      messages: [{ role: "assistant", interactions: { steps: [] } }],
      temperature: null,
      n: 1,
      parallel_tool_calls: true,
      reasoning_effort: "high",
    });
    const before = structuredClone(params);
    validateChatCompletionInteractionsParameters(params);
    expect(params).toEqual(before);
  });

  test.each(["{broken", "[]", "null", '"string"'])(
    "rejects invalid function arguments %s",
    argumentsText => {
      expect(() =>
        new ChatCompletionToInteractionsConverter().convertRequest(
          request({
            messages: [
              {
                role: "assistant",
                tool_calls: [
                  { id: "a", type: "function", function: { name: "f", arguments: argumentsText } },
                ],
              },
            ],
          })
        )
      ).toThrow(InteractionsRequestError);
    }
  );

  test("converts nonstream text and official thought-inclusive usage exactly", () => {
    const result = new ChatCompletionToInteractionsConverter().convertResponse(response);
    expect(result.choices[0].message.content).toBe("你好🙂");
    expect(result.usage).toMatchObject({
      prompt_tokens: 7,
      completion_tokens: 42,
      total_tokens: 49,
      completion_tokens_details: { reasoning_tokens: 22 },
    });
    expect(result.usage?.interactions).toEqual(usage);
  });

  test("does not double count tool tokens in the official function-call example", () => {
    const result = convertInteractionUsage({
      total_input_tokens: 100,
      total_output_tokens: 25,
      total_thought_tokens: 0,
      total_tokens: 125,
      total_tool_use_tokens: 50,
    });
    expect(result).toMatchObject({
      prompt_tokens: 100,
      completion_tokens: 25,
      total_tokens: 125,
      interactions: { total_tool_use_tokens: 50 },
    });
  });

  test("preserves all modality, grounding and future usage fields exactly, including zero", () => {
    const raw: InteractionUsage = {
      ...usage,
      total_cached_tokens: 3,
      input_tokens_by_modality: [
        { modality: "text", tokens: 2 },
        { modality: "audio", tokens: 3 },
        { modality: "audio", tokens: 2 },
      ],
      output_tokens_by_modality: [{ modality: "audio", tokens: 20 }],
      cached_tokens_by_modality: [{ modality: "audio", tokens: 3 }],
      tool_use_tokens_by_modality: [{ modality: "text", tokens: 0 }],
      grounding_tool_count: [
        { type: "google_search", count: 2 },
        { type: "google_maps", count: 1 },
      ],
      future_metric: { amount: 99 },
    };
    const converted = convertInteractionUsage(raw)!;
    expect(converted.interactions).toEqual(raw);
    expect(converted.prompt_tokens_details).toEqual({ cached_tokens: 3, audio_tokens: 5 });
    expect(converted.completion_tokens_details).toEqual({ reasoning_tokens: 22, audio_tokens: 20 });
    raw.input_tokens_by_modality![0].tokens = 999;
    expect(converted.interactions.input_tokens_by_modality![0].tokens).toBe(2);
  });

  test("does not invent absent counts or infer missing thought tokens", () => {
    expect(convertInteractionUsage(undefined)).toBeUndefined();
    const partial = convertInteractionUsage({ total_output_tokens: 10, total_input_tokens: 2 })!;
    expect(partial.prompt_tokens).toBe(2);
    expect(partial.completion_tokens).toBeUndefined();
    expect(partial.total_tokens).toBeUndefined();
    expect(partial.prompt_tokens_details).toBeUndefined();
    expect(partial.completion_tokens_details).toBeUndefined();
  });

  test("matches nonstream output/usage with duplicate events and partial final resource", () => {
    const converter = new ChatCompletionToInteractionsConverter();
    const events: InteractionEvent[] = [
      created,
      { event_type: "step.start", index: 0, step: { type: "model_output" } },
      {
        event_type: "step.delta",
        event_id: "a",
        index: 0,
        delta: { type: "text", text: "你好" },
        metadata: { total_usage: { ...usage, total_output_tokens: 1, total_tokens: 30 } },
      },
      { event_type: "step.delta", event_id: "a", index: 0, delta: { type: "text", text: "你好" } },
      { event_type: "step.delta", index: 0, delta: { type: "text", text: "🙂" } },
      { event_type: "step.stop", index: 0, usage, step_usage: usage },
      completed,
    ];
    const chunks = events.flatMap(event => converter.convertStreamEvent(event));
    converter.finishStream();
    expect(
      chunks.flatMap(chunk => chunk.choices.map(choice => choice.delta.content ?? "")).join("")
    ).toBe("你好🙂");
    expect(chunks.filter(chunk => chunk.usage)).toHaveLength(1);
    expect(chunks.at(-1)?.usage).toEqual(
      new ChatCompletionToInteractionsConverter().convertResponse(response).usage
    );
    expect(chunks.at(-2)?.choices[0].finish_reason).toBe("stop");
    expect(new Set(chunks.map(chunk => chunk.id)).size).toBe(1);
  });

  test.each([
    ["", "interaction_1"],
    ["interaction_1", "interaction_1"],
    ["", ""],
    ["interaction_1", ""],
  ])(
    "keeps one CC ID through text, tools and final usage (initial ID: %j, final ID: %j)",
    (initialId, finalId) => {
      const converter = new ChatCompletionToInteractionsConverter();
      const events: InteractionEvent[] = [
        { ...created, interaction: { ...created.interaction!, id: initialId } },
        { ...created, interaction: { ...created.interaction!, id: "" } },
        { event_type: "step.start", index: 0, step: { type: "model_output" } },
        { event_type: "step.delta", index: 0, delta: { type: "text", text: "你好🙂" } },
        { event_type: "step.stop", index: 0 },
        {
          event_type: "step.start",
          index: 1,
          step: { type: "function_call", id: "call_1", name: "weather" },
        },
        {
          event_type: "step.delta",
          index: 1,
          delta: { type: "arguments_delta", arguments: '{"city":"北京"}' },
        },
        { event_type: "step.stop", index: 1 },
        { ...completed, interaction: { ...completed.interaction!, id: finalId } },
      ];
      const original = structuredClone(events);
      const chunks = events.flatMap(event => converter.convertStreamEvent(event));
      converter.finishStream();
      expect(chunks[0].id).toMatch(initialId ? /^interaction_1$/ : /^chatcmpl-[a-z0-9]+$/);
      expect(new Set(chunks.map(chunk => chunk.id))).toEqual(new Set([chunks[0].id]));
      expect(events).toEqual(original);
      expect(
        chunks.flatMap(chunk => chunk.choices.map(choice => choice.delta.content ?? "")).join("")
      ).toBe("你好🙂");
      expect(
        chunks.flatMap(chunk => chunk.choices.flatMap(choice => choice.delta.tool_calls ?? []))
      ).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ id: "call_1" }),
          expect.objectContaining({ function: { arguments: '{"city":"北京"}' } }),
        ])
      );
      expect(chunks.at(-2)?.choices[0].finish_reason).toBe("tool_calls");
      expect(chunks.filter(chunk => chunk.usage)).toHaveLength(1);
      expect(chunks.at(-1)?.usage).toEqual(
        new ChatCompletionToInteractionsConverter().convertResponse(response).usage
      );
    }
  );

  test("uses an initial top-level interaction ID when the resource ID is empty", () => {
    const converter = new ChatCompletionToInteractionsConverter();
    const chunks = converter.convertStreamEvent({
      ...created,
      interaction_id: response.id,
      interaction: { ...created.interaction!, id: "" },
    });
    expect(chunks[0].id).toBe(response.id);
  });

  test("a generated CC ID does not hide truncation, missing final native ID or upstream errors", () => {
    const converter = new ChatCompletionToInteractionsConverter();
    converter.convertStreamEvent({
      ...created,
      interaction: { ...created.interaction!, id: "" },
    });
    expect(() => converter.finishStream()).toThrow("before interaction.completed");
    expect(() =>
      converter.convertStreamEvent({
        ...completed,
        interaction: { ...response, id: undefined as unknown as string },
      })
    ).toThrow("Missing interaction ID");
    expect(() => converter.finishStream()).toThrow("before interaction.completed");
    expect(() =>
      converter.convertStreamEvent({
        event_type: "error",
        error: { code: 429, message: "fixture rate limit" },
      })
    ).toThrow("fixture rate limit");
  });

  test.each(["failed", "cancelled", "in_progress"] as const)(
    "rejects a terminal %s status even with an empty native ID",
    status => {
      const converter = new ChatCompletionToInteractionsConverter();
      expect(() =>
        converter.convertStreamEvent({
          ...completed,
          interaction: { ...response, id: "", status },
        })
      ).toThrow(`Unexpected interaction status: ${status}`);
      expect(() => converter.finishStream()).toThrow("before interaction.completed");
    }
  );

  test("keeps parallel streamed tool calls separate and does not prepend empty JSON", () => {
    const converter = new ChatCompletionToInteractionsConverter();
    const chunks = [
      created,
      {
        event_type: "step.start",
        index: 0,
        step: { type: "function_call", id: "a", name: "f", arguments: {} },
      },
      {
        event_type: "step.start",
        index: 1,
        step: { type: "function_call", id: "b", name: "g", arguments: {} },
      },
      {
        event_type: "step.delta",
        index: 0,
        delta: { type: "arguments_delta", arguments: '{"city":' },
      },
      {
        event_type: "step.delta",
        index: 1,
        delta: { type: "arguments_delta", arguments: '{"n":2}' },
      },
      {
        event_type: "step.delta",
        index: 0,
        delta: { type: "arguments_delta", arguments: '"北京"}' },
      },
      { event_type: "step.stop", index: 0 },
      { event_type: "step.stop", index: 1 },
      { ...completed, interaction: { ...response, steps: undefined, status: "requires_action" } },
    ].flatMap(event => converter.convertStreamEvent(event as InteractionEvent));
    const calls = chunks.flatMap(chunk =>
      chunk.choices.flatMap(choice => choice.delta.tool_calls ?? [])
    );
    expect(
      calls
        .filter(call => call.index === 0)
        .map(call => call.function?.arguments ?? "")
        .join("")
    ).toBe('{"city":"北京"}');
    expect(
      calls
        .filter(call => call.index === 1)
        .map(call => call.function?.arguments ?? "")
        .join("")
    ).toBe('{"n":2}');
    expect(chunks.at(-2)?.choices[0].finish_reason).toBe("tool_calls");
  });

  test("converts a final-only snapshot once", () => {
    const converter = new ChatCompletionToInteractionsConverter();
    const chunks = converter.convertStreamEvent({ ...completed, interaction: response });
    expect(
      chunks.flatMap(chunk => chunk.choices.map(choice => choice.delta.content ?? "")).join("")
    ).toBe("你好🙂");
  });

  test("fills missing final text after an empty start without replaying prior deltas", () => {
    const converter = new ChatCompletionToInteractionsConverter();
    converter.convertStreamEvent(created);
    converter.convertStreamEvent({
      event_type: "step.start",
      index: 0,
      step: { type: "model_output" },
    });
    converter.convertStreamEvent({
      event_type: "step.delta",
      index: 0,
      delta: { type: "text", text: "你好" },
    });
    const chunks = converter.convertStreamEvent({ ...completed, interaction: response });
    expect(
      chunks.flatMap(chunk => chunk.choices.map(choice => choice.delta.content ?? "")).join("")
    ).toBe("🙂");
  });

  test("converts byte citation offsets for Chinese and emoji across text parts", () => {
    const result = new ChatCompletionToInteractionsConverter().convertResponse({
      ...response,
      steps: [
        {
          type: "model_output",
          content: [
            { type: "text", text: "🙂" },
            {
              type: "text",
              text: "你好",
              annotations: [
                { type: "url_citation", url: "https://example.com", start_index: 0, end_index: 6 },
              ],
            },
          ],
        },
      ],
    });
    expect(result.choices[0].message.annotations?.[0].url_citation).toMatchObject({
      start_index: 1,
      end_index: 3,
    });
  });

  test("rejects truncated stream and invalid function arguments instead of declaring success", () => {
    const converter = new ChatCompletionToInteractionsConverter();
    converter.convertStreamEvent(created);
    expect(() => converter.finishStream()).toThrow("before interaction.completed");
    converter.convertStreamEvent({
      event_type: "step.start",
      index: 0,
      step: { type: "function_call", id: "a", name: "f" },
    });
    converter.convertStreamEvent({
      event_type: "step.delta",
      index: 0,
      delta: { type: "arguments_delta", arguments: "{" },
    });
    expect(() => converter.convertStreamEvent(completed)).toThrow(
      "Incomplete or invalid function arguments"
    );
  });

  test.each(["failed", "cancelled", "in_progress"] as const)(
    "does not turn %s into stop",
    status => {
      expect(() =>
        new ChatCompletionToInteractionsConverter().convertResponse({ ...response, status })
      ).toThrow();
    }
  );
  test("maps incomplete to length and keeps real usage", () => {
    const result = new ChatCompletionToInteractionsConverter().convertResponse({
      ...response,
      status: "incomplete",
    });
    expect(result.choices[0].finish_reason).toBe("length");
    expect(result.usage?.total_tokens).toBe(49);
  });

  test.each(["image", "audio", "video", "document"])(
    "requires an explicit native carrier for %s output",
    type => {
      const media = {
        ...response,
        steps: [
          {
            type: "model_output",
            content: [{ type, mime_type: "application/octet-stream", data: "YWJj" }],
          },
        ],
      };
      expect(() => new ChatCompletionToInteractionsConverter().convertResponse(media)).toThrow(
        "media extension"
      );
      const converted = new ChatCompletionToInteractionsConverter({
        allowUnmappedContent: true,
      }).convertResponse(media);
      expect(converted.choices[0].message.content).toBeNull();
      expect(converted.choices[0].finish_reason).toBe("stop");
      expect(media.steps[0].content[0].data).toBe("YWJj");
    }
  );
});
