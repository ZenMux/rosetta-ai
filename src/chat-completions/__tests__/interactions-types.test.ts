import type { Interactions } from "@google/genai";
import type {
  InteractionCreateParams,
  InteractionEvent,
  InteractionStep,
} from "../../interactions/types";

test("uses SDK request, complete step and discriminated event contracts", () => {
  const request = {
    model: "gemini-omni-1.1-flash",
    input: "hello",
  } satisfies InteractionCreateParams;
  const sdkRequest: Interactions.CreateModelInteractionParamsNonStreaming = request;
  const step = {
    type: "function_call",
    id: "call",
    name: "weather",
    arguments: {},
  } satisfies InteractionStep;
  const sdkStep: Interactions.Step = step;
  const start = {
    event_type: "step.start",
    index: 0,
    step: { type: "function_call", id: "call", name: "weather" },
  } satisfies InteractionEvent;
  // @ts-expect-error Complete function calls still require arguments.
  const incomplete: InteractionStep = start.step;
  // @ts-expect-error Tool arguments are not model-output fields.
  const wrongStep: InteractionStep = { type: "model_output", arguments: {} };
  const wrongDelta: InteractionEvent = {
    event_type: "step.delta",
    index: 0,
    // @ts-expect-error Text is not a field of an arguments delta.
    delta: { type: "arguments_delta", text: "hello" },
  };
  const wrongEvent: InteractionEvent = {
    event_type: "step.start",
    // @ts-expect-error Step events cannot replace their index/step with an interaction.
    interaction: { id: "i", status: "completed" },
  };
  expect(sdkRequest.input).toBe("hello");
  expect(sdkStep.arguments).toEqual({});
  expect([incomplete, wrongStep, wrongDelta, wrongEvent]).toHaveLength(4);
});
