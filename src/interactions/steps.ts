import {
  InteractionsResponseError,
  type InteractionStartStep,
  type InteractionStep,
} from "./types";

/** Check final-snapshot identity before mutating a protocol-specific stream state. */
export function validateFinalStep(previous: InteractionStartStep, final: InteractionStep): void {
  if (previous.type !== final.type)
    throw new InteractionsResponseError("Final step type disagrees with stream");
  if (
    previous.type === "function_call" &&
    final.type === "function_call" &&
    (previous.id !== final.id || previous.name !== final.name)
  )
    throw new InteractionsResponseError("Final function identity disagrees with stream");
  if (
    "signature" in previous &&
    previous.signature &&
    "signature" in final &&
    final.signature != null &&
    previous.signature !== final.signature
  )
    throw new InteractionsResponseError("Final signature disagrees with stream");
  if ("content" in final && final.content) {
    const received = interactionStepContent(previous).filter(part => part.type !== "text");
    const media = final.content.filter(part => part.type !== "text");
    if (received.some((part, index) => !sameJson(part, media[index])))
      throw new InteractionsResponseError("Final media disagrees with stream");
  }
}

export function interactionStepContent(step: InteractionStartStep) {
  return step.type === "thought"
    ? (step.summary ?? [])
    : "content" in step
      ? (step.content ?? [])
      : [];
}

export function sameJson(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a == null || b == null || typeof a !== "object" || typeof b !== "object") return false;
  const left = a as Record<string, unknown>,
    right = b as Record<string, unknown>;
  return (
    Array.isArray(a) === Array.isArray(b) &&
    Object.keys(left).length === Object.keys(right).length &&
    Object.keys(left).every(key => Object.hasOwn(right, key) && sameJson(left[key], right[key]))
  );
}
