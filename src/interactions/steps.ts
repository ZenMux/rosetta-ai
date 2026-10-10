import type { InteractionStartStep } from "./types";

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
