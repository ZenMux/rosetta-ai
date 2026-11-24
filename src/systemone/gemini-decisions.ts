import type { Decisions } from "@google/genai";
import {
  convertSystemOneDecisionRequest,
  type ConvertedDecisionRequest,
  type DecisionImage,
  type SystemOneDecisionRequest,
} from "./gemini-decisions-request";

function within(n: unknown, min: number, max: number): n is number {
  return typeof n === "number" && Number.isFinite(n) && n >= min && n <= max;
}
function sameKeys(value: unknown, expected: string[]): boolean {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value).length === expected.length &&
    expected.every(k => Object.hasOwn(value, k))
  );
}
function invalid(path: string): never {
  throw new Error(`Invalid Gemini Decisions response: ${path}`);
}

type SystemOneAnswer =
  | Exclude<Decisions.Answer, { type: "noul" }>
  | { type: "noul"; noul: number; confidence?: number };
interface SystemOneDecisionResponse {
  model: string;
  answers: Record<string, SystemOneAnswer>;
  usage: { input_tokens: number; output_tokens: number };
}
export class SystemOneToGeminiDecisionsConverter {
  convertRequest(
    request: SystemOneDecisionRequest,
    images: ReadonlyMap<string, DecisionImage> = new Map()
  ): ConvertedDecisionRequest {
    return convertSystemOneDecisionRequest(request, images);
  }

  convertResponse(
    response: Decisions.CreateDecisionResponse,
    request: ConvertedDecisionRequest
  ): SystemOneDecisionResponse {
    const { answers, usage } = response;
    if (
      response.prompt_feedback?.block_reason &&
      response.prompt_feedback.block_reason !== "unspecified"
    )
      invalid("prompt_feedback.block_reason");
    if (
      !answers ||
      !sameKeys(
        answers,
        request.mapping.map(m => m.id)
      )
    )
      invalid("answers must match the requested question IDs");
    const input = usage?.total_input_tokens;
    if (typeof input !== "number" || !Number.isSafeInteger(input) || input < 0)
      invalid("usage.total_input_tokens");
    if (usage?.total_output_tokens !== undefined && usage.total_output_tokens !== 0)
      invalid("usage.total_output_tokens");
    const result = request.mapping.map<readonly [string, SystemOneAnswer]>(
      ({ id, originalId, question, originalLevels }) => {
        const a = answers[id];
        const p = `answers.${id}`;
        if (!a || a.type !== question.type) invalid(`${p}.type`);
        if (a.type === "noul") {
          if (!within(a.noul, 0.01, 0.99)) invalid(`${p}.noul`);
          if (a.confidence !== undefined && !within(a.confidence, 0, 1)) invalid(`${p}.confidence`);
          return [
            originalId,
            {
              type: a.type,
              noul: a.noul,
              ...(a.confidence !== undefined && { confidence: a.confidence }),
            },
          ] as const;
        }
        if (!within(a.confidence, 0, 1)) invalid(`${p}.confidence`);
        const keys = Object.keys(question.criteria!);
        if (
          !sameKeys(a.probabilities, keys) ||
          !Object.values(a.probabilities).every(v => within(v, 0, 1))
        )
          invalid(`${p}.probabilities`);
        if (a.type === "choice") {
          if (!keys.includes(a.choice)) invalid(`${p}.choice`);
          return [
            originalId,
            {
              type: a.type,
              choice: a.choice,
              probabilities: a.probabilities,
              confidence: a.confidence,
            },
          ] as const;
        }
        if (question.type !== "score" || !originalLevels || !within(a.score, 0, keys.length - 1))
          invalid(`${p}.score`);
        if (
          !sameKeys(a.legend, keys) ||
          !question.criteria.every((level, i) => a.legend[i] === level)
        )
          invalid(`${p}.legend`);
        return [
          originalId,
          {
            type: a.type,
            score: a.score,
            legend: Object.fromEntries(originalLevels.entries()),
            probabilities: a.probabilities,
            confidence: a.confidence,
          },
        ] as const;
      }
    );
    return {
      model: request.body.model,
      answers: Object.fromEntries(result),
      usage: { input_tokens: input, output_tokens: 0 },
    };
  }
}
