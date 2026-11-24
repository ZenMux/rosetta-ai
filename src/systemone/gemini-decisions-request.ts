import type { Decisions } from "@google/genai";

export interface SystemOneDecisionRequest {
  model: string;
  state: unknown;
  questions: unknown;
}

// Bindings are supplied by the caller; Rosetta never opens files or fetches URLs.
export type DecisionImage = Extract<Decisions.DecisionInputContent, { type: "image" }>;
export interface DecisionQuestionMapping {
  id: string;
  originalId: string;
  question: Decisions.Question;
  originalLevels?: string[];
}

export class SystemOneDecisionRequestError extends Error {}
function invalid(path: string, message: string): never {
  throw new SystemOneDecisionRequestError(`${path}: ${message}`);
}
function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function text(value: unknown, path: string): string {
  if (typeof value !== "string" || !value.trim()) invalid(path, "expected non-empty text");
  return value;
}
function count(n: number, path: string, min: number, max: number) {
  if (n < min || n > max) invalid(path, `expected ${min} to ${max} entries`);
}

// Only string values carry references. Object keys (including option IDs) are business data.
function render(value: unknown, path: string, replace: (s: string) => string): string {
  if (typeof value === "string") return text(replace(value), path);
  if (!record(value) && !Array.isArray(value))
    invalid(path, "expected text, an object or an array");
  return JSON.stringify(value, (_key, v) => (typeof v === "string" ? replace(v) : v));
}

function convertQuestion(
  value: unknown,
  path: string,
  replace: (s: string) => string
): Decisions.Question {
  if (!record(value)) invalid(path, "expected a question object");
  for (const field of Object.keys(value)) {
    if (!["type", "instructions", "criteria"].includes(field))
      invalid(`${path}.${field}`, "unsupported question field");
  }
  const { type, criteria } = value;
  if (type !== "noul" && type !== "choice" && type !== "score")
    invalid(`${path}.type`, "Gemini Decisions supports noul, choice and score");
  const instructions = render(value.instructions, `${path}.instructions`, replace);
  const p = `${path}.criteria`;
  if (type === "score") {
    if (!Array.isArray(criteria)) invalid(p, "expected an ordered array of levels");
    count(criteria.length, p, 2, 255);
    return { type, instructions, criteria: criteria.map((v, i) => replace(text(v, `${p}[${i}]`))) };
  }
  if (type === "noul" && criteria === undefined) return { type, instructions };
  if (!record(criteria)) invalid(p, "expected an object");
  if (type === "noul") {
    return {
      type,
      instructions,
      criteria: Object.fromEntries(
        Object.entries(criteria).map(([k, v]) => {
          if (k !== "true" && k !== "false") invalid(`${p}.${k}`, "expected true or false");
          if (typeof v !== "string") invalid(`${p}.${k}`, "expected text");
          return [`${k}_description`, replace(v)];
        })
      ),
    };
  }
  const entries = Object.entries(criteria);
  count(entries.length, p, 2, 255);
  return {
    type,
    instructions,
    criteria: Object.fromEntries(
      entries.map(([k, v]) => {
        if (
          !k ||
          k.trim() !== k ||
          [...k].length > 128 ||
          /[\p{Cc}\s]/u.test(k.replaceAll(" ", ""))
        )
          invalid(`${p}.${k}`, "invalid Google choice option ID");
        if (v !== null && typeof v !== "string") invalid(`${p}.${k}`, "expected text or null");
        return [k, v === null ? "" : replace(v)];
      })
    ),
  };
}

export interface ConvertedDecisionRequest {
  body: Decisions.CreateDecisionRequest;
  mapping: DecisionQuestionMapping[];
}
export function convertSystemOneDecisionRequest(
  request: SystemOneDecisionRequest,
  images: ReadonlyMap<string, DecisionImage>
): ConvertedDecisionRequest {
  if (!record(request.questions)) invalid("questions", "expected an object");
  const entries = Object.entries(request.questions);
  count(entries.length, "questions", 1, 1000);
  const labels = new Map<string, string>();
  if (images.size) {
    // Avoid duplicating the whole JSON body just to reserve existing labels.
    const occupied = new Set<string>();
    function reserve(value: unknown) {
      if (typeof value === "string") {
        for (const match of value.matchAll(/\[Image \d+\]/g)) occupied.add(match[0]);
      } else if (Array.isArray(value)) {
        value.forEach(reserve);
      } else if (record(value)) {
        for (const key of Object.keys(value)) {
          reserve(key);
          reserve(value[key]);
        }
      }
    }
    reserve(request.state);
    reserve(request.questions);
    let i = 0;
    for (const ref of images.keys()) {
      if (!ref.trim()) invalid("assets", "empty image reference");
      let label: string;
      do {
        label = `[Image ${++i}]`;
      } while (occupied.has(label));
      labels.set(ref, label);
    }
  }
  function replacer(used: Set<string>, path: string) {
    return (s: string): string => {
      // Track one occurrence per distinct image, not an array of every reference in a large state.
      const pending = [];
      for (const [ref, label] of labels) {
        const start = s.indexOf(ref);
        if (start !== -1) pending.push({ ref, label, start });
      }
      if (!pending.length) return s;
      let output = "",
        end = 0;
      while (pending.length) {
        let next = 0;
        for (let i = 1; i < pending.length; i++)
          if (pending[i].start < pending[next].start) next = i;
        const hit = pending[next];
        if (hit.start < end) invalid(path, "overlapping image references");
        output += s.slice(end, hit.start) + hit.label;
        end = hit.start + hit.ref.length;
        used.add(hit.ref);
        hit.start = s.indexOf(hit.ref, hit.start + 1);
        if (hit.start === -1) pending.splice(next, 1);
      }
      return output + s.slice(end);
    };
  }
  const shared = new Set<string>();
  const state = render(request.state, "state", replacer(shared, "state"));
  const reservedIds = new Set(Object.keys(request.questions));
  let sequence = 0;
  let firstExtra: Set<string> | undefined;
  const mapping = entries.map(([originalId, raw]): DecisionQuestionMapping => {
    const path = `questions.${originalId}`;
    const used = new Set<string>();
    const question = convertQuestion(raw, path, replacer(used, path));
    for (const ref of used) if (shared.has(ref)) used.delete(ref);
    const previous = firstExtra;
    if (previous && (previous.size !== used.size || [...used].some(r => !previous.has(r))))
      invalid(path, "Google Decisions requires the same images to be visible to every question");
    firstExtra ??= used;
    let id = originalId;
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(id)) {
      do {
        id = `q_${++sequence}`;
      } while (reservedIds.has(id));
    }
    return {
      id,
      originalId,
      question,
      ...(question.type === "score"
        ? { originalLevels: (raw as { criteria: string[] }).criteria }
        : {}),
    };
  });
  const blocks: Decisions.DecisionInputContent[] = [{ type: "text", text: state }];
  for (const [ref, image] of images) {
    if (!shared.has(ref) && !firstExtra?.has(ref))
      invalid(`assets.${ref}`, "image is not referenced by any string value");
    blocks.push({ type: "text", text: `\n${labels.get(ref)}:\n` }, image);
  }
  const body: Decisions.CreateDecisionRequest = {
    model: request.model,
    input: images.size ? blocks : state,
    questions: Object.fromEntries(mapping.map(m => [m.id, m.question])),
  };
  return { body, mapping };
}
