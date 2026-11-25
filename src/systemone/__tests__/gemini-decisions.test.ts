import type { Decisions } from "@google/genai";
import {
  SystemOneDecisionBlockedError,
  SystemOneDecisionResponseError,
  SystemOneDecisionRequestError,
  SystemOneToGeminiDecisionsConverter,
} from "../../index";

const converter = new SystemOneToGeminiDecisionsConverter();
const model = "gemini-decision-1";
const question = { type: "noul", instructions: "Any damage?" };
const choice = {
  type: "choice",
  instructions: "Pick",
  criteria: { "one item": null, 其他: "multiple" },
};
const score = { type: "score", instructions: "Rate", criteria: ["low", "high"] };
const image = (data: string) => ({ type: "image" as const, mime_type: "image/png", data });
const images = new Map([
  ["<a>", image("A")],
  ["<b>", image("B")],
]);
const req = (questions: unknown, state: unknown = "state") => ({ model, questions, state });
const usage = { total_input_tokens: 42, total_output_tokens: 0 };

it("preserves state structure, answer IDs, option IDs, numeric scores and metering across a mixed batch", () => {
  const input = req(
    { 问题: { ...question, criteria: { false: "intact", true: "broken" } }, q_1: choice, score },
    { msg: ["test", 2] }
  );
  const r = converter.convertRequest(input);
  expect(r.body).toEqual({
    model,
    input: '{"msg":["test",2]}',
    questions: {
      q_2: { ...question, criteria: { false_description: "intact", true_description: "broken" } },
      q_1: { ...choice, criteria: { "one item": "", 其他: "multiple" } },
      score,
    },
  });
  const answers = {
    q_2: { type: "noul", noul: 0.2, confidence: 0.8 },
    q_1: {
      type: "choice",
      choice: "其他",
      probabilities: { "one item": 0.1, 其他: 0.9 },
      confidence: 0.9,
    },
    score: {
      type: "score",
      score: 0.75,
      legend: { "0": "low", "1": "high" },
      probabilities: { "0": 0.25, "1": 0.75 },
      confidence: 0.75,
    },
  } as const;
  expect(converter.convertResponse({ usage, answers }, r)).toEqual({
    model,
    answers: { 问题: answers.q_2, q_1: answers.q_1, score: answers.score },
    usage: { input_tokens: 42, output_tokens: 0 },
  });
});
it("keeps question-specific text private, binds repeated references, restores the score legend", () => {
  const r = converter.convertRequest(
    req(
      {
        rating: { ...score, criteria: ["unlike <b>", "like <b>"] },
        compare: { ...question, instructions: "Private text: <b> vs <a>" },
      },
      { scene: ["[Image 1]", "<a><a>"] }
    ),
    images
  );
  expect(r.body.input).toEqual([
    { type: "text", text: JSON.stringify({ scene: ["[Image 1]", "[Image 2][Image 2]"] }) },
    { type: "text", text: "\n[Image 2]:\n" },
    image("A"),
    { type: "text", text: "\n[Image 3]:\n" },
    image("B"),
  ]);
  expect(r.body.questions.compare.instructions).toBe("Private text: [Image 3] vs [Image 2]");
  const answers = {
    rating: {
      type: "score",
      score: 0.4,
      legend: { 0: "unlike [Image 3]", 1: "like [Image 3]" },
      probabilities: { 0: 0.6, 1: 0.4 },
      confidence: 0.6,
    },
    compare: { type: "noul", noul: 0.7, confidence: 0.7 },
  } as const;
  expect(converter.convertResponse({ answers, usage }, r).answers.rating).toMatchObject({
    score: 0.4,
    legend: { 0: "unlike <b>", 1: "like <b>" },
  });
});
it("allows shared pictures with different focus and an image inside a single question's structured instructions", () => {
  expect(
    converter.convertRequest(
      req({ a: { ...question, instructions: "<a>" }, b: { ...question, instructions: "<b>" } }, [
        "<b>",
        "<a>",
      ]),
      images
    ).body.input
  ).toEqual(expect.arrayContaining([{ type: "text", text: '["[Image 2]","[Image 1]"]' }]));
  expect(
    converter.convertRequest(
      req({ a: { ...question, instructions: { compare: ["<b>", "<a>"] } } }),
      images
    ).body.questions.a.instructions
  ).toBe('{"compare":["[Image 2]","[Image 1]"]}');
});
it.each([
  [
    req({ a: { ...question, instructions: "<b>" }, b: question }, "<a>"),
    images,
    /questions\.a\.instructions: <b>/,
  ],
  [req({ a: question, b: { type: "box", instructions: "where" } }), new Map(), /questions.b.type/],
  [req({ a: question }, { "<a>": "<b>" }), images, /not referenced/],
  [
    req({ a: question }, "abc"),
    new Map([
      ["ab", image("A")],
      ["bc", image("B")],
    ]),
    /overlapping/,
  ],
  [req({ a: { ...choice, criteria: { "one ": null, two: null } } }), new Map(), /option ID/],
  [req({ a: { ...choice, criteria: { "a\u00a0b": null, two: null } } }), new Map(), /option ID/],
  [req({ a: { ...score, criteria: ["ok", ""] } }), new Map(), /criteria\[1\]/],
  [req({ a: { ...question, criteria: { unexpected: "yes" } } }), new Map(), /criteria.unexpected/],
  [req({ a: { ...question, extra: true } }), new Map(), /questions.a.extra/],
  [
    req(Object.fromEntries(Array.from({ length: 1001 }, (_, i) => [i, question]))),
    new Map(),
    /1000/,
  ],
] as const)("rejects unrepresentable requests (#%#)", (input, bindings, expected) => {
  const convert = () => converter.convertRequest(input, bindings);
  expect(convert).toThrow(SystemOneDecisionRequestError);
  expect(convert).toThrow(expected);
});
it("handles special object keys as data", () => {
  const request = converter.convertRequest(req({ ["__proto__"]: question }));
  const a = { type: "noul", noul: 0.75, confidence: 0.75 } as const;
  const result = converter.convertResponse({ answers: { ["__proto__"]: a }, usage }, request);
  expect(Object.keys(result.answers)).toEqual(["__proto__"]);
  expect(result.answers["__proto__"]).toEqual(a);
});
const good = { type: "noul" as const, noul: 0.7, confidence: 0.7 };
it.each([
  [{ answers: { q: good }, usage: {} }, /total_input_tokens/],
  [
    { answers: { q: good }, usage: { total_input_tokens: 2, total_output_tokens: 3 } },
    /total_output_tokens/,
  ],
  [{ answers: { other: good }, usage }, /question IDs/],
  [{ answers: { q: good, extra: good }, usage }, /question IDs/],
  [{ answers: { q: { ...good, noul: 1.2 } }, usage }, /noul/],
  [{ answers: { q: { type: "score" } }, usage }, /type/],
] as const)("does not fabricate an answer or usage (#%#)", (raw, expected) => {
  expect(() =>
    converter.convertResponse(
      raw as Decisions.CreateDecisionResponse,
      converter.convertRequest(req({ q: question }))
    )
  ).toThrow(expected);
});

it("rejects a probability map for different choices and a score legend that does not match the rubric", () => {
  const r = converter.convertRequest(req({ c: choice, s: score }));
  const c = {
    type: "choice",
    choice: "其他",
    probabilities: { "one item": 0.2, 其他: 0.8 },
    confidence: 0.8,
  } as const;
  const s = {
    type: "score",
    score: 0.5,
    legend: { 0: "low", 1: "high" },
    probabilities: { 0: 0.5, 1: 0.5 },
    confidence: 0.5,
  } as const;
  expect(() =>
    converter.convertResponse(
      { usage, answers: { c: { ...c, probabilities: { other: 1 } }, s } },
      r
    )
  ).toThrow("answers.c.probabilities");
  expect(() =>
    converter.convertResponse(
      { usage, answers: { c, s: { ...s, legend: { 0: "high", 1: "low" } } } },
      r
    )
  ).toThrow("answers.s.legend");
});

it("treats an object key as business text even if a question uses the same string as an image", () => {
  const r = converter.convertRequest(
    req({ q: { ...question, instructions: "compare <a>" } }, { "<a>": "left" }),
    new Map([["<a>", image("A")]])
  );
  expect(r.body.input).toEqual([
    { type: "text", text: '{"<a>":"left"}' },
    { type: "text", text: "\n[Image 1]:\n" },
    image("A"),
  ]);
  expect(r.body.questions.q.instructions).toBe("compare [Image 1]");
});
it.each([false, true])(
  "identifies the image and the field that introduced it regardless of question order (%s)",
  reverse => {
    const rows = [
      ["private", { ...score, criteria: ["low", "high <b>"] }],
      ["other", question],
    ] as const;
    expect(() =>
      converter.convertRequest(
        req(Object.fromEntries(reverse ? [...rows].reverse() : rows), "<a>"),
        images
      )
    ).toThrow("questions.private.criteria[1]: <b>");
  }
);
it.each(["safety", "other", "future_reason", "unspecified"])(
  "preserves explicit block reason %s before looking for answers or usage",
  reason => {
    const request = converter.convertRequest(req({ q: question }));
    const convert = () =>
      converter.convertResponse(
        { prompt_feedback: { block_reason: reason, block_reason_message: "filtered" } },
        request
      );
    expect(convert).toThrow(SystemOneDecisionBlockedError);
    expect(convert).toThrow(
      expect.objectContaining({
        blockReason: reason,
        blockReasonMessage: "filtered",
        message: "filtered",
      })
    );
  }
);
it("keeps empty feedback without answers a malformed response", () => {
  const convert = () =>
    converter.convertResponse(
      { prompt_feedback: {} },
      converter.convertRequest(req({ q: question }))
    );
  expect(convert).toThrow(SystemOneDecisionResponseError);
  expect(convert).not.toThrow(SystemOneDecisionBlockedError);
});
