import type Anthropic from "@anthropic-ai/sdk";
import { sanitizeAnthropicInputSchema } from "../sanitize-anthropic-schema";

describe("sanitizeAnthropicInputSchema", () => {
  it("leaves plain object schemas untouched", () => {
    const schema = {
      type: "object",
      properties: { a: { type: "string" } },
      required: ["a"],
    } as unknown as Anthropic.Tool.InputSchema;
    expect(sanitizeAnthropicInputSchema(schema)).toBe(schema);
  });

  it("leaves nested combinators untouched", () => {
    const schema = {
      type: "object",
      properties: { a: { anyOf: [{ type: "string" }, { type: "null" }] } },
    } as unknown as Anthropic.Tool.InputSchema;
    expect(sanitizeAnthropicInputSchema(schema)).toBe(schema);
  });

  it("flattens a top-level oneOf discriminated union (Codex automation_update shape)", () => {
    const schema = {
      type: "object",
      properties: {},
      oneOf: [{ $ref: "#/$defs/view" }, { $ref: "#/$defs/create" }],
      $defs: {
        view: {
          type: "object",
          properties: { id: { type: "string" }, mode: { type: "string", enum: ["view"] } },
          required: ["mode", "id"],
          additionalProperties: false,
        },
        create: {
          type: "object",
          properties: { name: { type: "string" }, mode: { type: "string", enum: ["create"] } },
          required: ["mode", "name"],
          additionalProperties: false,
        },
      },
    } as unknown as Anthropic.Tool.InputSchema;

    const out = sanitizeAnthropicInputSchema(schema) as Record<string, any>;

    // No top-level combinator remains (Anthropic rejects those).
    expect(out.oneOf).toBeUndefined();
    expect(out.anyOf).toBeUndefined();
    expect(out.allOf).toBeUndefined();
    expect(out.type).toBe("object");
    // Every branch property is preserved (union), so params stay visible to the model.
    expect(Object.keys(out.properties).sort()).toEqual(["id", "mode", "name"]);
    // required is the intersection across branches (present in both → "mode").
    expect(out.required).toEqual(["mode"]);
    // $defs kept so surviving $refs still resolve.
    expect(out.$defs).toBeDefined();
  });

  it("recurses through nested combinators (Codex nests oneOf two levels deep)", () => {
    const schema = {
      type: "object",
      properties: {},
      oneOf: [{ $ref: "#/$defs/view" }, { $ref: "#/$defs/create" }],
      $defs: {
        view: {
          type: "object",
          properties: { id: { type: "string" }, mode: { type: "string" } },
          required: ["mode", "id"],
        },
        // create is itself a union of sub-variants (nested combinator).
        create: { oneOf: [{ $ref: "#/$defs/createCron" }, { $ref: "#/$defs/createHeartbeat" }] },
        createCron: {
          type: "object",
          properties: { mode: { type: "string" }, rrule: { type: "string" } },
          required: ["mode", "rrule"],
        },
        createHeartbeat: {
          type: "object",
          properties: { mode: { type: "string" }, prompt: { type: "string" } },
          required: ["mode", "prompt"],
        },
      },
    } as unknown as Anthropic.Tool.InputSchema;

    const out = sanitizeAnthropicInputSchema(schema) as Record<string, any>;
    expect(out.oneOf).toBeUndefined();
    expect(Object.keys(out.properties).sort()).toEqual(["id", "mode", "prompt", "rrule"]);
    // Only "mode" is required across every leaf branch.
    expect(out.required).toEqual(["mode"]);
  });

  it("handles top-level anyOf and drops additionalProperties", () => {
    const schema = {
      type: "object",
      anyOf: [
        { type: "object", properties: { x: { type: "number" } } },
        { type: "object", properties: { y: { type: "number" } } },
      ],
      additionalProperties: false,
    } as unknown as Anthropic.Tool.InputSchema;

    const out = sanitizeAnthropicInputSchema(schema) as Record<string, any>;
    expect(out.anyOf).toBeUndefined();
    expect(Object.keys(out.properties).sort()).toEqual(["x", "y"]);
    expect(out.required).toBeUndefined();
    expect(out.additionalProperties).toBeUndefined();
  });
});
