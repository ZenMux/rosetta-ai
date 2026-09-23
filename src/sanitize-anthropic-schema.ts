import type Anthropic from "@anthropic-ai/sdk";

const TOP_LEVEL_COMBINATORS = ["oneOf", "anyOf", "allOf"] as const;

/**
 * Anthropic's Messages API rejects `oneOf` / `anyOf` / `allOf` at the *top level*
 * of a tool `input_schema` ("input_schema does not support oneOf, allOf, or anyOf
 * at the top level"). Some clients describe a tool as a discriminated union with a
 * top-level combinator — e.g. Codex namespace tools such as
 * `mcp__codex_app/automation_update`, whose schema is
 * `{ type: "object", properties: {}, oneOf: [<branch>, ...], $defs: {...} }`.
 *
 * Flatten the top-level combinator into a plain object schema by unioning the
 * branch properties, so every parameter stays visible to the model. Nested
 * combinators (inside properties / `$defs`) are left untouched — Anthropic allows
 * those. This relaxes the branch-exclusivity hint but preserves the parameter
 * set; the tool's own server still validates the concrete call.
 */
export function sanitizeAnthropicInputSchema(
  schema: Anthropic.Tool.InputSchema
): Anthropic.Tool.InputSchema {
  if (!schema || typeof schema !== "object") return schema;

  const s = schema as Record<string, any>;
  const combinator = TOP_LEVEL_COMBINATORS.find(k => Array.isArray(s[k]));
  if (!combinator) return schema;

  const defs = s.$defs ?? s.definitions;
  const seen = new Set<string>();
  const branches: Record<string, any>[] = (s[combinator] as any[]).flatMap(b =>
    collectObjectSchemas(b, defs, seen)
  );

  const properties: Record<string, any> = { ...(s.properties ?? {}) };
  const requiredSets: string[][] = [];
  for (const branch of branches) {
    if (branch.properties) Object.assign(properties, branch.properties);
    requiredSets.push(Array.isArray(branch.required) ? branch.required : []);
  }
  // required = properties required in every branch (safe intersection); a union
  // of branches can't require branch-specific fields.
  const required =
    requiredSets.length > 0
      ? requiredSets.reduce((acc, cur) => acc.filter(name => cur.includes(name)))
      : [];

  const result: Record<string, any> = { ...s, type: "object", properties };
  for (const k of TOP_LEVEL_COMBINATORS) delete result[k];
  if (required.length > 0) result.required = required;
  else delete result.required;
  // The union spans previously-disjoint branches, so don't forbid other branches' props.
  delete result.additionalProperties;

  return result as Anthropic.Tool.InputSchema;
}

/**
 * Walk a combinator branch to its object-schema leaves, following `$ref` and
 * recursing through nested `oneOf` / `anyOf` / `allOf` (Codex nests union modes
 * two levels deep: top-level oneOf → per-mode oneOf → object). Deduped by `$ref`
 * so shared defs and cycles are visited once.
 */
function collectObjectSchemas(
  node: any,
  defs: Record<string, any> | undefined,
  seen: Set<string>
): Record<string, any>[] {
  if (node && typeof node === "object" && typeof node.$ref === "string") {
    if (seen.has(node.$ref)) return [];
    seen.add(node.$ref);
  }
  const resolved = resolveRef(node, defs);
  if (!resolved || typeof resolved !== "object") return [];

  const combinator = TOP_LEVEL_COMBINATORS.find(k => Array.isArray(resolved[k]));
  if (combinator) {
    return (resolved[combinator] as any[]).flatMap(b => collectObjectSchemas(b, defs, seen));
  }
  if (resolved.type === "object" || resolved.properties) return [resolved];
  return [];
}

// ponytail: one level of $ref resolution against $defs/definitions; deep/remote refs unsupported (Codex schemas ref one hop).
function resolveRef(node: any, defs: Record<string, any> | undefined): any {
  let cur = node;
  const seen = new Set<string>();
  while (cur && typeof cur === "object" && typeof cur.$ref === "string") {
    const m = /^#\/(?:\$defs|definitions)\/(.+)$/.exec(cur.$ref);
    if (!m || !defs || seen.has(cur.$ref)) break;
    seen.add(cur.$ref);
    cur = defs[m[1]];
  }
  return cur;
}
