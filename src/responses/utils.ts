import type OpenAI from "openai";

/**
 * Expand Responses "namespace" tools into individual "function" tools.
 *
 * A namespace tool groups related function tools under a shared name:
 *
 *   {
 *     type: "namespace",
 *     name: "agents",
 *     description: "Multi-agent collaboration tools.",
 *     tools: [{ type: "function", name: "spawn_agent", parameters: {...} }]
 *   }
 *
 * Downstream provider protocols (chat-completions / messages / gemini) do not
 * understand the `namespace` tool type, so each inner tool is flattened into a
 * top-level function tool whose name is namespaced as `<namespace>__<tool>`
 * (e.g. `agents__spawn_agent`). A double-underscore separator is used (not a
 * dot) so the name matches the provider's required pattern `^[a-zA-Z0-9_-]+$`.
 *
 * Non-namespace tools pass through untouched and the original order is
 * preserved (a namespace expands in place). When no namespace tool is present
 * the original array is returned by reference, so this is a no-op for the
 * common case.
 */
export function expandNamespaceTools(
  tools: OpenAI.Responses.ResponseCreateParams["tools"]
): OpenAI.Responses.ResponseCreateParams["tools"] {
  if (!Array.isArray(tools)) return tools;

  let hasNamespace = false;
  const expanded: any[] = [];

  for (const tool of tools) {
    if (tool && typeof tool === "object" && (tool as any).type === "namespace") {
      hasNamespace = true;
      const ns = tool as any;
      const nsName: string | undefined = typeof ns.name === "string" ? ns.name : undefined;
      const innerTools = Array.isArray(ns.tools) ? ns.tools : [];
      for (const inner of innerTools) {
        if (!inner || typeof inner !== "object") continue;
        const innerName = (inner as any).name;
        if (typeof innerName !== "string" || innerName === "") continue;
        expanded.push({
          ...inner,
          type: "function",
          name: nsName ? `${nsName}__${innerName}` : innerName,
        });
      }
      continue;
    }
    expanded.push(tool);
  }

  return (hasNamespace ? expanded : tools) as OpenAI.Responses.ResponseCreateParams["tools"];
}
