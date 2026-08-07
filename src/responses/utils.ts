import type OpenAI from "openai";

/** Separator joining a namespace and tool name in a flattened function name. */
export const NAMESPACE_SEPARATOR = "___";

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
 * top-level function tool whose name is namespaced as `<namespace>___<tool>`
 * (e.g. `agents___spawn_agent`). A triple-underscore separator is used (not a
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
          name: nsName ? `${nsName}${NAMESPACE_SEPARATOR}${innerName}` : innerName,
        });
      }
      continue;
    }
    expanded.push(tool);
  }

  return (hasNamespace ? expanded : tools) as OpenAI.Responses.ResponseCreateParams["tools"];
}

/**
 * Split a flattened `<namespace>___<tool>` name back into its parts. Returns
 * `{ name }` (no namespace) when the name is not namespaced.
 */
export function splitNamespacedToolName(name: string): { namespace?: string; name: string } {
  if (typeof name !== "string") return { name };
  const idx = name.indexOf(NAMESPACE_SEPARATOR);
  if (idx <= 0) return { name };
  const namespace = name.slice(0, idx);
  const bare = name.slice(idx + NAMESPACE_SEPARATOR.length);
  if (!bare) return { name };
  return { namespace, name: bare };
}

/**
 * Reverse of {@link expandNamespaceTools}: re-nest flattened
 * `<namespace>___<tool>` function tools back into `namespace` tools. Already
 * nested namespace tools and non-namespaced tools pass through untouched;
 * order is preserved (a namespace appears at its first member's position).
 */
export function nestNamespaceTools(tools: any): any {
  if (!Array.isArray(tools)) return tools;

  let changed = false;
  const result: any[] = [];
  const nsByName = new Map<string, any>();

  for (const tool of tools) {
    const t = tool as any;
    if (t && typeof t === "object" && t.type === "function" && typeof t.name === "string") {
      const { namespace, name } = splitNamespacedToolName(t.name);
      if (namespace) {
        changed = true;
        let ns = nsByName.get(namespace);
        if (!ns) {
          ns = { type: "namespace", name: namespace, tools: [] };
          nsByName.set(namespace, ns);
          result.push(ns);
        }
        ns.tools.push({ ...t, name });
        continue;
      }
    }
    result.push(tool);
  }

  return changed ? result : tools;
}

/**
 * If a function_call output item's `name` is a flattened `<namespace>___<tool>`,
 * split it into a `namespace` field plus the bare `name`. Mutates in place.
 */
function denamespaceFunctionCallItem(item: any): void {
  if (!item || item.type !== "function_call" || typeof item.name !== "string") return;
  const { namespace, name } = splitNamespacedToolName(item.name);
  if (namespace) {
    item.name = name;
    item.namespace = namespace;
  }
}

/**
 * Apply the reverse namespace transform to a full Responses response (mutates):
 * split any namespaced `function_call` output items into `{ namespace, name }`
 * and re-nest the echoed `tools` into namespace tools.
 */
export function denamespaceResponse(resp: any): void {
  if (!resp || typeof resp !== "object") return;
  if (Array.isArray(resp.output)) {
    for (const item of resp.output) denamespaceFunctionCallItem(item);
  }
  if (Array.isArray(resp.tools)) {
    resp.tools = nestNamespaceTools(resp.tools);
  }
}

/**
 * Apply the reverse namespace transform to a batch of streaming events
 * (mutates in place, returns the same array). Covers function_call item events,
 * function_call argument events, and terminal events carrying a full response.
 */
export function denamespaceStreamEvents<T extends { type?: string }>(events: T[]): T[] {
  for (const event of events) {
    const e = event as any;
    if (!e || typeof e !== "object") continue;
    switch (e.type) {
      case "response.output_item.added":
      case "response.output_item.done":
        denamespaceFunctionCallItem(e.item);
        break;
      case "response.function_call_arguments.delta":
      case "response.function_call_arguments.done": {
        if (typeof e.name === "string") {
          const { namespace, name } = splitNamespacedToolName(e.name);
          if (namespace) {
            e.name = name;
            e.namespace = namespace;
          }
        }
        break;
      }
      case "response.completed":
      case "response.incomplete":
      case "response.failed":
        denamespaceResponse(e.response);
        break;
    }
  }
  return events;
}
