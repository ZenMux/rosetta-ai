import type { InteractionEvent, InteractionSnapshot, InteractionUsage } from "./types";

export function eventInteraction(event: InteractionEvent): InteractionSnapshot | undefined {
  return "interaction" in event ? event.interaction : undefined;
}

export function eventInteractionId(event: InteractionEvent): string | undefined {
  return "interaction_id" in event ? event.interaction_id : undefined;
}

export function eventStatus(event: InteractionEvent): InteractionSnapshot["status"] | undefined {
  return eventInteraction(event)?.status ?? ("status" in event ? event.status : undefined);
}

/** Cumulative snapshots only: never add step_usage to the interaction total. */
export function eventUsage(event: InteractionEvent): InteractionUsage | undefined {
  return eventInteraction(event)?.usage ?? event.metadata?.total_usage ?? event.usage;
}
