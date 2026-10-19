import {
  ChatCompletionToInteractionsConverter,
  ResponsesToInteractionsConverter,
  MessagesToInteractionsConverter,
  type Interaction,
} from "../../index";

describe.each([
  ChatCompletionToInteractionsConverter,
  ResponsesToInteractionsConverter,
  MessagesToInteractionsConverter,
])("standard Interactions identity: %p", Converter => {
  test.each([undefined, null, 42])("rejects an invalid resource ID %p", id => {
    // Deliberately malformed wire data, not an SDK-standard response fixture.
    const raw = { id, status: "completed", steps: [] } as unknown as Interaction;
    expect(() => new Converter().convertResponse(raw)).toThrow("Missing interaction ID");
    expect(() =>
      new Converter().convertStreamEvent({ event_type: "interaction.completed", interaction: raw })
    ).toThrow("Missing interaction ID");
  });
});
