# Changelog

## 1.0.41

### Responses failures converted to Messages

`MessagesToResponsesConverter.convertStreamEvent()` and `convertStream()` now
throw an OpenAI `APIError` on `response.failed` and standalone `error` events.
Previously, a failed response produced a successful Messages terminal, and
standalone error events were ignored.

This changes error handling behavior for callers: catch the exception from the
conversion call or asynchronous stream iteration. The exception preserves the
provider's error code and message, plus the parameter supplied by an `error`
event. Missing or null codes fall back to `server_error`.

The exception has no HTTP status because SSE failure events do not supply an
HTTP error status. Callers that classify failures by status must apply their own
mapping from the preserved error details.

Normal `response.completed` and `response.incomplete` conversion is unchanged.
This release does not add EOF validation or change Chat Completions converters.
