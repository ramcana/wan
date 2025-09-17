export function normalizeHeaders(h?: HeadersInit): Headers {
  return new Headers(h ?? {});
}