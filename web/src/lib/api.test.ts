import { afterEach, describe, expect, it, vi } from "vitest";
import { API_URL, ApiError, deletePrediction, listPredictions, predict } from "./api";

function mockFetch(response: Partial<Response> | Error) {
  const fn =
    response instanceof Error
      ? vi.fn().mockRejectedValue(response)
      : vi.fn().mockResolvedValue({ ok: true, status: 200, statusText: "OK", ...response });
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api client", () => {
  it("sends the file as multipart form data", async () => {
    const fetch = mockFetch({ json: async () => ({ id: 1, label: "River" }) });
    const result = await predict(new Blob(["x"], { type: "image/png" }), "tile.png");

    expect(result.label).toBe("River");
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe(`${API_URL}/predict`);
    expect(init.method).toBe("POST");
    expect((init.body as FormData).get("file")).toBeInstanceOf(Blob);
  });

  it("builds the history query string", async () => {
    const fetch = mockFetch({ json: async () => ({ items: [], total: 0, limit: 10, offset: 20 }) });
    await listPredictions({ limit: 10, offset: 20, label: "Forest" });
    expect(fetch.mock.calls[0][0]).toBe(`${API_URL}/predictions?limit=10&offset=20&label=Forest`);
  });

  it("omits the label filter when empty", async () => {
    const fetch = mockFetch({ json: async () => ({ items: [], total: 0, limit: 10, offset: 0 }) });
    await listPredictions({ limit: 10, offset: 0 });
    expect(fetch.mock.calls[0][0]).not.toContain("label=");
  });

  it("surfaces FastAPI's detail message on errors", async () => {
    mockFetch({
      ok: false,
      status: 415,
      statusText: "Unsupported Media Type",
      json: async () => ({ detail: "Unsupported file type 'text/plain'" }),
    });
    await expect(predict(new Blob(["x"]), "a.txt")).rejects.toMatchObject({
      status: 415,
      message: "Unsupported file type 'text/plain'",
    });
  });

  it("falls back to statusText when the body isn't json", async () => {
    mockFetch({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => {
        throw new SyntaxError("bad json");
      },
    });
    await expect(listPredictions({ limit: 1, offset: 0 })).rejects.toMatchObject({
      status: 500,
      message: "Internal Server Error",
    });
  });

  it("turns network failures into a friendly ApiError", async () => {
    mockFetch(new TypeError("Failed to fetch"));
    const err = await predict(new Blob(["x"]), "a.png").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
    expect(err.message).toContain("Can't reach the API");
  });

  it("handles 204 responses without parsing a body", async () => {
    const json = vi.fn();
    mockFetch({ status: 204, json });
    await expect(deletePrediction(3)).resolves.toBeUndefined();
    expect(json).not.toHaveBeenCalled();
  });
});
