import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { deletePrediction, listPredictions, type Prediction } from "@/lib/api";
import HistoryPage from "./page";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, listPredictions: vi.fn(), deletePrediction: vi.fn() };
});

function row(id: number, label = "Forest"): Prediction {
  return {
    id,
    filename: `tile_${id}.png`,
    label,
    confidence: 0.9,
    top_k: [],
    model_name: "resnet18",
    latency_ms: 2,
    created_at: "2026-10-28T04:15:00Z",
  };
}

afterEach(() => {
  cleanup();
  vi.mocked(listPredictions).mockReset();
  vi.mocked(deletePrediction).mockReset();
});

it("lists predictions and pages forward", async () => {
  vi.mocked(listPredictions).mockImplementation(async ({ offset }) => ({
    items: offset === 0 ? [row(12), row(11)] : [row(2)],
    total: 12,
    limit: 10,
    offset,
  }));
  render(<HistoryPage />);

  expect(await screen.findByText("tile_12.png")).toBeDefined();
  expect(screen.getByText("Page 1 of 2")).toBeDefined();

  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  expect(await screen.findByText("tile_2.png")).toBeDefined();
  expect(listPredictions).toHaveBeenLastCalledWith({ limit: 10, offset: 10, label: undefined });
});

it("filters by label and resets to the first page", async () => {
  vi.mocked(listPredictions).mockResolvedValue({ items: [row(1)], total: 1, limit: 10, offset: 0 });
  render(<HistoryPage />);
  await screen.findByText("tile_1.png");

  fireEvent.change(screen.getByRole("combobox"), { target: { value: "River" } });

  await waitFor(() =>
    expect(listPredictions).toHaveBeenLastCalledWith({ limit: 10, offset: 0, label: "River" }),
  );
});

it("deletes a row and reloads", async () => {
  vi.mocked(listPredictions)
    .mockResolvedValueOnce({ items: [row(5), row(4)], total: 2, limit: 10, offset: 0 })
    .mockResolvedValueOnce({ items: [row(4)], total: 1, limit: 10, offset: 0 });
  vi.mocked(deletePrediction).mockResolvedValue(undefined);
  render(<HistoryPage />);
  await screen.findByText("tile_5.png");

  fireEvent.click(screen.getAllByRole("button", { name: "Delete" })[0]);

  await waitFor(() => expect(screen.queryByText("tile_5.png")).toBeNull());
  expect(deletePrediction).toHaveBeenCalledWith(5);
  expect(screen.getByText("1 predictions stored")).toBeDefined();
});

it("shows an empty state", async () => {
  vi.mocked(listPredictions).mockResolvedValue({ items: [], total: 0, limit: 10, offset: 0 });
  render(<HistoryPage />);
  expect(await screen.findByText(/Nothing here yet/)).toBeDefined();
});
