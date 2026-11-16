import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ApiError, predict, predictBatch, type Prediction } from "@/lib/api";
import ClassifyPage from "./page";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, predict: vi.fn(), predictBatch: vi.fn() };
});

const river: Prediction = {
  id: 1,
  filename: "River.jpg",
  label: "River",
  confidence: 0.908,
  top_k: [
    { label: "River", probability: 0.908 },
    { label: "AnnualCrop", probability: 0.012 },
    { label: "SeaLake", probability: 0.011 },
  ],
  model_name: "resnet18",
  latency_ms: 2.5,
  created_at: "2026-10-28T04:15:00Z",
};

const forest: Prediction = {
  ...river,
  id: 2,
  filename: "Forest.jpg",
  label: "Forest",
  confidence: 0.81,
  top_k: [{ label: "Forest", probability: 0.81 }],
};

beforeEach(() => {
  // sample tiles are fetched from /public before being sent to the API
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ blob: async () => new Blob(["jpg"]) }));
  // jsdom doesn't implement object URLs
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.mocked(predict).mockReset();
  vi.mocked(predictBatch).mockReset();
});

it("classifies a sample tile and shows the result", async () => {
  vi.mocked(predict).mockResolvedValue(river);
  render(<ClassifyPage />);

  fireEvent.click(screen.getByRole("button", { name: "River" }));

  expect(await screen.findByText(/90\.8% confident/)).toBeDefined();
  expect(predict).toHaveBeenCalledWith(expect.any(Blob), "River.jpg");
  expect(predictBatch).not.toHaveBeenCalled();
});

it("shows the API error message", async () => {
  vi.mocked(predict).mockRejectedValue(new ApiError(0, "Can't reach the API"));
  render(<ClassifyPage />);

  fireEvent.click(screen.getByRole("button", { name: "Forest" }));

  expect(await screen.findByText("Can't reach the API")).toBeDefined();
});

it("classifies an uploaded file", async () => {
  vi.mocked(predict).mockResolvedValue(river);
  const { container } = render(<ClassifyPage />);
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  const file = new File(["png"], "my_tile.png", { type: "image/png" });

  fireEvent.change(input, { target: { files: [file] } });

  expect(await screen.findByText(/90\.8% confident/)).toBeDefined();
  expect(predict).toHaveBeenCalledWith(file, "my_tile.png");
});

it("sends several files through /predict/batch and shows a results grid", async () => {
  vi.mocked(predictBatch).mockResolvedValue([river, forest]);
  const { container } = render(<ClassifyPage />);
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  const files = [
    new File(["a"], "River.jpg", { type: "image/jpeg" }),
    new File(["b"], "Forest.jpg", { type: "image/jpeg" }),
  ];

  fireEvent.change(input, { target: { files } });

  expect(await screen.findByText("River")).toBeDefined();
  expect(await screen.findByText("Forest")).toBeDefined();
  expect(predictBatch).toHaveBeenCalledTimes(1);
  const arg = vi.mocked(predictBatch).mock.calls[0][0];
  expect(arg).toHaveLength(2);
  expect(arg[0].filename).toBe("River.jpg");
  expect(arg[1].filename).toBe("Forest.jpg");
  expect(predict).not.toHaveBeenCalled();
});
