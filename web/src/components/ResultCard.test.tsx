import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import type { Prediction } from "@/lib/api";
import ResultCard from "./ResultCard";

afterEach(cleanup);

const prediction: Prediction = {
  id: 7,
  filename: "tile.png",
  label: "PermanentCrop",
  confidence: 0.919,
  top_k: [
    { label: "PermanentCrop", probability: 0.919 },
    { label: "AnnualCrop", probability: 0.013 },
    { label: "Forest", probability: 0.011 },
  ],
  model_name: "resnet18",
  latency_ms: 2.24,
  created_at: "2026-10-28T04:15:59Z",
};

it("shows the predicted class with readable name", () => {
  render(<ResultCard prediction={prediction} />);
  expect(screen.getByText("Permanent Crop", { selector: "p" })).toBeDefined();
  expect(screen.getByText(/91\.9% confident/)).toBeDefined();
  expect(screen.getByText(/2\.2 ms/)).toBeDefined();
});

it("renders one bar per top-k entry", () => {
  render(<ResultCard prediction={prediction} />);
  expect(screen.getAllByRole("listitem")).toHaveLength(3);
  expect(screen.getByText("1.3%")).toBeDefined();
});
