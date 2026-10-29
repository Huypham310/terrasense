import { describe, expect, it } from "vitest";
import { LABEL_COLORS, percent, prettyLabel } from "./format";

describe("prettyLabel", () => {
  it("splits camel case class names", () => {
    expect(prettyLabel("HerbaceousVegetation")).toBe("Herbaceous Vegetation");
    expect(prettyLabel("SeaLake")).toBe("Sea Lake");
  });

  it("leaves single words alone", () => {
    expect(prettyLabel("Forest")).toBe("Forest");
  });
});

describe("percent", () => {
  it("formats with one decimal by default", () => {
    expect(percent(0.9819)).toBe("98.2%");
  });

  it("respects the digits argument", () => {
    expect(percent(0.98185, 2)).toBe("98.19%");
    expect(percent(1, 0)).toBe("100%");
  });
});

describe("LABEL_COLORS", () => {
  it("has a color for every EuroSAT class", () => {
    expect(Object.keys(LABEL_COLORS)).toHaveLength(10);
  });
});
