// Small display helpers shared by the pages.

// "HerbaceousVegetation" -> "Herbaceous Vegetation"
export function prettyLabel(label: string): string {
  return label.replace(/([a-z])([A-Z])/g, "$1 $2");
}

export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// màu cố định cho từng class để chart với badge khớp nhau
export const LABEL_COLORS: Record<string, string> = {
  AnnualCrop: "#eab308",
  Forest: "#15803d",
  HerbaceousVegetation: "#84cc16",
  Highway: "#6b7280",
  Industrial: "#9333ea",
  Pasture: "#65a30d",
  PermanentCrop: "#ca8a04",
  Residential: "#dc2626",
  River: "#2563eb",
  SeaLake: "#0e7490",
};
