// Typed client for the TerraSense FastAPI service.
// Shapes mirror api/app/schemas.py — keep them in sync when the API changes.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface ClassScore {
  label: string;
  probability: number;
}

export interface Prediction {
  id: number;
  filename: string;
  label: string;
  confidence: number;
  top_k: ClassScore[];
  model_name: string;
  latency_ms: number;
  created_at: string;
}

export interface PredictionPage {
  items: Prediction[];
  total: number;
  limit: number;
  offset: number;
}

export interface ModelInfo {
  model_name: string;
  class_names: string[];
  image_size: number;
  test_accuracy: number | null;
}

export interface Stats {
  total_predictions: number;
  by_label: { label: string; count: number; avg_confidence: number }[];
  avg_latency_ms: number | null;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    // fetch chỉ throw khi không kết nối được (API chưa chạy, sai URL, CORS...)
    throw new ApiError(0, `Can't reach the API at ${API_URL}. Is it running?`);
  }
  if (!res.ok) {
    // FastAPI puts the message in `detail`
    const body = await res.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : res.statusText;
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export function predict(file: Blob, filename: string): Promise<Prediction> {
  const form = new FormData();
  form.append("file", file, filename);
  return request<Prediction>("/predict", { method: "POST", body: form });
}

export function listPredictions(params: { limit: number; offset: number; label?: string }) {
  const query = new URLSearchParams({ limit: String(params.limit), offset: String(params.offset) });
  if (params.label) query.set("label", params.label);
  return request<PredictionPage>(`/predictions?${query}`);
}

export function deletePrediction(id: number): Promise<void> {
  return request<void>(`/predictions/${id}`, { method: "DELETE" });
}

export const getModelInfo = () => request<ModelInfo>("/model");
export const getStats = () => request<Stats>("/stats");
