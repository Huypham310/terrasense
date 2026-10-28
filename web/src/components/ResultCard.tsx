import type { Prediction } from "@/lib/api";
import { LABEL_COLORS, percent, prettyLabel } from "@/lib/format";

export default function ResultCard({ prediction }: { prediction: Prediction }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
      <p className="text-xs uppercase tracking-wide text-zinc-500">Prediction</p>
      <p className="mt-1 text-2xl font-semibold">{prettyLabel(prediction.label)}</p>
      <p className="text-sm text-zinc-500">
        {percent(prediction.confidence)} confident · {prediction.latency_ms.toFixed(1)} ms ·{" "}
        {prediction.model_name}
      </p>

      <ul className="mt-5 space-y-3">
        {prediction.top_k.map((score) => (
          <li key={score.label}>
            <div className="flex justify-between text-sm">
              <span>{prettyLabel(score.label)}</span>
              <span className="tabular-nums text-zinc-500">{percent(score.probability)}</span>
            </div>
            {/* thanh ngang tỉ lệ theo xác suất */}
            <div className="mt-1 h-2 rounded-full bg-zinc-100">
              <div
                className="h-2 rounded-full"
                style={{
                  width: `${Math.max(score.probability * 100, 1)}%`,
                  backgroundColor: LABEL_COLORS[score.label] ?? "#10b981",
                }}
              />
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
