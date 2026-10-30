"use client";

import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, getModelInfo, getStats, type ModelInfo, type Stats } from "@/lib/api";
import { LABEL_COLORS, percent, prettyLabel } from "@/lib/format";

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-5">
      <p className="text-xs uppercase tracking-wide text-zinc-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
    </div>
  );
}

export default function StatsPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getStats(), getModelInfo()])
      .then(([s, m]) => {
        setStats(s);
        setModel(m);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load stats"));
  }, []);

  const chartData =
    stats?.by_label.map((row) => ({
      name: prettyLabel(row.label),
      label: row.label,
      count: row.count,
      confidence: Math.round(row.avg_confidence * 1000) / 10,
    })) ?? [];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Stats</h1>
        <p className="mt-1 text-zinc-500">What the model has been seeing so far.</p>
      </div>

      {error && <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</p>}

      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard label="Predictions" value={stats ? String(stats.total_predictions) : "-"} />
        <StatCard
          label="Avg latency"
          value={stats?.avg_latency_ms != null ? `${stats.avg_latency_ms.toFixed(1)} ms` : "-"}
        />
        <StatCard
          label={`Test accuracy (${model?.model_name ?? "model"})`}
          value={model?.test_accuracy != null ? percent(model.test_accuracy, 2) : "-"}
        />
      </div>

      <div className="rounded-xl border border-zinc-200 bg-white p-5">
        <p className="mb-4 text-sm font-medium">Predictions per class</p>
        {chartData.length === 0 ? (
          <p className="py-16 text-center text-sm text-zinc-400">No predictions yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chartData} margin={{ left: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              {/* tên class dài nên xoay nghiêng cho đỡ đè nhau; height đủ cho "Herbaceous Vegetation" */}
              <XAxis
                dataKey="name"
                angle={-35}
                textAnchor="end"
                interval={0}
                height={90}
                tick={{ fontSize: 12 }}
              />
              <YAxis allowDecimals={false} />
              <Tooltip
                formatter={(value, _name, item) => [
                  `${value} (avg confidence ${item.payload.confidence}%)`,
                  "Predictions",
                ]}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {chartData.map((row) => (
                  <Cell key={row.label} fill={LABEL_COLORS[row.label] ?? "#10b981"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
