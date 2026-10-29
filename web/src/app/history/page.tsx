"use client";

import { useEffect, useState } from "react";
import { SAMPLE_CLASSES } from "@/components/SampleTiles";
import { ApiError, deletePrediction, listPredictions, type PredictionPage } from "@/lib/api";
import { LABEL_COLORS, formatTime, percent, prettyLabel } from "@/lib/format";

const PAGE_SIZE = 10;

export default function HistoryPage() {
  const [page, setPage] = useState<PredictionPage | null>(null);
  const [offset, setOffset] = useState(0);
  const [label, setLabel] = useState("");
  const [error, setError] = useState<string | null>(null);
  // bump to refetch the current page (e.g. after a delete)
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    // ignore responses that arrive after the user already changed page/filter
    let stale = false;
    listPredictions({ limit: PAGE_SIZE, offset, label: label || undefined })
      .then((data) => {
        if (stale) return;
        setPage(data);
        setError(null);
      })
      .catch((err) => {
        if (!stale) setError(err instanceof ApiError ? err.message : "Failed to load history");
      });
    return () => {
      stale = true;
    };
  }, [offset, label, reloadKey]);

  async function remove(id: number) {
    try {
      await deletePrediction(id);
      // xóa dòng cuối của trang cuối thì lùi về trang trước
      if (page && page.items.length === 1 && offset > 0) setOffset(offset - PAGE_SIZE);
      else setReloadKey((k) => k + 1);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Delete failed");
    }
  }

  const total = page?.total ?? 0;
  const lastPage = Math.max(Math.ceil(total / PAGE_SIZE) - 1, 0);
  const current = Math.floor(offset / PAGE_SIZE);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">History</h1>
          <p className="mt-1 text-zinc-500">{total} predictions stored</p>
        </div>
        <select
          value={label}
          onChange={(e) => {
            setLabel(e.target.value);
            setOffset(0); // đổi filter thì quay về trang đầu
          }}
          className="rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm"
        >
          <option value="">All labels</option>
          {SAMPLE_CLASSES.map((name) => (
            <option key={name} value={name}>
              {prettyLabel(name)}
            </option>
          ))}
        </select>
      </div>

      {error && <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</p>}

      <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white">
        <table className="w-full text-left text-sm">
          <thead className="bg-zinc-50 text-xs uppercase tracking-wide text-zinc-500">
            <tr>
              <th className="px-4 py-3">File</th>
              <th className="px-4 py-3">Label</th>
              <th className="px-4 py-3">Confidence</th>
              <th className="px-4 py-3">Latency</th>
              <th className="px-4 py-3">When</th>
              <th className="px-4 py-3" />
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100">
            {page?.items.map((p) => (
              <tr key={p.id}>
                <td className="max-w-48 truncate px-4 py-3" title={p.filename}>
                  {p.filename}
                </td>
                <td className="px-4 py-3">
                  <span className="inline-flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: LABEL_COLORS[p.label] }} />
                    {prettyLabel(p.label)}
                  </span>
                </td>
                <td className="px-4 py-3 tabular-nums">{percent(p.confidence)}</td>
                <td className="px-4 py-3 tabular-nums">{p.latency_ms.toFixed(1)} ms</td>
                <td className="px-4 py-3 text-zinc-500">{formatTime(p.created_at)}</td>
                <td className="px-4 py-3 text-right">
                  <button onClick={() => remove(p.id)} className="text-xs text-zinc-400 hover:text-red-600">
                    Delete
                  </button>
                </td>
              </tr>
            ))}
            {page && page.items.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-10 text-center text-zinc-400">
                  Nothing here yet. Classify a tile first.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between text-sm">
        <button
          disabled={current === 0}
          onClick={() => setOffset(offset - PAGE_SIZE)}
          className="rounded-lg border border-zinc-300 bg-white px-3 py-1.5 disabled:opacity-40"
        >
          Previous
        </button>
        <span className="text-zinc-500">
          Page {current + 1} of {lastPage + 1}
        </span>
        <button
          disabled={current >= lastPage}
          onClick={() => setOffset(offset + PAGE_SIZE)}
          className="rounded-lg border border-zinc-300 bg-white px-3 py-1.5 disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </div>
  );
}
