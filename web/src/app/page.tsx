"use client";

import { useEffect, useRef, useState } from "react";
import ResultCard from "@/components/ResultCard";
import SampleTiles from "@/components/SampleTiles";
import { ApiError, predict, predictBatch, type Prediction } from "@/lib/api";
import { percent, prettyLabel } from "@/lib/format";

const MAX_BATCH = 16;

type TileResult = {
  filename: string;
  preview: string;
  prediction: Prediction | null;
};

export default function ClassifyPage() {
  const [tiles, setTiles] = useState<TileResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  // id of the latest request; slower older responses must not overwrite a newer drop
  const latestRequest = useRef(0);

  // free every blob URL when the tiles change or the page unmounts
  useEffect(() => {
    return () => {
      for (const t of tiles) URL.revokeObjectURL(t.preview);
    };
  }, [tiles]);

  async function classifyMany(files: { blob: Blob; filename: string }[]) {
    if (files.length === 0) return;
    if (files.length > MAX_BATCH) {
      setError(`At most ${MAX_BATCH} tiles at once`);
      return;
    }
    const requestId = ++latestRequest.current;
    setError(null);
    setLoading(true);
    // object URLs so the grid shows up before the API answers
    const next: TileResult[] = files.map((f) => ({
      filename: f.filename,
      preview: URL.createObjectURL(f.blob),
      prediction: null,
    }));
    setTiles(next);
    try {
      const predictions =
        files.length === 1
          ? [await predict(files[0].blob, files[0].filename)]
          : await predictBatch(files);
      if (requestId !== latestRequest.current) return;
      setTiles(
        next.map((t, i) => ({
          ...t,
          prediction: predictions[i] ?? null,
        })),
      );
    } catch (err) {
      if (requestId === latestRequest.current) {
        setError(err instanceof ApiError ? err.message : "Something went wrong");
      }
    } finally {
      if (requestId === latestRequest.current) setLoading(false);
    }
  }

  async function classifySample(name: string) {
    const blob = await (await fetch(`/samples/${name}.jpg`)).blob();
    await classifyMany([{ blob, filename: `${name}.jpg` }]);
  }

  function filesFromList(list: FileList | null) {
    if (!list) return [];
    return Array.from(list).map((f) => ({ blob: f, filename: f.name }));
  }

  function onDrop(event: React.DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    void classifyMany(filesFromList(event.dataTransfer.files));
  }

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl font-semibold">Classify a satellite tile</h1>
        <p className="mt-1 text-zinc-500">
          Upload one or more Sentinel-2 RGB tiles; the model predicts each land-use class.
        </p>
      </section>

      <div className="grid gap-8 md:grid-cols-2">
        <div className="space-y-6">
          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`flex h-48 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed text-sm transition ${
              dragging
                ? "border-emerald-500 bg-emerald-50"
                : "border-zinc-300 bg-white hover:border-zinc-400"
            }`}
          >
            <span className="font-medium">Drop images here or click to browse</span>
            <span className="mt-1 text-zinc-500">
              PNG, JPEG, TIFF or WebP · up to {MAX_BATCH} at once · 5 MB each
            </span>
            <input
              type="file"
              accept="image/png,image/jpeg,image/tiff,image/webp"
              multiple
              className="hidden"
              disabled={loading}
              onChange={(e) => {
                void classifyMany(filesFromList(e.target.files));
                e.target.value = "";
              }}
            />
          </label>

          <SampleTiles disabled={loading} onPick={classifySample} />
        </div>

        <div className="space-y-4">
          {loading && <p className="text-sm text-zinc-500">Classifying...</p>}
          {error && (
            <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {error}
            </p>
          )}
          {tiles.length === 0 && !loading && (
            <p className="text-sm text-zinc-400">Results will show up here.</p>
          )}
          {tiles.length === 1 && tiles[0].prediction && (
            <div className="space-y-4">
              {/* eslint-disable-next-line @next/next/no-img-element -- blob URL preview */}
              <img
                src={tiles[0].preview}
                alt={tiles[0].filename}
                className="pixelated h-48 w-48 rounded-xl border border-zinc-200 object-cover"
              />
              <ResultCard prediction={tiles[0].prediction} />
            </div>
          )}
          {tiles.length > 1 && (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {tiles.map((t) => (
                <div
                  key={t.preview}
                  className="overflow-hidden rounded-xl border border-zinc-200 bg-white"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element -- blob URL preview */}
                  <img
                    src={t.preview}
                    alt={t.filename}
                    className="pixelated h-28 w-full object-cover"
                  />
                  <div className="space-y-0.5 p-2 text-xs">
                    <p className="truncate font-medium text-zinc-700" title={t.filename}>
                      {t.filename}
                    </p>
                    {t.prediction ? (
                      <>
                        <p className="font-semibold">{prettyLabel(t.prediction.label)}</p>
                        <p className="text-zinc-500">
                          {percent(t.prediction.confidence)} · {t.prediction.latency_ms.toFixed(1)}{" "}
                          ms
                        </p>
                      </>
                    ) : (
                      <p className="text-zinc-400">...</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
          {tiles.length === 1 && !tiles[0].prediction && loading && (
            // eslint-disable-next-line @next/next/no-img-element -- blob URL preview
            <img
              src={tiles[0].preview}
              alt={tiles[0].filename}
              className="pixelated h-48 w-48 rounded-xl border border-zinc-200 object-cover"
            />
          )}
        </div>
      </div>
    </div>
  );
}
