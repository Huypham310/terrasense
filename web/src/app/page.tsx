"use client";

import { useState } from "react";
import ResultCard from "@/components/ResultCard";
import SampleTiles from "@/components/SampleTiles";
import { ApiError, predict, type Prediction } from "@/lib/api";

export default function ClassifyPage() {
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);

  async function classify(file: Blob, filename: string) {
    setError(null);
    setResult(null);
    setLoading(true);
    // object URL để hiện ảnh ngay, không cần chờ API
    setPreview(URL.createObjectURL(file));
    try {
      setResult(await predict(file, filename));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function classifySample(name: string) {
    const blob = await (await fetch(`/samples/${name}.jpg`)).blob();
    await classify(blob, `${name}.jpg`);
  }

  function onDrop(event: React.DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) classify(file, file.name);
  }

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl font-semibold">Classify a satellite tile</h1>
        <p className="mt-1 text-zinc-500">
          Upload a Sentinel-2 RGB tile and the model predicts its land-use class.
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
              dragging ? "border-emerald-500 bg-emerald-50" : "border-zinc-300 bg-white hover:border-zinc-400"
            }`}
          >
            <span className="font-medium">Drop an image here or click to browse</span>
            <span className="mt-1 text-zinc-500">PNG, JPEG, TIFF or WebP, up to 5 MB</span>
            <input
              type="file"
              accept="image/png,image/jpeg,image/tiff,image/webp"
              className="hidden"
              disabled={loading}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) classify(file, file.name);
              }}
            />
          </label>

          <SampleTiles disabled={loading} onPick={classifySample} />
        </div>

        <div className="space-y-4">
          {preview && (
            // eslint-disable-next-line @next/next/no-img-element -- blob URL preview
            <img
              src={preview}
              alt="Uploaded tile"
              className="pixelated h-48 w-48 rounded-xl border border-zinc-200 object-cover"
            />
          )}
          {loading && <p className="text-sm text-zinc-500">Classifying...</p>}
          {error && (
            <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</p>
          )}
          {result && <ResultCard prediction={result} />}
          {!preview && !loading && (
            <p className="text-sm text-zinc-400">Results will show up here.</p>
          )}
        </div>
      </div>
    </div>
  );
}
