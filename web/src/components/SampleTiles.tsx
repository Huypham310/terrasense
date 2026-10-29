import { prettyLabel } from "@/lib/format";

// one tile per class from the held-out test split, served from /public/samples
export const SAMPLE_CLASSES = [
  "AnnualCrop",
  "Forest",
  "HerbaceousVegetation",
  "Highway",
  "Industrial",
  "Pasture",
  "PermanentCrop",
  "Residential",
  "River",
  "SeaLake",
];

interface Props {
  disabled: boolean;
  onPick: (name: string) => void;
}

export default function SampleTiles({ disabled, onPick }: Props) {
  return (
    <div>
      <p className="text-sm text-zinc-500">No tile handy? Try one from the test set:</p>
      <div className="mt-3 grid grid-cols-5 gap-3">
        {SAMPLE_CLASSES.map((name) => (
          <button
            key={name}
            type="button"
            disabled={disabled}
            onClick={() => onPick(name)}
            title={prettyLabel(name)}
            className="group overflow-hidden rounded-lg border border-zinc-200 bg-white disabled:opacity-50"
          >
            {/* eslint-disable-next-line @next/next/no-img-element -- 3 KB tiles, next/image is overkill */}
            <img
              src={`/samples/${name}.jpg`}
              alt={prettyLabel(name)}
              className="pixelated aspect-square w-full transition group-hover:scale-105"
            />
          </button>
        ))}
      </div>
    </div>
  );
}
