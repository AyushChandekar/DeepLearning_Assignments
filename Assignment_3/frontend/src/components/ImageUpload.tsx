import { useCallback, useRef, useState, type DragEvent } from "react";

interface Props {
  file: File | null;
  onFileSelect: (file: File) => void;
  disabled: boolean;
}

export default function ImageUpload({ file, onFileSelect, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFile = useCallback(
    (f: File) => {
      onFileSelect(f);
      const url = URL.createObjectURL(f);
      setPreview(url);
    },
    [onFileSelect],
  );

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  return (
    <section className="flex-1">
      <h2 className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-500">
        Upload Image
      </h2>

      <div
        onClick={() => !disabled && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed transition-all
          ${preview ? "p-3" : "py-8"}
          ${
            dragging
              ? "border-purple-500 bg-purple-500/10"
              : "border-slate-700 hover:border-purple-500/50"
          }
          ${disabled ? "pointer-events-none opacity-40" : "cursor-pointer"}
        `}
      >
        {preview ? (
          <div className="flex flex-col items-center gap-2">
            <img
              src={preview}
              alt="Preview"
              className="max-h-40 rounded-md object-contain"
            />
            <p className="text-xs text-slate-500">
              {file?.name}
              <span className="ml-1 text-slate-600">
                ({((file?.size ?? 0) / 1024).toFixed(1)} KB)
              </span>
            </p>
            <span className="text-[10px] text-purple-400">Click to change</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <svg
              className="size-8 text-slate-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
            <p className="text-xs text-slate-400">
              Drag & drop or click to browse
            </p>
            <p className="text-[10px] text-slate-600">
              JPG, PNG, BMP, WEBP
            </p>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/bmp,image/webp"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />
      </div>
    </section>
  );
}
