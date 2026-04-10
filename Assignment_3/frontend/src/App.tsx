import { useState } from "react";
import TaskSelector from "./components/TaskSelector";
import ImageUpload from "./components/ImageUpload";
import ResultsPanel from "./components/ResultsPanel";
import { runPrediction } from "./api/predict";
import type { Task, PredictionResponse } from "./types";

export default function App() {
  const [task, setTask] = useState<Task | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canRun = task !== null && file !== null && !loading;

  async function handleRun() {
    if (!task || !file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await runPrediction(task, file);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-slate-950 text-slate-200">
      {/* Compact header */}
      <header className="shrink-0 border-b border-purple-500/30 bg-gradient-to-r from-slate-900 via-purple-950 to-slate-900 px-6 py-3">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold text-white tracking-tight">
            Multi-Task YOLO Vision System
          </h1>
          <span className="hidden text-xs text-slate-500 sm:inline">
            Detection &middot; Classification &middot; Pose &middot; OBB
          </span>
        </div>
      </header>

      {/* Two-column body */}
      <div className="flex min-h-0 flex-1">
        {/* Left panel — controls */}
        <div className="flex w-[420px] shrink-0 flex-col gap-5 overflow-y-auto border-r border-slate-800 p-5">
          <TaskSelector
            selected={task}
            onSelect={setTask}
            disabled={loading}
          />

          <ImageUpload file={file} onFileSelect={setFile} disabled={loading} />

          <button
            disabled={!canRun}
            onClick={handleRun}
            className={`w-full shrink-0 rounded-xl py-3 text-sm font-semibold transition-all
              ${
                canRun
                  ? "bg-gradient-to-r from-purple-600 to-violet-500 text-white shadow-lg shadow-purple-500/25 hover:shadow-purple-500/40 active:scale-[0.99]"
                  : "bg-slate-800 text-slate-500 cursor-not-allowed"
              }
            `}
          >
            {loading ? (
              <span className="inline-flex items-center gap-2">
                <svg
                  className="size-4 animate-spin"
                  viewBox="0 0 24 24"
                  fill="none"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Running inference...
              </span>
            ) : (
              "Run Inference"
            )}
          </button>

          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-300">
              {error}
            </div>
          )}
        </div>

        {/* Right panel — results */}
        <div className="flex flex-1 items-center justify-center overflow-y-auto p-5">
          {result ? (
            <ResultsPanel result={result} />
          ) : (
            <div className="flex flex-col items-center gap-3 text-slate-600">
              <svg
                className="size-16 opacity-40"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={0.8}
              >
                <path d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
              </svg>
              <p className="text-sm">
                {loading
                  ? "Processing..."
                  : "Select a task, upload an image, and run inference"}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
