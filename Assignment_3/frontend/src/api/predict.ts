import type { Task, PredictionResponse } from "../types";

export async function runPrediction(
  task: Task,
  image: File,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("task", task);
  formData.append("image", image);

  const response = await fetch("/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => null);
    throw new Error(err?.error ?? `Server error (${response.status})`);
  }

  const data: PredictionResponse = await response.json();

  if (data.error) {
    throw new Error(data.error);
  }

  return data;
}
