export type Task = "detection" | "classification" | "pose" | "obb";

export interface TaskInfo {
  key: Task;
  label: string;
  icon: string;
  description: string;
}

export interface DetectionDetails {
  detections: number;
  classes: { name: string; conf: string }[];
}

export interface ClassificationDetails {
  prediction: string;
  confidence: string;
  top5: { name: string; conf: string }[];
}

export interface PoseDetails {
  poses_detected: number;
}

export interface ObbDetails {
  oriented_boxes: number;
}

export type PredictionDetails =
  | DetectionDetails
  | ClassificationDetails
  | PoseDetails
  | ObbDetails;

export interface PredictionResponse {
  image: string;
  task: Task;
  details: PredictionDetails;
  error?: string;
}
