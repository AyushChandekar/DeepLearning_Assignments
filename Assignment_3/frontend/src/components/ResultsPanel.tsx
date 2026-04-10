import type {
  PredictionResponse,
  DetectionDetails,
  ClassificationDetails,
  PoseDetails,
  ObbDetails,
} from "../types";

interface Props {
  result: PredictionResponse;
}

function isDetection(d: unknown): d is DetectionDetails {
  return (d as DetectionDetails).detections !== undefined;
}
function isClassification(d: unknown): d is ClassificationDetails {
  return (d as ClassificationDetails).prediction !== undefined;
}
function isPose(d: unknown): d is PoseDetails {
  return (d as PoseDetails).poses_detected !== undefined;
}
function isObb(d: unknown): d is ObbDetails {
  return (d as ObbDetails).oriented_boxes !== undefined;
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full bg-purple-500/15 px-2 py-0.5 text-[11px] font-medium text-purple-300">
      {children}
    </span>
  );
}

function DetailsView({ result }: Props) {
  const { details } = result;

  if (isDetection(details)) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-white">Detection</span>
          <Badge>{details.detections} object(s)</Badge>
        </div>
        {details.classes.length > 0 && (
          <div className="max-h-32 overflow-y-auto rounded border border-slate-700 text-xs">
            <table className="w-full text-left">
              <thead className="sticky top-0 bg-slate-800 text-[10px] uppercase text-slate-500">
                <tr>
                  <th className="px-3 py-1.5">Class</th>
                  <th className="px-3 py-1.5 text-right">Conf</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/50">
                {details.classes.map((c, i) => (
                  <tr key={i} className="text-slate-300">
                    <td className="px-3 py-1">{c.name}</td>
                    <td className="px-3 py-1 text-right font-mono text-purple-300">{c.conf}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  }

  if (isClassification(details)) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-white">Classification</span>
          <Badge>{details.prediction}</Badge>
          <span className="text-[11px] font-mono text-purple-300">{details.confidence}</span>
        </div>
        <div className="max-h-32 overflow-y-auto rounded border border-slate-700 text-xs">
          <table className="w-full text-left">
            <thead className="sticky top-0 bg-slate-800 text-[10px] uppercase text-slate-500">
              <tr>
                <th className="px-3 py-1.5">#</th>
                <th className="px-3 py-1.5">Class</th>
                <th className="px-3 py-1.5 text-right">Conf</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {details.top5.map((c, i) => (
                <tr
                  key={i}
                  className={i === 0 ? "text-white bg-purple-500/10" : "text-slate-300"}
                >
                  <td className="px-3 py-1 text-slate-500">{i + 1}</td>
                  <td className="px-3 py-1">{c.name}</td>
                  <td className="px-3 py-1 text-right font-mono text-purple-300">{c.conf}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  if (isPose(details)) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-white">Pose Estimation</span>
        <Badge>{details.poses_detected} pose(s) detected</Badge>
      </div>
    );
  }

  if (isObb(details)) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-white">Oriented Bounding Boxes</span>
        <Badge>{details.oriented_boxes} box(es)</Badge>
      </div>
    );
  }

  return (
    <pre className="text-[10px] text-slate-400">
      {JSON.stringify(details, null, 2)}
    </pre>
  );
}

export default function ResultsPanel({ result }: Props) {
  const taskLabel =
    {
      detection: "Detection",
      classification: "Classification",
      pose: "Pose Estimation",
      obb: "Oriented Bounding Boxes",
    }[result.task] ?? result.task;

  return (
    <div className="flex h-full w-full flex-col gap-3">
      <h2 className="shrink-0 text-xs font-medium uppercase tracking-wider text-slate-500">
        Results &mdash; {taskLabel}
      </h2>

      <div className="min-h-0 flex-1 overflow-hidden rounded-lg border border-slate-700 bg-slate-800/40">
        <img
          src={`data:image/jpeg;base64,${result.image}`}
          alt={`${taskLabel} result`}
          className="h-full w-full object-contain"
        />
      </div>

      <div className="shrink-0 rounded-lg border border-slate-700 bg-slate-800/40 p-3">
        <DetailsView result={result} />
      </div>
    </div>
  );
}
