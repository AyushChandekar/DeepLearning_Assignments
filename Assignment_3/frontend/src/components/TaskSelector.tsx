import type { Task, TaskInfo } from "../types";

const TASKS: TaskInfo[] = [
  { key: "detection", label: "Detection", icon: "crosshair", description: "Locate & label objects" },
  { key: "classification", label: "Classify", icon: "tag", description: "Identify image class" },
  { key: "pose", label: "Pose", icon: "person-standing", description: "Body keypoints" },
  { key: "obb", label: "OBB", icon: "box", description: "Oriented boxes" },
];

function TaskIcon({ icon }: { icon: string }) {
  const cls = "size-5";
  const props = { className: cls, fill: "none" as const, viewBox: "0 0 24 24", stroke: "currentColor", strokeWidth: 1.5 };

  switch (icon) {
    case "crosshair":
      return (
        <svg {...props}>
          <circle cx="12" cy="12" r="9" />
          <line x1="12" y1="2" x2="12" y2="6" /><line x1="12" y1="18" x2="12" y2="22" />
          <line x1="2" y1="12" x2="6" y2="12" /><line x1="18" y1="12" x2="22" y2="12" />
        </svg>
      );
    case "tag":
      return (
        <svg {...props}>
          <path d="M9.568 3H5.25A2.25 2.25 0 003 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 005.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 009.568 3z" />
          <path d="M6 6h.008v.008H6V6z" />
        </svg>
      );
    case "person-standing":
      return (
        <svg {...props}>
          <circle cx="12" cy="4" r="2" />
          <path d="M12 6v6m0 0l-3 6m3-6l3 6m-6-8l-2-1m8 1l2-1" />
        </svg>
      );
    case "box":
      return (
        <svg {...props}>
          <path d="M6 3l12 3v12l-12-3V3z" />
          <path d="M6 3l4 2v12l-4-2V3z" opacity={0.5} />
        </svg>
      );
    default:
      return null;
  }
}

interface Props {
  selected: Task | null;
  onSelect: (task: Task) => void;
  disabled: boolean;
}

export default function TaskSelector({ selected, onSelect, disabled }: Props) {
  return (
    <section>
      <h2 className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-500">
        Select Task
      </h2>
      <div className="grid grid-cols-2 gap-2">
        {TASKS.map((t) => {
          const active = selected === t.key;
          return (
            <button
              key={t.key}
              disabled={disabled}
              onClick={() => onSelect(t.key)}
              className={`group flex items-center gap-2.5 rounded-lg border-2 px-3 py-2.5 text-left transition-all
                ${
                  active
                    ? "border-purple-500 bg-purple-500/15 text-white"
                    : "border-slate-700/60 bg-slate-800/50 text-slate-400 hover:border-purple-500/50 hover:text-slate-200"
                }
                ${disabled ? "pointer-events-none opacity-40" : "cursor-pointer"}
              `}
            >
              <TaskIcon icon={t.icon} />
              <div>
                <span className="text-xs font-semibold leading-none">{t.label}</span>
                <span className="mt-0.5 block text-[10px] text-slate-500 group-hover:text-slate-400">
                  {t.description}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
