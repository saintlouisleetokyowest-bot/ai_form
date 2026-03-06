import { Fault, RealtimeFault } from "./types";

export interface RenderHint {
  joints: number[];
  highlightColor: string;
  label: string;
}

export function renderHintsForFaults(faults: Fault[]): RenderHint[] {
  return faults.map((f) => ({
    joints: f.joints,
    highlightColor: f.severity === "error" ? "#ff4d4f" : "#f5a524",
    label: f.message,
  }));
}

export function renderHintsForRealtimeFaults(faults: RealtimeFault[]): RenderHint[] {
  return faults.map((f) => ({
    joints: f.joints,
    highlightColor: f.severity === "error" ? "#ff4d4f" : f.severity === "warn" ? "#f5a524" : "#94a3b8",
    label: f.message,
  }));
}
