import { Fault, FeatureSequence, FrameFeatures, RealtimeFault, TemplateBand, TemplateBundle } from "./types";
import { clamp } from "./math";

interface ScoreOptions {
  strictWeight?: number;
  tolerantWeight?: number;
  strictThreshold?: number;
  tolerantThreshold?: number;
}

const DEFAULT_OPTIONS: Required<ScoreOptions> = {
  strictWeight: 10,
  tolerantWeight: 5,
  strictThreshold: 1, // z > 1 starts to penalize
  tolerantThreshold: 1.5,
};

function computeExcess(
  values: number[],
  mask: boolean[],
  band: TemplateBand,
  threshold: number
): { meanExcess: number; perPhase: number[] } {
  const excess: number[] = [];
  for (let i = 0; i < values.length; i++) {
    if (!mask[i]) {
      excess.push(0);
      continue;
    }
    const z = Math.abs(values[i] - band.mean[i]) / (band.std[i] + 1e-6);
    excess.push(Math.max(0, z - threshold));
  }
  const meanExcess = excess.reduce((a, b) => a + b, 0) / Math.max(excess.length, 1);
  return { meanExcess, perPhase: excess };
}

function faultMessage(featureId: string): string {
  switch (featureId) {
    case "arm_raise_left":
    case "arm_raise_right":
      return "Arm raise height out of band (too high/low).";
    case "arm_symmetry":
      return "Left/right arms not moving symmetrically.";
    case "torso_lean":
      return "Torso leaning or swinging.";
    case "hip_drift":
      return "Hips drifting forward/back/side.";
    case "elbow_angle_left":
    case "elbow_angle_right":
      return "Elbow bend out of band.";
    case "shoulder_elev_left":
    case "shoulder_elev_right":
      return "Shoulder shrugging.";
    default:
      return "Form deviates from template.";
  }
}

function jointsForFeature(featureId: string): number[] {
  switch (featureId) {
    case "arm_raise_left":
    case "elbow_angle_left":
    case "shoulder_elev_left":
      return [11, 13, 15];
    case "arm_raise_right":
    case "elbow_angle_right":
    case "shoulder_elev_right":
      return [12, 14, 16];
    case "arm_symmetry":
      return [11, 12, 13, 14, 15, 16];
    case "torso_lean":
    case "hip_drift":
      return [11, 12, 23, 24];
    default:
      return [];
  }
}

function findLongestRange(values: number[], minZ = 1): [number, number] {
  let best: [number, number] = [0, 0];
  let current: [number, number] | null = null;
  values.forEach((v, idx) => {
    if (v >= minZ) {
      if (!current) current = [idx, idx];
      else current[1] = idx;
      if (current[1] - current[0] > best[1] - best[0]) best = [current[0], current[1]];
    } else {
      current = null;
    }
  });
  return best;
}

export function scoreAgainstTemplate(
  seq: FeatureSequence,
  template: TemplateBundle,
  options: ScoreOptions = {}
): { score: number; faults: Fault[]; strictPenalty: number; tolerantPenalty: number } {
  const cfg = { ...DEFAULT_OPTIONS, ...options };
  let strictPenalty = 0;
  let tolerantPenalty = 0;
  const faults: Fault[] = [];

  for (const [featureId, band] of Object.entries(template.features)) {
    const values = seq.values[featureId] ?? [];
    const mask = seq.mask[featureId] ?? [];
    if (values.length === 0) continue;

    const threshold = band.kind === "strict" ? cfg.strictThreshold : cfg.tolerantThreshold;
    const { meanExcess, perPhase } = computeExcess(values, mask, band, threshold);
    if (band.kind === "strict") strictPenalty += meanExcess;
    if (band.kind === "tolerant") tolerantPenalty += meanExcess;

    const maxZ = Math.max(...perPhase);
    if (maxZ > 0.5) {
      const phaseRange = findLongestRange(perPhase, maxZ * 0.5);
      faults.push({
        id: `fault_${featureId}`,
        featureId,
        severity: maxZ > 2 ? "error" : maxZ > 1 ? "warn" : "info",
        message: faultMessage(featureId),
        joints: jointsForFeature(featureId),
        phaseRange,
      });
    }
  }

  const strictScore = cfg.strictWeight * strictPenalty;
  const tolerantScore = cfg.tolerantWeight * tolerantPenalty;
  const raw = 100 - strictScore - tolerantScore;
  return {
    score: clamp(raw, 0, 100),
    faults,
    strictPenalty: strictScore,
    tolerantPenalty: tolerantScore,
  };
}

/**
 * Score a single frame against the template at the given phase.
 * Returns faults sorted by zScore descending.
 */
export function scoreFrame(
  frame: FrameFeatures,
  template: TemplateBundle,
  phase: number,
  options: ScoreOptions = {},
): { faults: RealtimeFault[]; frameScore: number } {
  const cfg = { ...DEFAULT_OPTIONS, ...options };
  const p = clamp(Math.round(phase), 0, 99);
  const faults: RealtimeFault[] = [];
  let totalPenalty = 0;

  for (const [featureId, band] of Object.entries(template.features)) {
    const value = frame.values[featureId];
    const vis = frame.mask[featureId];
    if (value == null || !vis) continue;

    const mean = band.mean[p] ?? 0;
    const std = band.std[p] ?? 1;
    const z = Math.abs(value - mean) / (std + 1e-6);
    const threshold = band.kind === "strict" ? cfg.strictThreshold : cfg.tolerantThreshold;
    const excess = Math.max(0, z - threshold);
    const weight = band.kind === "strict" ? cfg.strictWeight : cfg.tolerantWeight;
    totalPenalty += weight * excess;

    if (z > 1) {
      faults.push({
        featureId,
        severity: z > 2.5 ? "error" : z > 1.5 ? "warn" : "info",
        message: faultMessage(featureId),
        joints: jointsForFeature(featureId),
        zScore: z,
      });
    }
  }

  faults.sort((a, b) => b.zScore - a.zScore);
  return { faults, frameScore: clamp(100 - totalPenalty, 0, 100) };
}
