export type Vec3 = [number, number, number];

export interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface PoseFrame {
  /** MediaPipe world landmarks in meters. */
  landmarks: Landmark[];
  /** 2D screen landmarks mirrored to the rendered video (optional). */
  imageLandmarks?: Landmark[];
  /** Frame index in the incoming stream. */
  index?: number;
  /** Timestamp in milliseconds. */
  ts?: number;
}

export interface BodyFrame {
  origin: Vec3;
  axes: {
    x: Vec3;
    y: Vec3;
    z: Vec3;
  };
  scale: number;
  /** Landmarks in body frame, already scale-normalized. */
  bodyLandmarks: Vec3[];
  /** Visibility mask preserved from MediaPipe. */
  visibility: number[];
  /** Raw source frame index. */
  index?: number;
  ts?: number;
}

export type FeatureKind = "strict" | "tolerant" | "invariant";

export interface FeatureDefinition {
  id: string;
  kind: FeatureKind;
  description?: string;
}

export interface FeatureSequence {
  /** phase-aligned 0..99 */
  phase: number[];
  values: Record<string, number[]>;
  mask: Record<string, boolean[]>;
}

export interface RepWindow {
  start: number;
  end: number;
  peak?: number;
}

export interface TemplateBand {
  mean: number[];
  std: number[];
  kind: FeatureKind;
}

export interface TemplateBundle {
  phaseCount: number;
  features: Record<string, TemplateBand>;
}

export interface ScoreBreakdown {
  score: number;
  strictPenalty: number;
  tolerantPenalty: number;
  details: Record<string, number>;
}

export interface Fault {
  id: string;
  featureId: string;
  severity: "info" | "warn" | "error";
  message: string;
  joints: number[];
  phaseRange: [number, number];
}

/** Real-time per-frame fault used by the online scoring path. */
export interface RealtimeFault {
  featureId: string;
  severity: "info" | "warn" | "error";
  message: string;
  joints: number[];
  zScore: number;
}

/** Ghost overlay that only covers the upper-body "A-class" joints. */
export interface GhostUpperBody {
  landmarks: Array<{ x: number; y: number }>;
}

/** A-class joint indices whose angles come from the template. */
export const GHOST_JOINT_IDS = [11, 12, 13, 14, 15, 16, 23, 24] as const;

/** Edges drawn for the ideal ghost overlay (upper body). */
export const GHOST_EDGES: [number, number][] = [
  [23, 24],
  [23, 11],
  [24, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

/** Single-frame feature snapshot. */
export interface FrameFeatures {
  values: Record<string, number>;
  mask: Record<string, boolean>;
}

/** ML corrector standard input. */
export interface CorrectorInput {
  bodyLandmarks: Float32Array;
  phase: number;
  visibility: Float32Array;
}

/** ML corrector standard output. */
export interface CorrectorOutput {
  correctedLandmarks: Float32Array;
  confidence: Float32Array;
}

/** Unified interface for real and mock correctors. */
export interface IPoseCorrector {
  readonly ready: boolean;
  load(): Promise<void>;
  run(input: CorrectorInput): CorrectorOutput;
}
