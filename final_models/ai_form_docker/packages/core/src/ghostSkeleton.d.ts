import { BodyFrame, GhostUpperBody, Landmark, TemplateBundle } from "./types";
export type { GhostUpperBody };
export interface PhaseState {
    phase: number;
    direction: "up" | "down" | "idle";
    peakAngle: number;
}
/**
 * Online phase estimator (unchanged from previous version).
 */
export declare function estimateOnlinePhase(armRaiseAngle: number, prev: PhaseState): PhaseState;
export declare const INITIAL_PHASE_STATE: PhaseState;
/**
 * Compute upper-body ghost via full 3D body-frame construction + 2×3 affine
 * projection to image space.
 *
 * Key improvements over the earlier 2×2 version:
 * - The elbow bend is a 3D rotation about the upper-arm axis, producing a
 *   forearm that extends *forward* (+z in body-frame) rather than staying
 *   flat in the coronal plane.
 * - A 2×3 affine (estimated from 3 non-coplanar reference vectors including
 *   a z-reference landmark like the nose) maps body-frame (x,y,z) → image.
 */
export declare function computeGhostUpperBody(imageLandmarks: Landmark[], bodyFrame: BodyFrame, template: TemplateBundle, phase: number): GhostUpperBody | null;
//# sourceMappingURL=ghostSkeleton.d.ts.map