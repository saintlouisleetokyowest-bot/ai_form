import { Fault, FeatureSequence, FrameFeatures, RealtimeFault, TemplateBundle } from "./types";
interface ScoreOptions {
    strictWeight?: number;
    tolerantWeight?: number;
    strictThreshold?: number;
    tolerantThreshold?: number;
    /** Weight for key features (arm raise, symmetry, torso, hip); default 16 for finer grading. */
    keyWeight?: number;
    /** Z-threshold for key features; default 0.8 so small deviations already penalize. */
    keyThreshold?: number;
}
export declare function scoreAgainstTemplate(seq: FeatureSequence, template: TemplateBundle, options?: ScoreOptions): {
    score: number;
    faults: Fault[];
    strictPenalty: number;
    tolerantPenalty: number;
};
/** Optional rest shoulder elevation (arms-down baseline) for shrug detection. */
export interface RestShoulderElev {
    left: number;
    right: number;
    /** Torso height (shoulder-center to hip-center) when rest was recorded; needed so elev uses fixed denominator. */
    torsoH_rest: number;
}
/**
 * Score a single frame against the template at the given phase.
 * When restShoulderElev + currentTorsoH are provided, shoulder_elev uses fixed rest torso height
 * so that shrugging (shoulder up) actually increases the measured elevation.
 * Returns faults sorted by zScore descending.
 */
export declare function scoreFrame(frame: FrameFeatures, template: TemplateBundle, phase: number, options?: ScoreOptions & {
    restShoulderElev?: RestShoulderElev;
    currentTorsoH?: number;
}): {
    faults: RealtimeFault[];
    frameScore: number;
};
export {};
//# sourceMappingURL=score.d.ts.map