import { BodyFrame, FeatureDefinition, FeatureSequence, FrameFeatures, Vec3 } from "./types";
export declare const featureDefinitions: FeatureDefinition[];
export declare function computeFeatureSequence(frames: BodyFrame[]): FeatureSequence;
/**
 * Compute the 9 features for a single body frame.
 * `baseHip` defaults to the frame's own hip center (no drift reference).
 */
export declare function computeFrameFeatures(fr: BodyFrame, baseHip?: Vec3): FrameFeatures;
//# sourceMappingURL=features.d.ts.map