import { BodyFrame, CorrectorOutput, Landmark } from "./types";
/** Project ML-corrected body-frame landmarks to image space via 2×3 affine. Hips 23,24 use userImageLandmarks. */
export declare function projectCorrectionToImage(correction: CorrectorOutput, bodyFrame: BodyFrame, userImageLandmarks: Landmark[]): Array<{
    x: number;
    y: number;
}>;
//# sourceMappingURL=correctionSkeleton.d.ts.map