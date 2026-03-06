import { BodyFrame, PoseFrame } from "./types";
/**
 * Build a per-frame body coordinate system to make scoring tolerant to
 * camera yaw/roll and user body translation.
 */
export declare function toBodyFrame(frame: PoseFrame): BodyFrame;
/**
 * World-space vertical (Y) of a body-frame landmark for shrug detection invariant to torso lean.
 * position_world = origin + scale * (bf[0]*axes.x + bf[1]*axes.y + bf[2]*axes.z).
 */
export declare function bodyLandmarkWorldY(fr: BodyFrame, idx: number): number;
//# sourceMappingURL=bodyFrame.d.ts.map