import { Landmark, Vec3 } from "./types";
import { median, norm, sub } from "./math";

const VIS_THRESHOLD = 0.35;

function visible(lm?: Landmark): boolean {
  return !!lm && (lm.visibility ?? 1) >= VIS_THRESHOLD;
}

export function estimateScale(landmarks: Landmark[], origin: Vec3): number {
  const dists: number[] = [];

  const shoulderL = landmarks[11];
  const shoulderR = landmarks[12];
  if (visible(shoulderL) && visible(shoulderR)) {
    dists.push(norm(sub([shoulderL.x, shoulderL.y, shoulderL.z], [shoulderR.x, shoulderR.y, shoulderR.z])));
  }

  const hipL = landmarks[23];
  const hipR = landmarks[24];
  if (visible(hipL) && visible(hipR)) {
    dists.push(norm(sub([hipL.x, hipL.y, hipL.z], [hipR.x, hipR.y, hipR.z])));
  }

  if (visible(shoulderL) && visible(shoulderR) && visible(hipL) && visible(hipR)) {
    const shoulderCenter: Vec3 = [
      (shoulderL.x + shoulderR.x) / 2,
      (shoulderL.y + shoulderR.y) / 2,
      (shoulderL.z + shoulderR.z) / 2,
    ];
    const hipCenter: Vec3 = [
      (hipL.x + hipR.x) / 2,
      (hipL.y + hipR.y) / 2,
      (hipL.z + hipR.z) / 2,
    ];
    dists.push(norm(sub(shoulderCenter, hipCenter)));
  }

  if (dists.length === 0) {
    // fallback: distance to origin of first visible point
    const first = landmarks.find((lm) => visible(lm));
    if (first) dists.push(norm(sub([first.x, first.y, first.z], origin)));
  }

  const s = median(dists);
  // avoid extremely small scales that blow up values
  return Math.min(Math.max(s, 0.15), 2.5);
}
