import { BodyFrame, Landmark, PoseFrame, Vec3 } from "./types";
import { cross, dot, normalize, scale, sub } from "./math";
import { estimateScale } from "./scaling";

const VIS_THRESHOLD = 0.35;

function isVisible(l?: number): boolean {
  return typeof l === "number" ? l >= VIS_THRESHOLD : true;
}

function pickPairMean(ids: [number, number], landmarks: Landmark[]): Vec3 | null {
  const [a, b] = ids;
  const pa = landmarks[a];
  const pb = landmarks[b];
  if (!pa || !pb) return null;
  if (!isVisible(pa.visibility) || !isVisible(pb.visibility)) return null;
  return [(pa.x + pb.x) / 2, (pa.y + pb.y) / 2, (pa.z + pb.z) / 2];
}

function fallbackOrigin(landmarks: Landmark[]): Vec3 {
  for (const lm of landmarks) {
    if (isVisible(lm.visibility)) return [lm.x, lm.y, lm.z];
  }
  return [0, 0, 0];
}

/** Per-frame body frame for scoring (tolerant to camera yaw/roll and translation). */
export function toBodyFrame(frame: PoseFrame): BodyFrame {
  const lms = frame.landmarks;
  const hipCenter = pickPairMean([23, 24], lms);
  const shoulderCenter = pickPairMean([11, 12], lms);
  const origin: Vec3 = hipCenter ?? shoulderCenter ?? fallbackOrigin(lms);

  const hipLeft = lms[23];
  const hipRight = lms[24];
  let xAxis: Vec3 | null = null;
  if (hipLeft && hipRight && isVisible(hipLeft.visibility) && isVisible(hipRight.visibility)) {
    xAxis = normalize(sub([hipRight.x, hipRight.y, hipRight.z], [hipLeft.x, hipLeft.y, hipLeft.z]));
  } else if (shoulderCenter) {
    const shoulderLeft = lms[11];
    const shoulderRight = lms[12];
    if (shoulderLeft && shoulderRight && isVisible(shoulderLeft.visibility) && isVisible(shoulderRight.visibility)) {
      xAxis = normalize(sub([shoulderRight.x, shoulderRight.y, shoulderRight.z], [shoulderLeft.x, shoulderLeft.y, shoulderLeft.z]));
    }
  }
  if (!xAxis) xAxis = [1, 0, 0];

  let yAxis: Vec3;
  if (hipCenter && shoulderCenter) {
    yAxis = normalize(sub(shoulderCenter, hipCenter));
  } else {
    yAxis = [0, 1, 0];
  }

  let zAxis = cross(xAxis, yAxis);
  if (dot(zAxis, zAxis) < 1e-6) zAxis = [0, 0, 1];
  zAxis = normalize(zAxis);

  xAxis = normalize(cross(yAxis, zAxis));
  yAxis = normalize(cross(zAxis, xAxis));

  const scaleVal = estimateScale(lms, origin);
  const bodyLandmarks: Vec3[] = lms.map((lm) => {
    const p: Vec3 = [lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0];
    const rel = sub(p, origin);
    const bx = dot(rel, xAxis);
    const by = dot(rel, yAxis);
    const bz = dot(rel, zAxis);
    return scale([bx, by, bz], 1 / (scaleVal || 1));
  });

  const visibility = lms.map((lm) => lm?.visibility ?? 0);

  return {
    origin,
    axes: { x: xAxis, y: yAxis, z: zAxis },
    scale: scaleVal,
    bodyLandmarks,
    visibility,
    index: frame.index,
    ts: frame.ts,
  };
}
/** World-space Y of a body-frame landmark (shrug detection, invariant to torso lean). */
export function bodyLandmarkWorldY(fr: BodyFrame, idx: number): number {
  const bf = fr.bodyLandmarks[idx];
  if (!bf) return 0;
  const s = fr.scale || 1;
  const ax = fr.axes.x;
  const ay = fr.axes.y;
  const az = fr.axes.z;
  return fr.origin[1] + s * (bf[0] * ax[1] + bf[1] * ay[1] + bf[2] * az[1]);
}

