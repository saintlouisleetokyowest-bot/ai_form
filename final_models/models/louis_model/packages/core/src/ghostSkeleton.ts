import { BodyFrame, GhostUpperBody, Landmark, TemplateBundle } from "./types";
import { angleBetween, clamp, degrees } from "./math";

export type { GhostUpperBody };

export interface PhaseState {
  phase: number;
  direction: "up" | "down" | "idle";
  peakAngle: number;
}

const REST_ANGLE = 10;
const PEAK_ANGLE = 85;
const ACTIVE_THRESHOLD = 18;

function dist2(a: { x: number; y: number }, b: { x: number; y: number }): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function toRad(deg: number): number {
  return (deg * Math.PI) / 180;
}

/**
 * Online phase estimator (unchanged from previous version).
 */
export function estimateOnlinePhase(
  armRaiseAngle: number,
  prev: PhaseState,
): PhaseState {
  const angle = clamp(armRaiseAngle, 0, 120);

  if (angle < ACTIVE_THRESHOLD && prev.direction === "idle") {
    return { phase: 0, direction: "idle", peakAngle: prev.peakAngle };
  }

  let direction = prev.direction;
  if (angle >= ACTIVE_THRESHOLD && direction === "idle") {
    direction = "up";
  }

  if (direction === "up" && angle < prev.peakAngle - 8) {
    direction = "down";
  }

  const peakAngle = direction === "up"
    ? Math.max(prev.peakAngle, angle)
    : prev.peakAngle;

  if (direction === "down" && angle <= ACTIVE_THRESHOLD) {
    return { phase: 99, direction: "idle", peakAngle: PEAK_ANGLE };
  }

  const progress = clamp((angle - REST_ANGLE) / (peakAngle - REST_ANGLE), 0, 1);
  const phase = direction === "up"
    ? Math.round(progress * 49)
    : Math.round(50 + (1 - progress) * 49);

  return { phase: clamp(phase, 0, 99), direction, peakAngle };
}

export const INITIAL_PHASE_STATE: PhaseState = {
  phase: 0,
  direction: "idle",
  peakAngle: PEAK_ANGLE,
};

/**
 * Compute upper-body ghost (A-class joints: 11,12,13,14,15,16,23,24).
 * Anchored to user's joint positions and limb lengths; angles from template.
 * B-class joints are NOT included — the caller simply omits them.
 */
export function computeGhostUpperBody(
  imageLandmarks: Landmark[],
  bodyFrame: BodyFrame,
  template: TemplateBundle,
  phase: number,
): GhostUpperBody | null {
  const ls = imageLandmarks[11];
  const rs = imageLandmarks[12];
  if (!ls || !rs) return null;
  if ((ls.visibility ?? 0) < 0.3 || (rs.visibility ?? 0) < 0.3) return null;

  const le = imageLandmarks[13];
  const re = imageLandmarks[14];
  const lw = imageLandmarks[15];
  const rw = imageLandmarks[16];
  const lh = imageLandmarks[23];
  const rh = imageLandmarks[24];

  const shoulderWidth = dist2(ls, rs);
  const defaultUpper = shoulderWidth * 0.75;
  const defaultForearm = shoulderWidth * 0.7;

  const upperL = le && (le.visibility ?? 0) > 0.3 ? dist2(ls, le) : defaultUpper;
  const forearmL = le && lw && (lw.visibility ?? 0) > 0.3 ? dist2(le, lw) : defaultForearm;
  const upperR = re && (re.visibility ?? 0) > 0.3 ? dist2(rs, re) : defaultUpper;
  const forearmR = re && rw && (rw.visibility ?? 0) > 0.3 ? dist2(re, rw) : defaultForearm;

  const p = clamp(Math.round(phase), 0, 99);

  const raiseL = toRad(template.features["arm_raise_left"]?.mean[p] ?? 10);
  const raiseR = toRad(template.features["arm_raise_right"]?.mean[p] ?? 10);
  const elbowL = toRad(template.features["elbow_angle_left"]?.mean[p] ?? 170);
  const elbowR = toRad(template.features["elbow_angle_right"]?.mean[p] ?? 170);

  // Torso: ideal lean from template
  const idealLean = toRad(template.features["torso_lean"]?.mean[p] ?? 5);

  // Shoulder elevation from template (symmetric, slight rise at peak)
  const shElevL = template.features["shoulder_elev_left"]?.mean[p] ?? 0.05;
  const shElevR = template.features["shoulder_elev_right"]?.mean[p] ?? 0.05;

  // --- Hips: use user's actual hip positions ---
  const ghostHipL = lh && (lh.visibility ?? 0) > 0.3
    ? { x: lh.x, y: lh.y } : { x: ls.x, y: ls.y + shoulderWidth * 0.8 };
  const ghostHipR = rh && (rh.visibility ?? 0) > 0.3
    ? { x: rh.x, y: rh.y } : { x: rs.x, y: rs.y + shoulderWidth * 0.8 };

  // --- Ideal shoulder center from hip center + ideal lean ---
  const hipCx = (ghostHipL.x + ghostHipR.x) / 2;
  const hipCy = (ghostHipL.y + ghostHipR.y) / 2;
  const torsoLen = dist2(
    { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 },
    { x: hipCx, y: hipCy },
  ) || shoulderWidth * 0.8;

  const shoulderCx = hipCx + Math.sin(idealLean) * torsoLen;
  const shoulderCy = hipCy - Math.cos(idealLean) * torsoLen;

  // Camera perspective: inherit user's left-right shoulder height difference
  // so ghost shoulder tilt matches the projected tilt of truly level shoulders.
  const perspectiveDiffY = ls.y - rs.y;

  const halfSW = shoulderWidth / 2;
  const ghostLS = {
    x: shoulderCx + halfSW,
    y: shoulderCy - shElevL * shoulderWidth + perspectiveDiffY / 2,
  };
  const ghostRS = {
    x: shoulderCx - halfSW,
    y: shoulderCy - shElevR * shoulderWidth - perspectiveDiffY / 2,
  };

  // --- Left arm ---
  const dirLx = Math.sin(raiseL);
  const dirLy = Math.cos(raiseL);
  const ghostLE = {
    x: ghostLS.x + dirLx * upperL,
    y: ghostLS.y + dirLy * upperL,
  };

  const bendL = Math.PI - elbowL;
  const fDirLx = dirLx * Math.cos(bendL) - dirLy * Math.sin(bendL);
  const fDirLy = dirLx * Math.sin(bendL) + dirLy * Math.cos(bendL);
  const ghostLW = {
    x: ghostLE.x + fDirLx * forearmL,
    y: ghostLE.y + fDirLy * forearmL,
  };

  // --- Right arm ---
  const dirRx = -Math.sin(raiseR);
  const dirRy = Math.cos(raiseR);
  const ghostRE = {
    x: ghostRS.x + dirRx * upperR,
    y: ghostRS.y + dirRy * upperR,
  };

  const bendR = Math.PI - elbowR;
  const fDirRx = dirRx * Math.cos(bendR) + dirRy * Math.sin(bendR);
  const fDirRy = -dirRx * Math.sin(bendR) + dirRy * Math.cos(bendR);
  const ghostRW = {
    x: ghostRE.x + fDirRx * forearmR,
    y: ghostRE.y + fDirRy * forearmR,
  };

  // Build sparse array — only A-class indices populated
  const landmarks: Array<{ x: number; y: number }> = new Array(33);
  landmarks[11] = ghostLS;
  landmarks[12] = ghostRS;
  landmarks[13] = ghostLE;
  landmarks[14] = ghostRE;
  landmarks[15] = ghostLW;
  landmarks[16] = ghostRW;
  landmarks[23] = ghostHipL;
  landmarks[24] = ghostHipR;

  return { landmarks };
}
