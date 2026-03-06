import { BodyFrame, GhostUpperBody, Landmark, TemplateBundle, Vec3 } from "./types";
import { clamp, cross, norm, normalize, scale as vecScale, sub } from "./math";

export type { GhostUpperBody };

export interface PhaseState {
  phase: number;
  direction: "up" | "down" | "idle";
  peakAngle: number;
}

const REST_ANGLE = 10;
const PEAK_ANGLE = 85;
const ACTIVE_THRESHOLD = 18;

function toRad(deg: number): number {
  return (deg * Math.PI) / 180;
}

/** Online phase estimator (0..99 from raise angle). */
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

type Mat3 = [Vec3, Vec3, Vec3]; // three column vectors

function det3(m: Mat3): number {
  const [a, b, c] = m;
  return (
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
    b[0] * (a[1] * c[2] - a[2] * c[1]) +
    c[0] * (a[1] * b[2] - a[2] * b[1])
  );
}

/** Returns M⁻¹ as three ROW vectors: result[row][col] = M⁻¹[row][col]. */
function inv3(m: Mat3): Mat3 | null {
  const d = det3(m);
  if (Math.abs(d) < 1e-10) return null;
  const [a, b, c] = m;
  const id = 1 / d;
  return [
    [(b[1] * c[2] - b[2] * c[1]) * id, (b[2] * c[0] - b[0] * c[2]) * id, (b[0] * c[1] - b[1] * c[0]) * id],
    [(a[2] * c[1] - a[1] * c[2]) * id, (a[0] * c[2] - a[2] * c[0]) * id, (a[1] * c[0] - a[0] * c[1]) * id],
    [(a[1] * b[2] - a[2] * b[1]) * id, (a[2] * b[0] - a[0] * b[2]) * id, (a[0] * b[1] - a[1] * b[0]) * id],
  ];
}

function mul23(row0: Vec3, row1: Vec3, v: Vec3): [number, number] {
  return [
    row0[0] * v[0] + row0[1] * v[1] + row0[2] * v[2],
    row1[0] * v[0] + row1[1] * v[1] + row1[2] * v[2],
  ];
}

function rotateAround(v: Vec3, k: Vec3, theta: number): Vec3 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const kxv = cross(k, v);
  const kdv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
  return [
    v[0] * c + kxv[0] * s + k[0] * kdv * (1 - c),
    v[1] * c + kxv[1] * s + k[1] * kdv * (1 - c),
    v[2] * c + kxv[2] * s + k[2] * kdv * (1 - c),
  ];
}

/** Upper-body ghost: 3D body-frame + 2×3 affine to image (elbow bend in 3D, z from e.g. nose). */
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

  const lh = imageLandmarks[23];
  const rh = imageLandmarks[24];
  const bf = bodyFrame.bodyLandmarks;
  const vis = bodyFrame.visibility;

  const bfHipC: Vec3 = [
    (bf[23][0] + bf[24][0]) / 2,
    (bf[23][1] + bf[24][1]) / 2,
    (bf[23][2] + bf[24][2]) / 2,
  ];
  const bfShoulderC: Vec3 = [
    (bf[11][0] + bf[12][0]) / 2,
    (bf[11][1] + bf[12][1]) / 2,
    (bf[11][2] + bf[12][2]) / 2,
  ];

  const imgShoulderC: [number, number] = [(ls.x + rs.x) / 2, (ls.y + rs.y) / 2];
  const hipVisL = !!(lh && (lh.visibility ?? 0) > 0.3);
  const hipVisR = !!(rh && (rh.visibility ?? 0) > 0.3);

  const imgHipC: [number, number] = (hipVisL && hipVisR)
    ? [(lh!.x + rh!.x) / 2, (lh!.y + rh!.y) / 2]
    : [imgShoulderC[0], imgShoulderC[1] + Math.abs(ls.x - rs.x) * 0.8];

  let v1_bf: Vec3, v1_img: [number, number];
  if (hipVisR) {
    v1_bf = sub(bf[24], bfHipC);
    v1_img = [rh!.x - imgHipC[0], rh!.y - imgHipC[1]];
  } else {
    v1_bf = sub(bf[12], bfShoulderC);
    v1_img = [rs.x - imgShoulderC[0], rs.y - imgShoulderC[1]];
  }

  const v2_bf: Vec3 = sub(bfShoulderC, bfHipC);
  let v2_img: [number, number] = [
    imgShoulderC[0] - imgHipC[0],
    imgShoulderC[1] - imgHipC[1],
  ];

  const p = clamp(Math.round(phase), 0, 99);
  const idealLeanDeg = template.features["torso_lean"]?.mean[p] ?? 5;
  const idealLeanStd = template.features["torso_lean"]?.std[p] ?? 5;
  const maxLeanRad = toRad(idealLeanDeg + idealLeanStd);
  const v2Len = Math.sqrt(v2_img[0] ** 2 + v2_img[1] ** 2);
  if (v2Len > 1e-6) {
    const curAngle = Math.atan2(v2_img[0], -v2_img[1]);
    const clamped = clamp(curAngle, -maxLeanRad, maxLeanRad);
    v2_img[0] = Math.sin(clamped) * v2Len;
    v2_img[1] = -Math.cos(clamped) * v2Len;
  }

  let v3_bf: Vec3, v3_img: [number, number];
  const nose = imageLandmarks[0];
  const noseVis = nose && (nose.visibility ?? 0) > 0.3;
  if (noseVis) {
    v3_bf = sub(bf[0], bfHipC);
    v3_img = [nose!.x - imgHipC[0], nose!.y - imgHipC[1]];
  } else {
    const perpLen = v2Len * 0.15;
    v3_bf = [0, 0, 1];
    v3_img = [-v2_img[1] / v2Len * perpLen, v2_img[0] / v2Len * perpLen];
  }

  const M_bf: Mat3 = [v1_bf, v2_bf, v3_bf];
  const M_bf_inv = inv3(M_bf);
  if (!M_bf_inv) return null;

  const Arow0: Vec3 = [
    v1_img[0] * M_bf_inv[0][0] + v2_img[0] * M_bf_inv[1][0] + v3_img[0] * M_bf_inv[2][0],
    v1_img[0] * M_bf_inv[0][1] + v2_img[0] * M_bf_inv[1][1] + v3_img[0] * M_bf_inv[2][1],
    v1_img[0] * M_bf_inv[0][2] + v2_img[0] * M_bf_inv[1][2] + v3_img[0] * M_bf_inv[2][2],
  ];
  const Arow1: Vec3 = [
    v1_img[1] * M_bf_inv[0][0] + v2_img[1] * M_bf_inv[1][0] + v3_img[1] * M_bf_inv[2][0],
    v1_img[1] * M_bf_inv[0][1] + v2_img[1] * M_bf_inv[1][1] + v3_img[1] * M_bf_inv[2][1],
    v1_img[1] * M_bf_inv[0][2] + v2_img[1] * M_bf_inv[1][2] + v3_img[1] * M_bf_inv[2][2],
  ];

  const project3 = (bx: number, by: number, bz: number): { x: number; y: number } => {
    const d: Vec3 = [bx - bfHipC[0], by - bfHipC[1], bz - bfHipC[2]];
    const [ix, iy] = mul23(Arow0, Arow1, d);
    return { x: imgHipC[0] + ix, y: imgHipC[1] + iy };
  };

  const raiseL = toRad(template.features["arm_raise_left"]?.mean[p] ?? 10);
  const raiseR = toRad(template.features["arm_raise_right"]?.mean[p] ?? 10);
  const elbowLAngle = toRad(template.features["elbow_angle_left"]?.mean[p] ?? 170);
  const elbowRAngle = toRad(template.features["elbow_angle_right"]?.mean[p] ?? 170);
  const shoulderHW = Math.abs(bf[12][0] - bf[11][0]) / 2 || 0.5;
  const torsoH = bfShoulderC[1] - bfHipC[1] || 1.0;
  const visTh = 0.25;

  const upperLFromUser = vis[13] >= visTh ? norm(sub(bf[13], bf[11])) : 0;
  const forearmLFromUser = (vis[13] >= visTh && vis[15] >= visTh) ? norm(sub(bf[15], bf[13])) : 0;
  const upperRFromUser = vis[14] >= visTh ? norm(sub(bf[14], bf[12])) : 0;
  const forearmRFromUser = (vis[14] >= visTh && vis[16] >= visTh) ? norm(sub(bf[16], bf[14])) : 0;

  const upperLLen = upperLFromUser > 0 ? upperLFromUser : (upperRFromUser > 0 ? upperRFromUser : shoulderHW * 1.5);
  const forearmLLen = forearmLFromUser > 0 ? forearmLFromUser : (forearmRFromUser > 0 ? forearmRFromUser : shoulderHW * 1.4);
  const upperRLen = upperRFromUser > 0 ? upperRFromUser : (upperLFromUser > 0 ? upperLFromUser : shoulderHW * 1.5);
  const forearmRLen = forearmRFromUser > 0 ? forearmRFromUser : (forearmLFromUser > 0 ? forearmLFromUser : shoulderHW * 1.4);

  const armsDownPhaseThreshold = 22;
  const restShoulderY = bfHipC[1] + torsoH;
  const shoulderCenterY = (bf[11][1] + bf[12][1]) / 2;
  const maxElevL = (template.features["shoulder_elev_left"]?.mean[p] ?? 0.05) + (template.features["shoulder_elev_left"]?.std[p] ?? 0.05);
  const maxElevR = (template.features["shoulder_elev_right"]?.mean[p] ?? 0.05) + (template.features["shoulder_elev_right"]?.std[p] ?? 0.05);
  const correctMaxYLeft = bfHipC[1] + torsoH * (1 + maxElevL);
  const correctMaxYRight = bfHipC[1] + torsoH * (1 + maxElevR);
  const idealLeftShoulderY = p < armsDownPhaseThreshold
    ? shoulderCenterY
    : clamp(bf[11][1], restShoulderY, correctMaxYLeft);
  const idealRightShoulderY = p < armsDownPhaseThreshold
    ? shoulderCenterY
    : clamp(bf[12][1], restShoulderY, correctMaxYRight);

  const idealHipL: Vec3 = [bf[23][0], bf[23][1], bf[23][2]];
  const idealHipR: Vec3 = [bf[24][0], bf[24][1], bf[24][2]];
  const idealSL: Vec3 = [-shoulderHW, idealLeftShoulderY, 0];
  const idealSR: Vec3 = [shoulderHW, idealRightShoulderY, 0];

  const upperDirL: Vec3 = [-Math.sin(raiseL), -Math.cos(raiseL), 0];
  const upperDirR: Vec3 = [Math.sin(raiseR), -Math.cos(raiseR), 0];

  const idealLE: Vec3 = [
    idealSL[0] + upperDirL[0] * upperLLen,
    idealSL[1] + upperDirL[1] * upperLLen,
    0,
  ];
  const idealRE: Vec3 = [
    idealSR[0] + upperDirR[0] * upperRLen,
    idealSR[1] + upperDirR[1] * upperRLen,
    0,
  ];

  const bendL = Math.PI - elbowLAngle;
  const bendAxisL = normalize(cross(upperDirL, [0, 0, -1]));
  const forearmDirL: Vec3 = norm(cross(upperDirL, [0, 0, -1])) > 1e-6
    ? rotateAround(upperDirL, bendAxisL, bendL)
    : upperDirL;

  const idealLW: Vec3 = [
    idealLE[0] + forearmDirL[0] * forearmLLen,
    idealLE[1] + forearmDirL[1] * forearmLLen,
    idealLE[2] + forearmDirL[2] * forearmLLen,
  ];

  const bendR = Math.PI - elbowRAngle;
  const bendAxisR = normalize(cross(upperDirR, [0, 0, -1]));
  const forearmDirR: Vec3 = norm(cross(upperDirR, [0, 0, -1])) > 1e-6
    ? rotateAround(upperDirR, bendAxisR, bendR)
    : upperDirR;

  const idealRW: Vec3 = [
    idealRE[0] + forearmDirR[0] * forearmRLen,
    idealRE[1] + forearmDirR[1] * forearmRLen,
    idealRE[2] + forearmDirR[2] * forearmRLen,
  ];

  const landmarks: Array<{ x: number; y: number }> = new Array(33);
  landmarks[11] = project3(idealSL[0], idealSL[1], idealSL[2]);
  landmarks[12] = project3(idealSR[0], idealSR[1], idealSR[2]);
  landmarks[13] = project3(idealLE[0], idealLE[1], idealLE[2]);
  landmarks[14] = project3(idealRE[0], idealRE[1], idealRE[2]);
  landmarks[15] = project3(idealLW[0], idealLW[1], idealLW[2]);
  landmarks[16] = project3(idealRW[0], idealRW[1], idealRW[2]);
  landmarks[23] = project3(idealHipL[0], idealHipL[1], idealHipL[2]);
  landmarks[24] = project3(idealHipR[0], idealHipR[1], idealHipR[2]);

  return { landmarks };
}
