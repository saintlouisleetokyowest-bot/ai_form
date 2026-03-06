import { BodyFrame, CorrectorOutput, Landmark, GHOST_JOINT_IDS, Vec3 } from "./types";
import { sub } from "./math";

const DEBUG_HIP = false;
const ARM_PASSTHROUGH = false;

/** 2×3 affine (same as ghostSkeleton): body-frame → image, avoids shoulder-hip cross / hand flip at oblique view */
type Mat3 = [Vec3, Vec3, Vec3];

function det3(m: Mat3): number {
  const [a, b, c] = m;
  return (
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
    b[0] * (a[1] * c[2] - a[2] * c[1]) +
    c[0] * (a[1] * b[2] - a[2] * b[1])
  );
}

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

/** Project ML-corrected body-frame landmarks to image space via 2×3 affine. Hips 23,24 use userImageLandmarks. */
export function projectCorrectionToImage(
  correction: CorrectorOutput,
  bodyFrame: BodyFrame,
  userImageLandmarks: Landmark[],
): Array<{ x: number; y: number }> {
  const result: Array<{ x: number; y: number }> = new Array(33);

  const ls = userImageLandmarks[11];
  const rs = userImageLandmarks[12];

  if (!ls || !rs) {
    for (let i = 0; i < 33; i++) {
      const lm = userImageLandmarks[i];
      result[i] = lm ? { x: lm.x, y: lm.y } : { x: 0, y: 0 };
    }
    return result;
  }

  const imgShoulderDist = Math.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2);
  const shoulderW = imgShoulderDist;
  const torsoLen = shoulderW * 1.2;
  const imgShoulderCenterY = (ls.y + rs.y) / 2;
  const leftShoulderOnLeft = ls.x < rs.x;

  const lh = userImageLandmarks[23];
  const rh = userImageLandmarks[24];
  const hipVisOk = !!(lh && rh && (lh.visibility ?? 0) > 0.3 && (rh.visibility ?? 0) > 0.3);
  const hipNotTooHigh = hipVisOk && lh!.y > imgShoulderCenterY + shoulderW * 0.2;
  const useUserHip = hipVisOk && hipNotTooHigh;

  const imgShoulderC: [number, number] = [(ls.x + rs.x) / 2, (ls.y + rs.y) / 2];
  const hipVisL = !!(lh && (lh.visibility ?? 0) > 0.3);
  const hipVisR = !!(rh && (rh.visibility ?? 0) > 0.3);

  const imgHipC: [number, number] = (hipVisL && hipVisR)
    ? [(lh!.x + rh!.x) / 2, (lh!.y + rh!.y) / 2]
    : [imgShoulderC[0], imgShoulderC[1] + Math.abs(ls.x - rs.x) * 0.8];

  const bf = bodyFrame.bodyLandmarks;
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

  // Build 2×3 affine: body-frame (x,y,z) → image (x,y)
  let v1_bf: Vec3, v1_img: [number, number];
  if (hipVisR) {
    v1_bf = sub(bf[24], bfHipC);
    v1_img = [rh!.x - imgHipC[0], rh!.y - imgHipC[1]];
  } else {
    v1_bf = sub(bf[12], bfShoulderC);
    v1_img = [rs.x - imgShoulderC[0], rs.y - imgShoulderC[1]];
  }

  const v2_bf: Vec3 = sub(bfShoulderC, bfHipC);
  const v2_img: [number, number] = [
    imgShoulderC[0] - imgHipC[0],
    imgShoulderC[1] - imgHipC[1],
  ];
  const v2Len = Math.sqrt(v2_img[0] ** 2 + v2_img[1] ** 2);

  let v3_bf: Vec3, v3_img: [number, number];
  const nose = userImageLandmarks[0];
  const noseVis = nose && (nose.visibility ?? 0) > 0.3;
  if (noseVis) {
    v3_bf = sub(bf[0], bfHipC);
    v3_img = [nose!.x - imgHipC[0], nose!.y - imgHipC[1]];
  } else {
    const perpLen = v2Len > 1e-6 ? v2Len * 0.15 : 0.15;
    v3_bf = [0, 0, 1];
    v3_img = v2Len > 1e-6
      ? [-v2_img[1] / v2Len * perpLen, v2_img[0] / v2Len * perpLen]
      : [0, 0];
  }

  const M_bf: Mat3 = [v1_bf, v2_bf, v3_bf];
  const M_bf_inv = inv3(M_bf);

  let project3: ((bx: number, by: number, bz: number) => { x: number; y: number }) | null = null;
  if (M_bf_inv) {
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
    project3 = (bx: number, by: number, bz: number) => {
      const d: Vec3 = [bx - bfHipC[0], by - bfHipC[1], bz - bfHipC[2]];
      const [ix, iy] = mul23(Arow0, Arow1, d);
      return { x: imgHipC[0] + ix, y: imgHipC[1] + iy };
    };
  }

  const aClassSet = new Set<number>(GHOST_JOINT_IDS);
  const armIds = new Set([11, 12, 13, 14, 15, 16]);

  for (let i = 0; i < 33; i++) {
    if (ARM_PASSTHROUGH && armIds.has(i)) {
      const lm = userImageLandmarks[i];
      result[i] = lm ? { x: lm.x, y: lm.y } : { x: 0, y: 0 };
    } else if (aClassSet.has(i) && correction.confidence[i] > 0.1 && project3) {
      const cx = correction.correctedLandmarks[i * 3];
      const cy = correction.correctedLandmarks[i * 3 + 1];
      const cz = correction.correctedLandmarks[i * 3 + 2] ?? 0;
      result[i] = project3(cx, cy, cz);
    } else {
      const lm = userImageLandmarks[i];
      result[i] = lm ? { x: lm.x, y: lm.y } : { x: 0, y: 0 };
    }
  }

  // Hips 23,24: always from userImageLandmarks when valid
  if (useUserHip && lh && rh) {
    result[23] = { x: lh.x, y: lh.y };
    result[24] = { x: rh.x, y: rh.y };
  } else {
    const halfW = shoulderW * 0.4;
    result[23] = leftShoulderOnLeft
      ? { x: imgHipC[0] - halfW, y: imgHipC[1] }
      : { x: imgHipC[0] + halfW, y: imgHipC[1] };
    result[24] = leftShoulderOnLeft
      ? { x: imgHipC[0] + halfW, y: imgHipC[1] }
      : { x: imgHipC[0] - halfW, y: imgHipC[1] };
  }

  if (DEBUG_HIP && result[23] && result[24]) {
    // eslint-disable-next-line no-console
    console.log("hip", { 23: result[23], 24: result[24], useUserHip, hipCenterY: imgHipC[1] });
  }

  return result;
}
