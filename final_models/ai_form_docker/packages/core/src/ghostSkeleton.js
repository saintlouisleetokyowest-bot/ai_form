import { clamp, cross, norm, normalize, sub } from "./math";
const REST_ANGLE = 10;
const PEAK_ANGLE = 85;
const ACTIVE_THRESHOLD = 18;
function toRad(deg) {
    return (deg * Math.PI) / 180;
}
/**
 * Online phase estimator (unchanged from previous version).
 */
export function estimateOnlinePhase(armRaiseAngle, prev) {
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
export const INITIAL_PHASE_STATE = {
    phase: 0,
    direction: "idle",
    peakAngle: PEAK_ANGLE,
};
function det3(m) {
    const [a, b, c] = m;
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) -
        b[0] * (a[1] * c[2] - a[2] * c[1]) +
        c[0] * (a[1] * b[2] - a[2] * b[1]));
}
/** Returns M⁻¹ as three ROW vectors: result[row][col] = M⁻¹[row][col]. */
function inv3(m) {
    const d = det3(m);
    if (Math.abs(d) < 1e-10)
        return null;
    const [a, b, c] = m;
    const id = 1 / d;
    return [
        // row 0 of M⁻¹
        [(b[1] * c[2] - b[2] * c[1]) * id, (b[2] * c[0] - b[0] * c[2]) * id, (b[0] * c[1] - b[1] * c[0]) * id],
        // row 1
        [(a[2] * c[1] - a[1] * c[2]) * id, (a[0] * c[2] - a[2] * c[0]) * id, (a[1] * c[0] - a[0] * c[1]) * id],
        // row 2
        [(a[1] * b[2] - a[2] * b[1]) * id, (a[2] * b[0] - a[0] * b[2]) * id, (a[0] * b[1] - a[1] * b[0]) * id],
    ];
}
// Multiply 2×3 row (stored as two Vec3 rows) by Vec3 column → [number, number]
function mul23(row0, row1, v) {
    return [
        row0[0] * v[0] + row0[1] * v[1] + row0[2] * v[2],
        row1[0] * v[0] + row1[1] * v[1] + row1[2] * v[2],
    ];
}
// Rodrigues rotation: rotate `v` around unit axis `k` by angle `theta`
function rotateAround(v, k, theta) {
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
export function computeGhostUpperBody(imageLandmarks, bodyFrame, template, phase) {
    const ls = imageLandmarks[11];
    const rs = imageLandmarks[12];
    if (!ls || !rs)
        return null;
    if ((ls.visibility ?? 0) < 0.3 || (rs.visibility ?? 0) < 0.3)
        return null;
    const lh = imageLandmarks[23];
    const rh = imageLandmarks[24];
    const bf = bodyFrame.bodyLandmarks;
    const vis = bodyFrame.visibility;
    // ─── Body-frame anchors (3D) ───
    const bfHipC = [
        (bf[23][0] + bf[24][0]) / 2,
        (bf[23][1] + bf[24][1]) / 2,
        (bf[23][2] + bf[24][2]) / 2,
    ];
    const bfShoulderC = [
        (bf[11][0] + bf[12][0]) / 2,
        (bf[11][1] + bf[12][1]) / 2,
        (bf[11][2] + bf[12][2]) / 2,
    ];
    // ─── Image anchors ───
    const imgShoulderC = [(ls.x + rs.x) / 2, (ls.y + rs.y) / 2];
    const hipVisL = !!(lh && (lh.visibility ?? 0) > 0.3);
    const hipVisR = !!(rh && (rh.visibility ?? 0) > 0.3);
    const imgHipC = (hipVisL && hipVisR)
        ? [(lh.x + rh.x) / 2, (lh.y + rh.y) / 2]
        : [imgShoulderC[0], imgShoulderC[1] + Math.abs(ls.x - rs.x) * 0.8];
    // ─── Build 2×3 affine: body-frame (x,y,z) → image (x,y) ───
    // v1 (x-direction): hip center → hip right
    let v1_bf, v1_img;
    if (hipVisR) {
        v1_bf = sub(bf[24], bfHipC);
        v1_img = [rh.x - imgHipC[0], rh.y - imgHipC[1]];
    }
    else {
        v1_bf = sub(bf[12], bfShoulderC);
        v1_img = [rs.x - imgShoulderC[0], rs.y - imgShoulderC[1]];
    }
    // v2 (y-direction): hip center → shoulder center
    const v2_bf = sub(bfShoulderC, bfHipC);
    let v2_img = [
        imgShoulderC[0] - imgHipC[0],
        imgShoulderC[1] - imgHipC[1],
    ];
    // ─── Clamp torso lean in image space to ideal range ───
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
    // v3 (z-direction): use a landmark with significant z in body-frame.
    // Prefer nose (0); fall back to an ear or synthetic z-axis direction.
    let v3_bf, v3_img;
    const nose = imageLandmarks[0];
    const noseVis = nose && (nose.visibility ?? 0) > 0.3;
    if (noseVis) {
        v3_bf = sub(bf[0], bfHipC);
        v3_img = [nose.x - imgHipC[0], nose.y - imgHipC[1]];
    }
    else {
        // Synthetic fallback: z-axis maps to a small perpendicular component.
        // Perpendicular to v2_img in image (pointing "into screen" → small effect).
        const perpLen = v2Len * 0.15;
        v3_bf = [0, 0, 1]; // unit z in body-frame
        v3_img = [-v2_img[1] / v2Len * perpLen, v2_img[0] / v2Len * perpLen];
    }
    // Solve A_2×3 = M_img × M_bf⁻¹
    // M_bf is 3×3: columns [v1_bf | v2_bf | v3_bf]
    const M_bf = [v1_bf, v2_bf, v3_bf];
    const M_bf_inv = inv3(M_bf);
    if (!M_bf_inv)
        return null;
    // A = M_img_2×3 × M_bf⁻¹_3×3   →   A[i][j] = Σ_k M_img[i][k] · M_bf⁻¹[k][j]
    // inv3 now returns rows, so M_bf_inv[k][j] = M_bf⁻¹[k][j]
    const Arow0 = [
        v1_img[0] * M_bf_inv[0][0] + v2_img[0] * M_bf_inv[1][0] + v3_img[0] * M_bf_inv[2][0],
        v1_img[0] * M_bf_inv[0][1] + v2_img[0] * M_bf_inv[1][1] + v3_img[0] * M_bf_inv[2][1],
        v1_img[0] * M_bf_inv[0][2] + v2_img[0] * M_bf_inv[1][2] + v3_img[0] * M_bf_inv[2][2],
    ];
    const Arow1 = [
        v1_img[1] * M_bf_inv[0][0] + v2_img[1] * M_bf_inv[1][0] + v3_img[1] * M_bf_inv[2][0],
        v1_img[1] * M_bf_inv[0][1] + v2_img[1] * M_bf_inv[1][1] + v3_img[1] * M_bf_inv[2][1],
        v1_img[1] * M_bf_inv[0][2] + v2_img[1] * M_bf_inv[1][2] + v3_img[1] * M_bf_inv[2][2],
    ];
    const project3 = (bx, by, bz) => {
        const d = [bx - bfHipC[0], by - bfHipC[1], bz - bfHipC[2]];
        const [ix, iy] = mul23(Arow0, Arow1, d);
        return { x: imgHipC[0] + ix, y: imgHipC[1] + iy };
    };
    // ─── Template angles for current phase ───
    const raiseL = toRad(template.features["arm_raise_left"]?.mean[p] ?? 10);
    const raiseR = toRad(template.features["arm_raise_right"]?.mean[p] ?? 10);
    const elbowLAngle = toRad(template.features["elbow_angle_left"]?.mean[p] ?? 170);
    const elbowRAngle = toRad(template.features["elbow_angle_right"]?.mean[p] ?? 170);
    // ─── User's arm segment lengths (ideal = same as real skeleton) ───
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
    // ─── Ideal shoulder height (body-frame Y): level when arms down (average of both); clamp to correct range when raised (shrug → ideal stays, hint shows) ───
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
    // ─── Ideal body-frame positions (full 3D) ───
    const idealHipL = [bf[23][0], bf[23][1], bf[23][2]];
    const idealHipR = [bf[24][0], bf[24][1], bf[24][2]];
    const idealSL = [-shoulderHW, idealLeftShoulderY, 0];
    const idealSR = [shoulderHW, idealRightShoulderY, 0];
    // Upper arm directions in body-frame (coronal plane, z=0)
    const upperDirL = [-Math.sin(raiseL), -Math.cos(raiseL), 0];
    const upperDirR = [Math.sin(raiseR), -Math.cos(raiseR), 0];
    const idealLE = [
        idealSL[0] + upperDirL[0] * upperLLen,
        idealSL[1] + upperDirL[1] * upperLLen,
        0,
    ];
    const idealRE = [
        idealSR[0] + upperDirR[0] * upperRLen,
        idealSR[1] + upperDirR[1] * upperRLen,
        0,
    ];
    // Forearm: 3D Rodrigues rotation — bend forward (toward camera = -z in body-frame).
    // Body-frame z-axis points AWAY from camera, so forward = [0,0,-1].
    const bendL = Math.PI - elbowLAngle;
    const bendAxisL = normalize(cross(upperDirL, [0, 0, -1]));
    const forearmDirL = norm(cross(upperDirL, [0, 0, -1])) > 1e-6
        ? rotateAround(upperDirL, bendAxisL, bendL)
        : upperDirL;
    const idealLW = [
        idealLE[0] + forearmDirL[0] * forearmLLen,
        idealLE[1] + forearmDirL[1] * forearmLLen,
        idealLE[2] + forearmDirL[2] * forearmLLen,
    ];
    const bendR = Math.PI - elbowRAngle;
    const bendAxisR = normalize(cross(upperDirR, [0, 0, -1]));
    const forearmDirR = norm(cross(upperDirR, [0, 0, -1])) > 1e-6
        ? rotateAround(upperDirR, bendAxisR, bendR)
        : upperDirR;
    const idealRW = [
        idealRE[0] + forearmDirR[0] * forearmRLen,
        idealRE[1] + forearmDirR[1] * forearmRLen,
        idealRE[2] + forearmDirR[2] * forearmRLen,
    ];
    // ─── Project all ideal joints to image ───
    const landmarks = new Array(33);
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
