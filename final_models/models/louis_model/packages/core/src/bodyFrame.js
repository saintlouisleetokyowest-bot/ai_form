import { cross, dot, normalize, scale, sub } from "./math";
import { estimateScale } from "./scaling";
const VIS_THRESHOLD = 0.35;
function isVisible(l) {
    return typeof l === "number" ? l >= VIS_THRESHOLD : true;
}
function pickPairMean(ids, landmarks) {
    const [a, b] = ids;
    const pa = landmarks[a];
    const pb = landmarks[b];
    if (!pa || !pb)
        return null;
    if (!isVisible(pa.visibility) || !isVisible(pb.visibility))
        return null;
    return [(pa.x + pb.x) / 2, (pa.y + pb.y) / 2, (pa.z + pb.z) / 2];
}
function fallbackOrigin(landmarks) {
    for (const lm of landmarks) {
        if (isVisible(lm.visibility))
            return [lm.x, lm.y, lm.z];
    }
    return [0, 0, 0];
}
/**
 * Build a per-frame body coordinate system to make scoring tolerant to
 * camera yaw/roll and user body translation.
 */
export function toBodyFrame(frame) {
    const lms = frame.landmarks;
    const hipCenter = pickPairMean([23, 24], lms);
    const shoulderCenter = pickPairMean([11, 12], lms);
    const origin = hipCenter ?? shoulderCenter ?? fallbackOrigin(lms);
    // x axis (left -> right)
    const hipLeft = lms[23];
    const hipRight = lms[24];
    let xAxis = null;
    if (hipLeft && hipRight && isVisible(hipLeft.visibility) && isVisible(hipRight.visibility)) {
        xAxis = normalize(sub([hipRight.x, hipRight.y, hipRight.z], [hipLeft.x, hipLeft.y, hipLeft.z]));
    }
    else if (shoulderCenter) {
        const shoulderLeft = lms[11];
        const shoulderRight = lms[12];
        if (shoulderLeft && shoulderRight && isVisible(shoulderLeft.visibility) && isVisible(shoulderRight.visibility)) {
            xAxis = normalize(sub([shoulderRight.x, shoulderRight.y, shoulderRight.z], [shoulderLeft.x, shoulderLeft.y, shoulderLeft.z]));
        }
    }
    if (!xAxis)
        xAxis = [1, 0, 0];
    // y axis (hip -> shoulder)
    let yAxis;
    if (hipCenter && shoulderCenter) {
        yAxis = normalize(sub(shoulderCenter, hipCenter));
    }
    else {
        yAxis = [0, 1, 0];
    }
    // z axis (forward)
    let zAxis = cross(xAxis, yAxis);
    if (dot(zAxis, zAxis) < 1e-6)
        zAxis = [0, 0, 1];
    zAxis = normalize(zAxis);
    // re-orthogonalize x to be perpendicular to y & z
    xAxis = normalize(cross(yAxis, zAxis));
    yAxis = normalize(cross(zAxis, xAxis));
    const scaleVal = estimateScale(lms, origin);
    const bodyLandmarks = lms.map((lm) => {
        const p = [lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0];
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
/**
 * World-space vertical (Y) of a body-frame landmark for shrug detection invariant to torso lean.
 * position_world = origin + scale * (bf[0]*axes.x + bf[1]*axes.y + bf[2]*axes.z).
 */
export function bodyLandmarkWorldY(fr, idx) {
    const bf = fr.bodyLandmarks[idx];
    if (!bf)
        return 0;
    const s = fr.scale || 1;
    const ax = fr.axes.x;
    const ay = fr.axes.y;
    const az = fr.axes.z;
    return fr.origin[1] + s * (bf[0] * ax[1] + bf[1] * ay[1] + bf[2] * az[1]);
}
