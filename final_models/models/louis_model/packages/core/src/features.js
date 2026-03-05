import { angleBetween, degrees, sub } from "./math";
const Y_AXIS = [0, 1, 0];
/**
 * Global knee valgus: compare distance between knees vs. distance between hips
 * in body-frame X (left-right). When knees are narrower than hips, this returns
 * a positive value in [0, 1] roughly equal to (hipWidth - kneeWidth) / hipWidth.
 */
function kneeValgusGlobal(fr) {
    const hipL = landmark(fr, 23);
    const hipR = landmark(fr, 24);
    const kneeL = landmark(fr, 25);
    const kneeR = landmark(fr, 26);
    const hipsVisible = visible(fr, 23) && visible(fr, 24);
    const kneesVisible = visible(fr, 25) && visible(fr, 26);
    if (!hipsVisible || !kneesVisible)
        return 0;
    const hipWidth = Math.abs(hipR[0] - hipL[0]);
    const kneeWidth = Math.abs(kneeR[0] - kneeL[0]);
    const denom = hipWidth + 1e-6;
    if (denom < 1e-6)
        return 0;
    const ratio = kneeWidth / denom;
    return Math.max(0, 1 - ratio);
}
export const featureDefinitions = [
    { id: "arm_raise_left", kind: "strict", description: "Left upper-arm abduction vs torso" },
    { id: "arm_raise_right", kind: "strict", description: "Right upper-arm abduction vs torso" },
    { id: "arm_symmetry", kind: "strict", description: "Left-right raise angle difference" },
    { id: "torso_lean", kind: "strict", description: "Torso lean relative to vertical" },
    { id: "hip_drift", kind: "strict", description: "Hip drift in x/z" },
    { id: "knee_valgus_left", kind: "strict", description: "Left knee valgus (inward cave)" },
    { id: "knee_valgus_right", kind: "strict", description: "Right knee valgus (inward cave)" },
    { id: "elbow_angle_left", kind: "tolerant", description: "Left elbow bend" },
    { id: "elbow_angle_right", kind: "tolerant", description: "Right elbow bend" },
    { id: "shoulder_elev_left", kind: "tolerant", description: "Left shoulder elevation" },
    { id: "shoulder_elev_right", kind: "tolerant", description: "Right shoulder elevation" },
];
function landmark(fr, idx) {
    return fr.bodyLandmarks[idx] ?? [0, 0, 0];
}
function visible(fr, idx) {
    return (fr.visibility[idx] ?? 0) > 0.5;
}
function angleFromVertical(a, b) {
    const v = sub(b, a);
    return degrees(angleBetween(v, Y_AXIS));
}
function elbowAngle(fr, shoulderIdx, elbowIdx, wristIdx) {
    const s = landmark(fr, shoulderIdx);
    const e = landmark(fr, elbowIdx);
    const w = landmark(fr, wristIdx);
    const upper = sub(s, e);
    const lower = sub(w, e);
    return degrees(angleBetween(upper, lower));
}
function torsoLean(fr) {
    // World-space torso direction (axes.y) vs world vertical (Y_AXIS).
    // Normalize to 0°=upright, 90°=horizontal regardless of world Y up/down (e.g. MediaPipe Y-down).
    const angleRad = angleBetween(fr.axes.y, Y_AXIS);
    const angleDeg = degrees(angleRad);
    return angleDeg > 90 ? 180 - angleDeg : angleDeg;
}
function hipDrift(fr, baseHip) {
    const hipCenter = landmark(fr, 23).map((_, i) => (landmark(fr, 23)[i] + landmark(fr, 24)[i]) / 2);
    const dx = hipCenter[0] - baseHip[0];
    const dz = hipCenter[2] - baseHip[2];
    return Math.sqrt(dx * dx + dz * dz);
}
export function computeFeatureSequence(frames) {
    const values = {};
    const mask = {};
    featureDefinitions.forEach((d) => {
        values[d.id] = [];
        mask[d.id] = [];
    });
    if (frames.length === 0) {
        return { phase: [], values, mask };
    }
    const baseHip = landmark(frames[0], 23).map((_, i) => (landmark(frames[0], 23)[i] + landmark(frames[0], 24)[i]) / 2);
    frames.forEach((fr, idx) => {
        const leftVisible = visible(fr, 11) && visible(fr, 13) && visible(fr, 15);
        const rightVisible = visible(fr, 12) && visible(fr, 14) && visible(fr, 16);
        const torsoVisible = visible(fr, 11) && visible(fr, 12) && visible(fr, 23) && visible(fr, 24);
        const armRaiseLeft = leftVisible ? 180 - angleFromVertical(landmark(fr, 11), landmark(fr, 13)) : 0;
        const armRaiseRight = rightVisible ? 180 - angleFromVertical(landmark(fr, 12), landmark(fr, 14)) : 0;
        const symDiff = leftVisible && rightVisible ? Math.abs(armRaiseLeft - armRaiseRight) : 0;
        const torsoLeanVal = torsoVisible ? torsoLean(fr) : 0;
        const hipDriftVal = torsoVisible ? hipDrift(fr, baseHip) : 0;
        const kneesVisibleGlobal = visible(fr, 23) && visible(fr, 24) && visible(fr, 25) && visible(fr, 26);
        const kneeValgusValue = kneesVisibleGlobal ? kneeValgusGlobal(fr) : 0;
        const elbowLeft = leftVisible ? elbowAngle(fr, 11, 13, 15) : 0;
        const elbowRight = rightVisible ? elbowAngle(fr, 12, 14, 16) : 0;
        const sCY = (landmark(fr, 11)[1] + landmark(fr, 12)[1]) / 2;
        const hCY = (landmark(fr, 23)[1] + landmark(fr, 24)[1]) / 2;
        const tH = Math.abs(sCY - hCY) || 1;
        const shoulderElevL = torsoVisible ? (landmark(fr, 11)[1] - landmark(fr, 23)[1]) / tH - 1.0 : 0;
        const shoulderElevR = torsoVisible ? (landmark(fr, 12)[1] - landmark(fr, 24)[1]) / tH - 1.0 : 0;
        values["arm_raise_left"].push(armRaiseLeft);
        mask["arm_raise_left"].push(leftVisible);
        values["arm_raise_right"].push(armRaiseRight);
        mask["arm_raise_right"].push(rightVisible);
        values["arm_symmetry"].push(symDiff);
        mask["arm_symmetry"].push(leftVisible && rightVisible);
        values["torso_lean"].push(torsoLeanVal);
        mask["torso_lean"].push(torsoVisible);
        values["hip_drift"].push(hipDriftVal);
        mask["hip_drift"].push(torsoVisible);
        values["knee_valgus_left"].push(kneeValgusValue);
        mask["knee_valgus_left"].push(kneesVisibleGlobal);
        values["knee_valgus_right"].push(kneeValgusValue);
        mask["knee_valgus_right"].push(kneesVisibleGlobal);
        values["elbow_angle_left"].push(elbowLeft);
        mask["elbow_angle_left"].push(leftVisible);
        values["elbow_angle_right"].push(elbowRight);
        mask["elbow_angle_right"].push(rightVisible);
        values["shoulder_elev_left"].push(shoulderElevL);
        mask["shoulder_elev_left"].push(torsoVisible && visible(fr, 11));
        values["shoulder_elev_right"].push(shoulderElevR);
        mask["shoulder_elev_right"].push(torsoVisible && visible(fr, 12));
    });
    const phase = frames.map((_, i) => Math.round((i / Math.max(frames.length - 1, 1)) * 99));
    return { phase, values, mask };
}
/**
 * Compute the 9 features for a single body frame.
 * `baseHip` defaults to the frame's own hip center (no drift reference).
 */
export function computeFrameFeatures(fr, baseHip) {
    const leftVisible = visible(fr, 11) && visible(fr, 13) && visible(fr, 15);
    const rightVisible = visible(fr, 12) && visible(fr, 14) && visible(fr, 16);
    const torsoVisible = visible(fr, 11) && visible(fr, 12) && visible(fr, 23) && visible(fr, 24);
    const defaultHip = [
        (landmark(fr, 23)[0] + landmark(fr, 24)[0]) / 2,
        (landmark(fr, 23)[1] + landmark(fr, 24)[1]) / 2,
        (landmark(fr, 23)[2] + landmark(fr, 24)[2]) / 2,
    ];
    const hip = baseHip ?? defaultHip;
    const shoulderCY = (landmark(fr, 11)[1] + landmark(fr, 12)[1]) / 2;
    const hipCY = (landmark(fr, 23)[1] + landmark(fr, 24)[1]) / 2;
    const torsoH = Math.abs(shoulderCY - hipCY) || 1;
    const values = {
        arm_raise_left: leftVisible ? 180 - angleFromVertical(landmark(fr, 11), landmark(fr, 13)) : 0,
        arm_raise_right: rightVisible ? 180 - angleFromVertical(landmark(fr, 12), landmark(fr, 14)) : 0,
        arm_symmetry: 0,
        torso_lean: torsoVisible ? torsoLean(fr) : 0,
        hip_drift: torsoVisible ? hipDrift(fr, hip) : 0,
        knee_valgus_left: 0,
        knee_valgus_right: 0,
        elbow_angle_left: leftVisible ? elbowAngle(fr, 11, 13, 15) : 0,
        elbow_angle_right: rightVisible ? elbowAngle(fr, 12, 14, 16) : 0,
        shoulder_elev_left: torsoVisible ? (landmark(fr, 11)[1] - landmark(fr, 23)[1]) / torsoH - 1.0 : 0,
        shoulder_elev_right: torsoVisible ? (landmark(fr, 12)[1] - landmark(fr, 24)[1]) / torsoH - 1.0 : 0,
    };
    values.arm_symmetry = leftVisible && rightVisible
        ? Math.abs(values.arm_raise_left - values.arm_raise_right) : 0;
    const kneesVisibleGlobal = visible(fr, 23) && visible(fr, 24) && visible(fr, 25) && visible(fr, 26);
    const kneeValgusValue = kneesVisibleGlobal ? kneeValgusGlobal(fr) : 0;
    values.knee_valgus_left = kneeValgusValue;
    values.knee_valgus_right = kneeValgusValue;
    const mask = {
        arm_raise_left: leftVisible,
        arm_raise_right: rightVisible,
        arm_symmetry: leftVisible && rightVisible,
        torso_lean: torsoVisible,
        hip_drift: torsoVisible,
        knee_valgus_left: kneesVisibleGlobal,
        knee_valgus_right: kneesVisibleGlobal,
        elbow_angle_left: leftVisible,
        elbow_angle_right: rightVisible,
        shoulder_elev_left: torsoVisible && visible(fr, 11),
        shoulder_elev_right: torsoVisible && visible(fr, 12),
    };
    return { values, mask };
}
