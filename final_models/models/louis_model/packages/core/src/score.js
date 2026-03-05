import { clamp } from "./math";
/** Key features (arm raise + symmetry): higher weight + tighter threshold. torso_lean is treated as normal strict to avoid zeroing score too aggressively. hip_drift excluded to avoid score zeroing from reference drift. */
const KEY_FEATURE_IDS = new Set([
    "arm_raise_left",
    "arm_raise_right",
    "arm_symmetry",
]);
const DEFAULT_OPTIONS = {
    strictWeight: 10,
    tolerantWeight: 5,
    strictThreshold: 1,
    tolerantThreshold: 1.5,
    keyWeight: 16,
    keyThreshold: 0.8,
};
function computeExcess(values, mask, band, threshold) {
    const excess = [];
    for (let i = 0; i < values.length; i++) {
        if (!mask[i]) {
            excess.push(0);
            continue;
        }
        const z = Math.abs(values[i] - band.mean[i]) / (band.std[i] + 1e-6);
        excess.push(Math.max(0, z - threshold));
    }
    const meanExcess = excess.reduce((a, b) => a + b, 0) / Math.max(excess.length, 1);
    return { meanExcess, perPhase: excess };
}
function faultMessage(featureId) {
    switch (featureId) {
        case "arm_raise_left":
        case "arm_raise_right":
            return "Arm height off";
        case "arm_symmetry":
            return "Arms asymmetric";
        case "torso_lean":
            return "Torso leaning";
        case "hip_drift":
            return "Hip drift";
        case "knee_valgus_left":
        case "knee_valgus_right":
            return "Knee valgus (caving in)";
        case "elbow_angle_left":
        case "elbow_angle_right":
            return "Elbow angle off";
        case "shoulder_elev_left":
        case "shoulder_elev_right":
            return "Shoulder shrugging";
        default:
            return "Form deviation";
    }
}
function jointsForFeature(featureId) {
    switch (featureId) {
        case "arm_raise_left":
        case "elbow_angle_left":
            return [11, 13, 15];
        case "arm_raise_right":
        case "elbow_angle_right":
            return [12, 14, 16];
        case "shoulder_elev_left":
            return [11];
        case "shoulder_elev_right":
            return [12];
        case "arm_symmetry":
            return [11, 12, 13, 14, 15, 16];
        case "torso_lean":
            return [0]; // nose / head — hint at top of body
        case "hip_drift":
            return [11, 12, 23, 24];
        case "knee_valgus_left":
            return [23, 25, 27];
        case "knee_valgus_right":
            return [24, 26, 28];
        default:
            return [];
    }
}
function findLongestRange(values, minZ = 1) {
    let best = [0, 0];
    let current = null;
    values.forEach((v, idx) => {
        if (v >= minZ) {
            if (!current)
                current = [idx, idx];
            else
                current[1] = idx;
            if (current[1] - current[0] > best[1] - best[0])
                best = [current[0], current[1]];
        }
        else {
            current = null;
        }
    });
    return best;
}
export function scoreAgainstTemplate(seq, template, options = {}) {
    const cfg = { ...DEFAULT_OPTIONS, ...options };
    let strictPenalty = 0;
    let tolerantPenalty = 0;
    let keyPenalty = 0;
    const faults = [];
    for (const [featureId, band] of Object.entries(template.features)) {
        const values = seq.values[featureId] ?? [];
        const mask = seq.mask[featureId] ?? [];
        if (values.length === 0)
            continue;
        const isKey = KEY_FEATURE_IDS.has(featureId);
        const threshold = isKey
            ? cfg.keyThreshold
            : band.kind === "strict"
                ? cfg.strictThreshold
                : cfg.tolerantThreshold;
        const { meanExcess, perPhase } = computeExcess(values, mask, band, threshold);
        if (isKey) {
            keyPenalty += meanExcess;
        }
        else if (band.kind === "strict") {
            strictPenalty += meanExcess;
        }
        else {
            tolerantPenalty += meanExcess;
        }
        const maxZ = Math.max(...perPhase);
        if (maxZ > 0.5) {
            const phaseRange = findLongestRange(perPhase, maxZ * 0.5);
            faults.push({
                id: `fault_${featureId}`,
                featureId,
                severity: maxZ > 2 ? "error" : maxZ > 1 ? "warn" : "info",
                message: faultMessage(featureId),
                joints: jointsForFeature(featureId),
                phaseRange,
            });
        }
    }
    const strictScore = cfg.strictWeight * strictPenalty;
    const tolerantScore = cfg.tolerantWeight * tolerantPenalty;
    const keyScore = cfg.keyWeight * keyPenalty;
    const raw = 100 - strictScore - tolerantScore - keyScore;
    return {
        score: clamp(raw, 0, 100),
        faults,
        strictPenalty: strictScore,
        tolerantPenalty: tolerantScore,
    };
}
/** Z-threshold for "shrug" when using rest baseline: only fault when excess above rest exceeds this. */
const SHRUG_ABOVE_REST_Z_THRESHOLD = 0.3;
/**
 * Score a single frame against the template at the given phase.
 * When restShoulderElev + currentTorsoH are provided, shoulder_elev uses fixed rest torso height
 * so that shrugging (shoulder up) actually increases the measured elevation.
 * Returns faults sorted by zScore descending.
 */
export function scoreFrame(frame, template, phase, options = {}) {
    const { restShoulderElev, currentTorsoH, ...scoreOpts } = options;
    const cfg = { ...DEFAULT_OPTIONS, ...scoreOpts };
    const p = clamp(Math.round(phase), 0, 99);
    const faults = [];
    let totalPenalty = 0;
    const hasRestAndTorso = restShoulderElev &&
        restShoulderElev.torsoH_rest > 1e-6 &&
        currentTorsoH != null &&
        currentTorsoH > 1e-6;
    for (const [featureId, band] of Object.entries(template.features)) {
        const value = frame.values[featureId];
        const vis = frame.mask[featureId];
        if (value == null || !vis)
            continue;
        let mean = band.mean[p] ?? 0;
        const std = band.std[p] ?? 1;
        let z;
        if (featureId === "shoulder_elev_left" && hasRestAndTorso) {
            const rawHeight = (value + 1) * currentTorsoH;
            const restRawLeft = (restShoulderElev.left + 1) * restShoulderElev.torsoH_rest;
            const elevVsRest = restRawLeft > 1e-6 ? rawHeight / restRawLeft - 1 : 0;
            const excessAboveRest = Math.max(0, elevVsRest);
            z = excessAboveRest / (std + 1e-6);
        }
        else if (featureId === "shoulder_elev_right" && hasRestAndTorso) {
            const rawHeight = (value + 1) * currentTorsoH;
            const restRawRight = (restShoulderElev.right + 1) * restShoulderElev.torsoH_rest;
            const elevVsRest = restRawRight > 1e-6 ? rawHeight / restRawRight - 1 : 0;
            const excessAboveRest = Math.max(0, elevVsRest);
            z = excessAboveRest / (std + 1e-6);
        }
        else if (featureId === "shoulder_elev_left" && restShoulderElev) {
            mean = restShoulderElev.left;
            const excessAboveRest = Math.max(0, value - mean);
            z = excessAboveRest / (std + 1e-6);
        }
        else if (featureId === "shoulder_elev_right" && restShoulderElev) {
            mean = restShoulderElev.right;
            const excessAboveRest = Math.max(0, value - mean);
            z = excessAboveRest / (std + 1e-6);
        }
        else {
            z = Math.abs(value - mean) / (std + 1e-6);
        }
        const isShrugWithRest = restShoulderElev != null && (featureId === "shoulder_elev_left" || featureId === "shoulder_elev_right");
        const isKey = KEY_FEATURE_IDS.has(featureId);
        const threshold = isShrugWithRest
            ? SHRUG_ABOVE_REST_Z_THRESHOLD
            : isKey
                ? cfg.keyThreshold
                : band.kind === "strict"
                    ? cfg.strictThreshold
                    : cfg.tolerantThreshold;
        const excess = Math.max(0, z - threshold);
        const weight = isKey ? cfg.keyWeight : (band.kind === "strict" ? cfg.strictWeight : cfg.tolerantWeight);
        totalPenalty += weight * excess;
        /** Hip drift: only show fault when z is higher, to avoid persistent message from small reference drift. */
        const faultZThreshold = featureId === "hip_drift" ? 2.5 : 1.5;
        if (z > faultZThreshold && !isShrugWithRest) {
            faults.push({
                featureId,
                severity: z > 2.5 ? "error" : z > 2 ? "warn" : "info",
                message: faultMessage(featureId),
                joints: jointsForFeature(featureId),
                zScore: z,
            });
        }
        else if (isShrugWithRest && z > SHRUG_ABOVE_REST_Z_THRESHOLD) {
            faults.push({
                featureId,
                severity: z > 2.5 ? "error" : z > 2 ? "warn" : "info",
                message: faultMessage(featureId),
                joints: jointsForFeature(featureId),
                zScore: z,
            });
        }
    }
    faults.sort((a, b) => b.zScore - a.zScore);
    return { faults, frameScore: clamp(100 - totalPenalty, 0, 100) };
}
