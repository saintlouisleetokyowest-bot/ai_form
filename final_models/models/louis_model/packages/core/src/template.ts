import { TemplateBand, TemplateBundle } from "./types";

function rampCurve(start: number, peak: number, end: number, length = 100): number[] {
  const arr: number[] = [];
  const upLen = Math.floor(length / 2);
  for (let i = 0; i < length; i++) {
    const t = i / (length - 1);
    if (i <= upLen) {
      const tu = i / upLen;
      arr.push(start + (peak - start) * tu);
    } else {
      const td = (i - upLen) / (length - 1 - upLen);
      arr.push(peak + (end - peak) * td);
    }
  }
  return arr;
}

function flatCurve(value: number, length = 100): number[] {
  return Array(length).fill(value);
}

function constantBand(mean: number[], stdVal: number, kind: TemplateBand["kind"]): TemplateBand {
  return { mean, std: Array(mean.length).fill(stdVal), kind };
}

/**
 * Lightweight template approximating an ideal lateral raise.
 * Replace with offline-generated mean/std when available.
 */
export function defaultLateralRaiseTemplate(): TemplateBundle {
  const armRaise = rampCurve(10, 85, 10);
  const symmetry = flatCurve(0);
  const torsoLean = flatCurve(5); // small natural lean
  const hipDrift = flatCurve(0);
  const elbow = rampCurve(170, 150, 170); // slight bend near top
  const shoulderElev = rampCurve(0.05, 0.12, 0.05);

  return {
    phaseCount: 100,
    features: {
      arm_raise_left: constantBand(armRaise, 8, "strict"),
      arm_raise_right: constantBand(armRaise, 8, "strict"),
      arm_symmetry: constantBand(symmetry, 6, "strict"),
      torso_lean: constantBand(torsoLean, 5, "strict"),
      hip_drift: constantBand(hipDrift, 0.04, "strict"),
      elbow_angle_left: constantBand(elbow, 15, "tolerant"),
      elbow_angle_right: constantBand(elbow, 15, "tolerant"),
      shoulder_elev_left: constantBand(shoulderElev, 0.05, "tolerant"),
      shoulder_elev_right: constantBand(shoulderElev, 0.05, "tolerant"),
    },
  };
}
