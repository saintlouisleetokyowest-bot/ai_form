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

/** Power-easing ramp (t^exp): slow start, fast near peak. */
function easedRampCurve(
  start: number,
  peak: number,
  end: number,
  exp = 2,
  length = 100,
): number[] {
  const arr: number[] = [];
  const upLen = Math.floor(length / 2);
  for (let i = 0; i < length; i++) {
    if (i <= upLen) {
      const t = Math.pow(i / upLen, exp);          // ease-in
      arr.push(start + (peak - start) * t);
    } else {
      const t = Math.pow((i - upLen) / (length - 1 - upLen), exp); // ease-in (reverse)
      arr.push(peak + (end - peak) * t);
    }
  }
  return arr;
}

function constantBand(mean: number[], stdVal: number, kind: TemplateBand["kind"]): TemplateBand {
  return { mean, std: Array(mean.length).fill(stdVal), kind };
}

/** Default lateral-raise template (replace with offline mean/std when available). */
export function defaultLateralRaiseTemplate(): TemplateBundle {
  const armRaise = rampCurve(10, 90, 10);
  const symmetry = flatCurve(0);
  const torsoLean = flatCurve(0);
  const hipDrift = flatCurve(0);
  const kneeValgus = flatCurve(0);
  const elbow = easedRampCurve(160, 140, 160, 2);
  const shoulderElev = rampCurve(0.05, 0.12, 0.05);

  return {
    phaseCount: 100,
    features: {
      arm_raise_left: constantBand(armRaise, 8, "strict"),
      arm_raise_right: constantBand(armRaise, 8, "strict"),
      arm_symmetry: constantBand(symmetry, 10, "strict"),
      torso_lean: constantBand(torsoLean, 10, "strict"),
      hip_drift: constantBand(hipDrift, 0.1, "strict"),
      knee_valgus_left: constantBand(kneeValgus, 0.07, "strict"),
      knee_valgus_right: constantBand(kneeValgus, 0.07, "strict"),
      elbow_angle_left: constantBand(elbow, 15, "tolerant"),
      elbow_angle_right: constantBand(elbow, 15, "tolerant"),
      shoulder_elev_left: constantBand(shoulderElev, 0.05, "tolerant"),
      shoulder_elev_right: constantBand(shoulderElev, 0.05, "tolerant"),
    },
  };
}
