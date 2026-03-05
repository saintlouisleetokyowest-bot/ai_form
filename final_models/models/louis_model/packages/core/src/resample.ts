import { FeatureSequence } from "./types";
import { clamp, lerp } from "./math";

export function resampleSeries(series: number[], target = 100): number[] {
  if (series.length === 0) return Array(target).fill(0);
  if (series.length === 1) return Array(target).fill(series[0]);
  const out: number[] = [];
  const lastIdx = series.length - 1;
  for (let i = 0; i < target; i++) {
    const t = (i / (target - 1)) * lastIdx;
    const lo = Math.floor(t);
    const hi = Math.min(lastIdx, lo + 1);
    const frac = t - lo;
    out.push(lerp(series[lo], series[hi], frac));
  }
  return out;
}

export function resampleMask(mask: boolean[], target = 100): boolean[] {
  if (mask.length === 0) return Array(target).fill(false);
  const floatMask = mask.map((m) => (m ? 1 : 0));
  return resampleSeries(floatMask, target).map((v) => v > 0.5);
}

export function resampleFeatureSequence(seq: FeatureSequence, target = 100): FeatureSequence {
  const values: Record<string, number[]> = {};
  const masks: Record<string, boolean[]> = {};
  for (const [k, v] of Object.entries(seq.values)) {
    values[k] = resampleSeries(v, target);
  }
  for (const [k, v] of Object.entries(seq.mask)) {
    masks[k] = resampleMask(v, target);
  }
  const phase = Array.from({ length: target }, (_, i) => clamp(Math.round((i / (target - 1)) * 99), 0, 99));
  return { phase, values, mask: masks };
}
