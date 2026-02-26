import { FeatureSequence, RepWindow } from "./types";

interface SegmentOptions {
  upThreshold?: number;
  downThreshold?: number;
  minFrames?: number;
}

export function segmentRepsFromAngles(
  signal: number[],
  visibility: boolean[],
  opts: SegmentOptions = {}
): RepWindow[] {
  const up = opts.upThreshold ?? 50; // degrees
  const down = opts.downThreshold ?? 20;
  const minFrames = opts.minFrames ?? 16;

  const reps: RepWindow[] = [];
  let inRep = false;
  let start = 0;
  let peakIdx = 0;
  let peakVal = -Infinity;
  let lastVisible = false;

  for (let i = 1; i < signal.length; i++) {
    const visible = visibility[i];
    if (!visible) {
      // break current rep if tracking goes bad
      if (inRep && i - start >= minFrames) {
        reps.push({ start, end: i - 1, peak: peakIdx });
      }
      inRep = false;
      peakVal = -Infinity;
      lastVisible = false;
      continue;
    }

    const val = signal[i];
    if (!inRep && val >= up && lastVisible) {
      start = i - 1;
      inRep = true;
      peakIdx = i;
      peakVal = val;
    }

    if (inRep) {
      if (val > peakVal) {
        peakVal = val;
        peakIdx = i;
      }
      if (val <= down && i - start >= minFrames) {
        reps.push({ start, end: i, peak: peakIdx });
        inRep = false;
        peakVal = -Infinity;
      }
    }
    lastVisible = visible;
  }

  // close trailing rep
  if (inRep && signal.length - start >= minFrames) {
    reps.push({ start, end: signal.length - 1, peak: peakIdx });
  }

  return reps;
}

/**
 * Convenience: choose average arm raise angle as the segmentation signal.
 */
export function segmentReps(seq: FeatureSequence): RepWindow[] {
  const left = seq.values["arm_raise_left"] ?? [];
  const right = seq.values["arm_raise_right"] ?? [];
  const leftMask = seq.mask["arm_raise_left"] ?? [];
  const rightMask = seq.mask["arm_raise_right"] ?? [];
  const signal: number[] = [];
  const mask: boolean[] = [];
  for (let i = 0; i < Math.max(left.length, right.length); i++) {
    const lVis = leftMask[i] ?? false;
    const rVis = rightMask[i] ?? false;
    mask.push(lVis || rVis);
    const lVal = lVis ? left[i] : 0;
    const rVal = rVis ? right[i] : 0;
    const count = (lVis ? 1 : 0) + (rVis ? 1 : 0);
    signal.push(count > 0 ? (lVal + rVal) / count : 0);
  }
  return segmentRepsFromAngles(signal, mask);
}
