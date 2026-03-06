import { useMemo, useRef } from "react";
import {
  alignTo100,
  computeFeatureSequence,
  computeGhostArms,
  defaultLateralRaiseTemplate,
  estimateOnlinePhase,
  GhostArms,
  INITIAL_PHASE_STATE,
  PhaseState,
  renderHintsForFaults,
  scoreAgainstTemplate,
  segmentReps,
  sliceFeatureSequence,
  toBodyFrame,
} from "@ai-form/core";
import { PoseFrame } from "@ai-form/core";

export function useRepScoring(frames: PoseFrame[]) {
  const bodyFrames = useMemo(() => frames.map((f) => toBodyFrame(f)), [frames]);
  const featureSeq = useMemo(() => computeFeatureSequence(bodyFrames), [bodyFrames]);
  const reps = useMemo(() => segmentReps(featureSeq), [featureSeq]);
  const template = useMemo(() => defaultLateralRaiseTemplate(), []);

  const lastRep = reps[reps.length - 1];
  const repFeatures = lastRep ? sliceFeatureSequence(featureSeq, lastRep) : null;
  const aligned = repFeatures ? alignTo100(repFeatures) : null;
  const scoring = aligned ? scoreAgainstTemplate(aligned, template) : null;
  const hints = scoring ? renderHintsForFaults(scoring.faults) : [];

  const phaseRef = useRef<PhaseState>(INITIAL_PHASE_STATE);

  const { ghostArms, onlinePhase } = useMemo(() => {
    const lastFrame = frames[frames.length - 1];
    if (!lastFrame?.imageLandmarks) {
      return { ghostArms: null as GhostArms | null, onlinePhase: 0 };
    }

    const leftRaise = featureSeq.values["arm_raise_left"];
    const rightRaise = featureSeq.values["arm_raise_right"];
    const leftMask = featureSeq.mask["arm_raise_left"];
    const rightMask = featureSeq.mask["arm_raise_right"];

    const len = leftRaise?.length ?? 0;
    if (len === 0) return { ghostArms: null as GhostArms | null, onlinePhase: 0 };

    const i = len - 1;
    const lVis = leftMask?.[i] ?? false;
    const rVis = rightMask?.[i] ?? false;
    const lVal = lVis ? (leftRaise?.[i] ?? 0) : 0;
    const rVal = rVis ? (rightRaise?.[i] ?? 0) : 0;
    const count = (lVis ? 1 : 0) + (rVis ? 1 : 0);
    const avgRaise = count > 0 ? (lVal + rVal) / count : 0;

    const abductionAngle = 180 - avgRaise;

    const nextPhase = estimateOnlinePhase(abductionAngle, phaseRef.current);
    phaseRef.current = nextPhase;

    const arms = computeGhostArms(lastFrame.imageLandmarks, template, nextPhase.phase);
    return { ghostArms: arms, onlinePhase: nextPhase.phase };
  }, [frames, featureSeq, template]);

  return {
    bodyFrames,
    featureSeq,
    reps,
    lastRep,
    score: scoring?.score ?? null,
    faults: scoring?.faults ?? [],
    hints,
    template,
    ghostArms,
    onlinePhase,
  };
}
