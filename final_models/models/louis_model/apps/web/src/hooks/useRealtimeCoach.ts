import { useEffect, useMemo, useRef } from "react";
import {
  computeFrameFeatures,
  computeGhostUpperBody,
  defaultLateralRaiseTemplate,
  estimateOnlinePhase,
  FaultSmoother,
  GhostUpperBody,
  INITIAL_PHASE_STATE,
  IPoseCorrector,
  PhaseState,
  projectCorrectionToImage,
  RealtimeFault,
  renderHintsForRealtimeFaults,
  RenderHint,
  scoreFrame,
  toBodyFrame,
} from "@ai-form/core";
import type { PoseFrame } from "@ai-form/core";
import { createCorrector } from "@ai-form/models";

export function useRealtimeCoach(frames: PoseFrame[], correctorUrl?: string) {
  const template = useMemo(() => defaultLateralRaiseTemplate(), []);
  const phaseRef = useRef<PhaseState>(INITIAL_PHASE_STATE);
  const smootherRef = useRef(new FaultSmoother());
  const correctorRef = useRef<IPoseCorrector>(createCorrector(correctorUrl));

  useEffect(() => {
    const c = createCorrector(correctorUrl);
    correctorRef.current = c;
    c.load();
  }, [correctorUrl]);

  return useMemo(() => {
    const lastFrame = frames[frames.length - 1];
    if (!lastFrame?.imageLandmarks) {
      return {
        ghostUpperBody: null as GhostUpperBody | null,
        correctionLandmarks: null as Array<{ x: number; y: number }> | null,
        top3Faults: [] as RealtimeFault[],
        hints: [] as RenderHint[],
        onlinePhase: 0,
        frameScore: 100,
        correctorReady: correctorRef.current.ready,
      };
    }

    const bodyFrame = toBodyFrame(lastFrame);
    const features = computeFrameFeatures(bodyFrame);

    // Online phase estimation
    const leftRaise = features.mask.arm_raise_left ? features.values.arm_raise_left : 0;
    const rightRaise = features.mask.arm_raise_right ? features.values.arm_raise_right : 0;
    const count = (features.mask.arm_raise_left ? 1 : 0) + (features.mask.arm_raise_right ? 1 : 0);
    const avgRaise = count > 0 ? (leftRaise + rightRaise) / count : 0;
    const abduction = 180 - avgRaise;

    const nextPhase = estimateOnlinePhase(abduction, phaseRef.current);
    phaseRef.current = nextPhase;
    const phase = nextPhase.phase;

    // Module 1: Ideal overlay + realtime scoring
    const { faults, frameScore } = scoreFrame(features, template, phase);
    const top3 = smootherRef.current.update(faults);
    const ghostUpperBody = computeGhostUpperBody(
      lastFrame.imageLandmarks, bodyFrame, template, phase,
    );
    const hints = renderHintsForRealtimeFaults(top3);

    // Module 2: ML correction
    const corrector = correctorRef.current;
    let correctionLandmarks: Array<{ x: number; y: number }> | null = null;
    if (corrector.ready) {
      const flat = new Float32Array(33 * 3);
      for (let i = 0; i < 33; i++) {
        const lm = bodyFrame.bodyLandmarks[i];
        if (lm) { flat[i * 3] = lm[0]; flat[i * 3 + 1] = lm[1]; flat[i * 3 + 2] = lm[2]; }
      }
      const vis = new Float32Array(bodyFrame.visibility);
      const output = corrector.run({ bodyLandmarks: flat, phase, visibility: vis });
      correctionLandmarks = projectCorrectionToImage(output, bodyFrame, lastFrame.imageLandmarks);
    }

    return {
      ghostUpperBody,
      correctionLandmarks,
      top3Faults: top3,
      hints,
      onlinePhase: phase,
      frameScore,
      correctorReady: corrector.ready,
    };
  }, [frames, template]);
}
