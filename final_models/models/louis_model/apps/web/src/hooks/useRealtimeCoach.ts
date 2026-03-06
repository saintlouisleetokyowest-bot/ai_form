import { useEffect, useMemo, useRef, useState } from "react";
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
import type { RestShoulderElev } from "@ai-form/core";
import type { PoseFrame } from "@ai-form/core";
import { createCorrector } from "@ai-form/models";

const ARMS_DOWN_PHASE = 22;

function orientationKey(bf: { bodyLandmarks: (number[] | undefined)[] }): "posZ" | "negZ" {
  const z = ((bf.bodyLandmarks[11]?.[2] ?? 0) + (bf.bodyLandmarks[12]?.[2] ?? 0)) / 2;
  return z >= 0 ? "posZ" : "negZ";
}

export function useRealtimeCoach(frames: PoseFrame[], correctorUrl?: string) {
  const template = useMemo(() => defaultLateralRaiseTemplate(), []);
  const phaseRef = useRef<PhaseState>(INITIAL_PHASE_STATE);
  const smootherRef = useRef(new FaultSmoother());
  const correctorRef = useRef<IPoseCorrector>(createCorrector(correctorUrl));
  const restByOrientationRef = useRef<Record<string, RestShoulderElev>>({});
  const prevOrientRef = useRef<string | null>(null);
  const baseHipImageRef = useRef<[number, number] | null>(null);
  const lowDriftFramesRef = useRef(0);
  const armsDownFramesRef = useRef(0);
  const [correctionOutput, setCorrectionOutput] = useState<import("@ai-form/core").CorrectorOutput | null>(null);
  const [correctorReady, setCorrectorReady] = useState(false);
  const correctionInputRef = useRef<{ bodyLandmarks: Float32Array; phase: number; visibility: Float32Array } | null>(null);

  useEffect(() => {
    const c = createCorrector(correctorUrl);
    correctorRef.current = c;
    setCorrectionOutput(null);
    setCorrectorReady(false);
    c.load().then(() => {
      setCorrectorReady(Boolean(correctorUrl) && c.ready);
    }).catch((e) => {
      console.warn("Corrector load failed:", e);
      setCorrectorReady(false);
    });
  }, [correctorUrl]);

  useEffect(() => {
    const input = correctionInputRef.current;
    const corrector = correctorRef.current;
    if (!input || !correctorReady || !("runAsync" in corrector)) return;
    (corrector as { runAsync: (i: typeof input) => Promise<import("@ai-form/core").CorrectorOutput> })
      .runAsync(input)
      .then(setCorrectionOutput)
      .catch(() => {});
  }, [frames, correctorReady]);

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
        correctorReady,
      };
    }

    const bodyFrame = toBodyFrame(lastFrame);
    const lm = (idx: number) => bodyFrame.bodyLandmarks[idx] ?? [0, 0, 0];
    const imgLm = (idx: number) => lastFrame.imageLandmarks?.[idx];
    const torsoVisible =
      (bodyFrame.visibility[11] ?? 0) > 0.5 &&
      (bodyFrame.visibility[12] ?? 0) > 0.5 &&
      (bodyFrame.visibility[23] ?? 0) > 0.5 &&
      (bodyFrame.visibility[24] ?? 0) > 0.5;
    const i23 = imgLm(23);
    const i24 = imgLm(24);
    if (torsoVisible && i23 && i24 && baseHipImageRef.current == null) {
      baseHipImageRef.current = [
        (i23.x + i24.x) / 2,
        (i23.y + i24.y) / 2,
      ];
    }
    const features = computeFrameFeatures(bodyFrame, undefined);
    const armsDownByRaise =
      (features.mask.arm_raise_left && features.mask.arm_raise_right &&
       (features.values.arm_raise_left ?? 0) < 28 && (features.values.arm_raise_right ?? 0) < 28);
    if (torsoVisible && i23 && i24) {
      const cx = (i23.x + i24.x) / 2;
      const cy = (i23.y + i24.y) / 2;
      if (baseHipImageRef.current == null) {
        baseHipImageRef.current = [cx, cy];
        lowDriftFramesRef.current = 0;
        armsDownFramesRef.current = 0;
      } else {
        const [bx, by] = baseHipImageRef.current;
        const drift = Math.sqrt((cx - bx) ** 2 + (cy - by) ** 2);
        features.values.hip_drift = drift;
        const stableThreshold = 0.08;
        if (drift < stableThreshold) {
          lowDriftFramesRef.current += 1;
          if (lowDriftFramesRef.current >= 5) {
            baseHipImageRef.current = [cx, cy];
            lowDriftFramesRef.current = 0;
            features.values.hip_drift = 0;
          }
        } else {
          lowDriftFramesRef.current = 0;
        }
        if (armsDownByRaise) {
          armsDownFramesRef.current += 1;
          if (armsDownFramesRef.current >= 12) {
            baseHipImageRef.current = [cx, cy];
            armsDownFramesRef.current = 0;
            lowDriftFramesRef.current = 0;
            features.values.hip_drift = 0;
          }
        } else {
          armsDownFramesRef.current = 0;
        }
      }
    } else {
      armsDownFramesRef.current = 0;
    }
    const shoulderCY = (lm(11)[1] + lm(12)[1]) / 2;
    const hipCY = (lm(23)[1] + lm(24)[1]) / 2;
    const currentTorsoH = Math.abs(shoulderCY - hipCY) || 1;

    const leftRaise = features.mask.arm_raise_left ? features.values.arm_raise_left : 0;
    const rightRaise = features.mask.arm_raise_right ? features.values.arm_raise_right : 0;
    const count = (features.mask.arm_raise_left ? 1 : 0) + (features.mask.arm_raise_right ? 1 : 0);
    const avgRaise = count > 0 ? (leftRaise + rightRaise) / count : 0;
    const nextPhase = estimateOnlinePhase(avgRaise, phaseRef.current);
    phaseRef.current = nextPhase;
    const phase = nextPhase.phase;

    const orient = orientationKey(bodyFrame);
    const hasShoulderElev =
      features.mask.shoulder_elev_left && features.mask.shoulder_elev_right;
    const armsDown = phase < ARMS_DOWN_PHASE;
    const orientationChanged = prevOrientRef.current !== orient;
    if (hasShoulderElev && armsDown) {
      const restBy = restByOrientationRef.current;
      const needUpdate = restBy[orient] == null || orientationChanged;
      if (needUpdate) {
        restBy[orient] = {
          left: features.values.shoulder_elev_left,
          right: features.values.shoulder_elev_right,
          torsoH_rest: currentTorsoH,
        };
      }
    }
    prevOrientRef.current = orient;

    const restShoulderElev =
      restByOrientationRef.current[orient] ??
      Object.values(restByOrientationRef.current)[0];

    const { faults, frameScore } = scoreFrame(features, template, phase, {
      restShoulderElev,
      currentTorsoH,
    });
    const top3 = smootherRef.current.update(faults);
    const ghostUpperBody = computeGhostUpperBody(
      lastFrame.imageLandmarks, bodyFrame, template, phase,
    );
    const hints = renderHintsForRealtimeFaults(top3);

    const corrector = correctorRef.current;
    let correctionLandmarks: Array<{ x: number; y: number }> | null = null;
    if (correctorReady) {
      const flat = new Float32Array(33 * 3);
      for (let i = 0; i < 33; i++) {
        const lm = bodyFrame.bodyLandmarks[i];
        if (lm) { flat[i * 3] = lm[0]; flat[i * 3 + 1] = lm[1]; flat[i * 3 + 2] = lm[2]; }
      }
      const vis = new Float32Array(bodyFrame.visibility);
      const input = { bodyLandmarks: flat, phase, visibility: vis };
      correctionInputRef.current = input;
      const output = correctionOutput ?? corrector.run(input);
      correctionLandmarks = projectCorrectionToImage(output, bodyFrame, lastFrame.imageLandmarks);
    }

    return {
      ghostUpperBody,
      correctionLandmarks,
      top3Faults: top3,
      hints,
      onlinePhase: phase,
      frameScore,
      correctorReady,
    };
  }, [frames, template, correctionOutput, correctorReady]);
}
