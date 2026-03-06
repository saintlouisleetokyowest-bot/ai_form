import { useEffect, useRef, useState, type RefObject } from "react";
import { PoseFrame } from "@ai-form/core";
import {
  FilesetResolver,
  PoseLandmarker,
  PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";

interface Options {
  enabled?: boolean;
  bufferSize?: number;
}

export function usePose(
  videoRef: RefObject<HTMLVideoElement>,
  { enabled = true, bufferSize = 240 }: Options = {}
) {
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [frames, setFrames] = useState<PoseFrame[]>([]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!enabled) return;
      setStatus("loading");
      try {
        const fileset = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
        );
        const landmarker = await PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URL },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        if (!cancelled) {
          landmarkerRef.current = landmarker;
          setStatus("ready");
        }
      } catch (e) {
        console.error(e);
        if (!cancelled) setStatus("error");
      }
    }
    load();
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
    };
  }, [enabled]);

  useEffect(() => {
    let rafId: number;
    function loop() {
      const landmarker = landmarkerRef.current;
      const video = videoRef.current;
      if (!landmarker || !video || status !== "ready") {
        rafId = requestAnimationFrame(loop);
        return;
      }
      const ts = performance.now();
      const result: PoseLandmarkerResult = landmarker.detectForVideo(video, ts);
      if (result.worldLandmarks?.length) {
        const world = result.worldLandmarks[0];
        const image = result.landmarks?.[0];
        setFrames((prev) => {
          const next = [
            ...prev,
            {
              landmarks: world.map((p) => ({
                x: p.x,
                y: p.y,
                z: p.z,
                visibility: p.visibility ?? 1,
              })),
              imageLandmarks: image?.map((p) => ({
                x: p.x,
                y: p.y,
                z: p.z ?? 0,
                visibility: p.visibility ?? 1,
              })),
              ts,
              index: prev.length,
            },
          ];
          return next.slice(-bufferSize);
        });
      }
      rafId = requestAnimationFrame(loop);
    }
    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [status, bufferSize, videoRef]);

  const lastFrame = frames[frames.length - 1];
  return { status, frames, lastFrame };
}
