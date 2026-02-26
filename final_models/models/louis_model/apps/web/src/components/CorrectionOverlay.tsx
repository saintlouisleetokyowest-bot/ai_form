import React, { useEffect, useRef } from "react";
import { PoseFrame } from "@ai-form/core";

const EDGES: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
];

const CORRECTION_COLOR = "rgba(255, 180, 50, 0.6)";
const CORRECTION_JOINT = "rgba(255, 180, 50, 0.8)";

function drawUserSkeleton(ctx: CanvasRenderingContext2D, pts: { x: number; y: number; visibility?: number }[]) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.strokeStyle = "rgba(255,255,255,0.65)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  EDGES.forEach(([a, b]) => {
    const pa = pts[a];
    const pb = pts[b];
    if (!pa || !pb) return;
    if ((pa.visibility ?? 0) < 0.4 || (pb.visibility ?? 0) < 0.4) return;
    ctx.moveTo(pa.x * w, pa.y * h);
    ctx.lineTo(pb.x * w, pb.y * h);
  });
  ctx.stroke();

  pts.forEach((p) => {
    if (!p || (p.visibility ?? 0) < 0.4) return;
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawCorrectionSkeleton(
  ctx: CanvasRenderingContext2D,
  pts: Array<{ x: number; y: number }>,
  userPts: { x: number; y: number; visibility?: number }[],
) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;

  // Check if correction differs from user (confidence > 0 means real model)
  let hasDiff = false;
  for (let i = 0; i < Math.min(pts.length, userPts.length); i++) {
    const up = userPts[i];
    const cp = pts[i];
    if (!up || !cp) continue;
    if (Math.abs(up.x - cp.x) > 0.005 || Math.abs(up.y - cp.y) > 0.005) {
      hasDiff = true;
      break;
    }
  }

  if (!hasDiff) {
    // Mock mode — no ML model loaded. Show placeholder.
    ctx.save();
    ctx.fillStyle = "rgba(255, 180, 50, 0.7)";
    ctx.font = "bold 14px 'Inter', sans-serif";
    ctx.fillText("ML model not loaded", 12, ctx.canvas.height - 16);
    ctx.restore();
    return;
  }

  ctx.save();
  ctx.strokeStyle = CORRECTION_COLOR;
  ctx.lineWidth = 4;
  ctx.setLineDash([8, 5]);
  ctx.lineCap = "round";
  ctx.beginPath();
  EDGES.forEach(([a, b]) => {
    const pa = pts[a];
    const pb = pts[b];
    if (!pa || !pb) return;
    ctx.moveTo(pa.x * w, pa.y * h);
    ctx.lineTo(pb.x * w, pb.y * h);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = CORRECTION_JOINT;
  pts.forEach((p) => {
    if (!p) return;
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  frame?: PoseFrame;
  correctionLandmarks?: Array<{ x: number; y: number }> | null;
}

export const CorrectionOverlay: React.FC<Props> = ({ videoRef, frame, correctionLandmarks }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth || canvas.clientWidth;
    canvas.height = video.videoHeight || canvas.clientHeight;
  }, [videoRef.current?.videoWidth, videoRef.current?.videoHeight]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame?.imageLandmarks) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (correctionLandmarks) {
      drawCorrectionSkeleton(ctx, correctionLandmarks, frame.imageLandmarks);
    }

    drawUserSkeleton(ctx, frame.imageLandmarks);
  }, [frame, correctionLandmarks]);

  return <canvas ref={canvasRef} />;
};
