import React, { useEffect, useRef } from "react";
import { GhostUpperBody, GHOST_EDGES, PoseFrame, RealtimeFault } from "@ai-form/core";
import { RenderHint } from "@ai-form/core";

const USER_EDGES: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
];

function drawUserSkeleton(ctx: CanvasRenderingContext2D, pts: { x: number; y: number; visibility?: number }[]) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.strokeStyle = "rgba(255,255,255,0.65)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  USER_EDGES.forEach(([a, b]) => {
    const pa = pts[a];
    const pb = pts[b];
    if (!pa || !pb) return;
    if ((pa.visibility ?? 0) < 0.4 || (pb.visibility ?? 0) < 0.4) return;
    ctx.moveTo(pa.x * w, pa.y * h);
    ctx.lineTo(pb.x * w, pb.y * h);
  });
  ctx.stroke();
}

function drawUserJoints(
  ctx: CanvasRenderingContext2D,
  pts: { x: number; y: number; visibility?: number }[],
  highlight: number[],
) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  pts.forEach((p, idx) => {
    if (!p || (p.visibility ?? 0) < 0.4) return;
    const x = p.x * w;
    const y = p.y * h;
    ctx.fillStyle = highlight.includes(idx) ? "#f87171" : "rgba(255,255,255,0.7)";
    ctx.beginPath();
    ctx.arc(x, y, highlight.includes(idx) ? 7 : 5, 0, Math.PI * 2);
    ctx.fill();
  });
}

const GHOST_COLOR = "rgba(0, 230, 180, 0.55)";
const GHOST_JOINT_COLOR = "rgba(0, 230, 180, 0.7)";

function drawGhostUpperBody(ctx: CanvasRenderingContext2D, ghost: GhostUpperBody) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;

  ctx.save();
  ctx.strokeStyle = GHOST_COLOR;
  ctx.lineWidth = 5;
  ctx.setLineDash([10, 6]);
  ctx.lineCap = "round";
  ctx.beginPath();
  GHOST_EDGES.forEach(([a, b]) => {
    const pa = ghost.landmarks[a];
    const pb = ghost.landmarks[b];
    if (!pa || !pb) return;
    ctx.moveTo(pa.x * w, pa.y * h);
    ctx.lineTo(pb.x * w, pb.y * h);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = GHOST_JOINT_COLOR;
  for (const idx of [11, 12, 13, 14, 15, 16, 23, 24]) {
    const p = ghost.landmarks[idx];
    if (!p) continue;
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

const HINT_FONT = "bold 24px 'Inter', sans-serif";
const HINT_LINE_HEIGHT = 39;
const HINT_PAD_X = 9;
const HINT_PAD_Y = 6;

function drawHintLabels(
  ctx: CanvasRenderingContext2D,
  hints: RenderHint[],
  imageLandmarks: { x: number; y: number; visibility?: number }[],
) {
  if (hints.length === 0) return;
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.font = HINT_FONT;

  const labels = hints.map((hint) => {
    const isShoulderSymmetry =
      hint.joints.length === 6 && hint.joints.includes(11) && hint.joints.includes(12);
    const p11 = imageLandmarks[11];
    const p12 = imageLandmarks[12];

    let anchorX: number, anchorY: number;
    if (isShoulderSymmetry && p11 && p12) {
      const midX = (p11.x + p12.x) / 2;
      const midY = (p11.y + p12.y) / 2;
      anchorX = (1 - midX) * w;
      anchorY = midY * h + 20;
    } else {
      const midIdx = hint.joints[Math.floor(hint.joints.length / 2)];
      const p = imageLandmarks[midIdx] ?? imageLandmarks[hint.joints[0]];
      anchorX = p ? (1 - p.x) * w + 10 : 10;
      anchorY = p ? p.y * h : 30;
    }
    const textW = ctx.measureText(hint.label).width;
    return { label: hint.label, color: hint.highlightColor, anchorX, anchorY, textW, y: 0 };
  });

  labels.sort((a, b) => a.anchorY - b.anchorY);
  let prevBottom = -Infinity;
  for (const lbl of labels) {
    const idealTop = lbl.anchorY - HINT_LINE_HEIGHT / 2;
    lbl.y = Math.max(idealTop, prevBottom + 4);
    prevBottom = lbl.y + HINT_LINE_HEIGHT;
  }

  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);

  for (const lbl of labels) {
    const bx = lbl.anchorX - HINT_PAD_X;
    const by = lbl.y - HINT_PAD_Y;
    const bw = lbl.textW + HINT_PAD_X * 2;
    const bh = HINT_LINE_HEIGHT;
    const clampedX = Math.max(2, Math.min(bx, w - bw - 2));
    const clampedY = Math.max(2, Math.min(by, h - bh - 2));

    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.beginPath();
    ctx.roundRect(clampedX, clampedY, bw, bh, 5);
    ctx.fill();
    ctx.fillStyle = lbl.color;
    ctx.font = HINT_FONT;
    ctx.fillText(lbl.label, clampedX + HINT_PAD_X, clampedY + HINT_LINE_HEIGHT - HINT_PAD_Y - 2);
  }

  ctx.restore();
}

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  frame?: PoseFrame;
  ghostUpperBody?: GhostUpperBody | null;
  hints: RenderHint[];
  faults: RealtimeFault[];
  onlinePhase?: number;
}

export const IdealOverlay: React.FC<Props> = ({ videoRef, frame, ghostUpperBody, hints, faults, onlinePhase }) => {
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

    if (ghostUpperBody) drawGhostUpperBody(ctx, ghostUpperBody);

    drawUserSkeleton(ctx, frame.imageLandmarks);
    const highlightJoints = Array.from(new Set(faults.flatMap((f) => f.joints)));
    drawUserJoints(ctx, frame.imageLandmarks, highlightJoints);
    drawHintLabels(ctx, hints, frame.imageLandmarks);

    if (typeof onlinePhase === "number") {
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.fillStyle = "rgba(0, 230, 180, 0.85)";
      ctx.font = "bold 20px 'Inter', sans-serif";
      ctx.fillText(`Phase: ${onlinePhase}`, 18, canvas.height - 24);
      ctx.restore();
    }
  }, [frame, ghostUpperBody, hints, faults, onlinePhase]);

  return <canvas ref={canvasRef} />;
};
