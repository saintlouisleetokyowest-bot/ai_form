import React, { useEffect, useRef } from "react";
import { Fault, GhostArms, PoseFrame } from "@ai-form/core";
import { RenderHint } from "@ai-form/core";

const EDGES: [number, number][] = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [24, 26],
  [25, 27],
  [26, 28],
];

function drawSkeleton(ctx: CanvasRenderingContext2D, pts: { x: number; y: number; visibility?: number }[]) {
  ctx.strokeStyle = "rgba(255,255,255,0.65)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  EDGES.forEach(([a, b]) => {
    const pa = pts[a];
    const pb = pts[b];
    if (!pa || !pb) return;
    if ((pa.visibility ?? 0) < 0.4 || (pb.visibility ?? 0) < 0.4) return;
    ctx.moveTo(pa.x * ctx.canvas.width, pa.y * ctx.canvas.height);
    ctx.lineTo(pb.x * ctx.canvas.width, pb.y * ctx.canvas.height);
  });
  ctx.stroke();
}

function drawJoints(ctx: CanvasRenderingContext2D, pts: { x: number; y: number; visibility?: number }[], highlight: number[]) {
  pts.forEach((p, idx) => {
    if (!p || (p.visibility ?? 0) < 0.4) return;
    const x = p.x * ctx.canvas.width;
    const y = p.y * ctx.canvas.height;
    ctx.fillStyle = highlight.includes(idx) ? "#f87171" : "rgba(255,255,255,0.7)";
    ctx.beginPath();
    ctx.arc(x, y, highlight.includes(idx) ? 7 : 5, 0, Math.PI * 2);
    ctx.fill();
  });
}

const GHOST_COLOR = "rgba(0, 230, 180, 0.55)";
const GHOST_JOINT_COLOR = "rgba(0, 230, 180, 0.7)";

function drawGhostArms(ctx: CanvasRenderingContext2D, ghost: GhostArms) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;

  const segments: [{ x: number; y: number }, { x: number; y: number }][] = [
    [ghost.leftShoulder, ghost.leftElbow],
    [ghost.leftElbow, ghost.leftWrist],
    [ghost.rightShoulder, ghost.rightElbow],
    [ghost.rightElbow, ghost.rightWrist],
  ];

  ctx.save();
  ctx.strokeStyle = GHOST_COLOR;
  ctx.lineWidth = 5;
  ctx.setLineDash([10, 6]);
  ctx.lineCap = "round";
  ctx.beginPath();
  segments.forEach(([a, b]) => {
    ctx.moveTo(a.x * w, a.y * h);
    ctx.lineTo(b.x * w, b.y * h);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  const joints = [
    ghost.leftShoulder,
    ghost.leftElbow,
    ghost.leftWrist,
    ghost.rightShoulder,
    ghost.rightElbow,
    ghost.rightWrist,
  ];
  ctx.fillStyle = GHOST_JOINT_COLOR;
  joints.forEach((p) => {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

const HINT_FONT = "bold 20px 'Inter', sans-serif";
const HINT_LINE_HEIGHT = 33;
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
    const midIdx = hint.joints[Math.floor(hint.joints.length / 2)];
    const p = imageLandmarks[midIdx] ?? imageLandmarks[hint.joints[0]];
    const textW = ctx.measureText(hint.label).width;
    return {
      label: hint.label,
      color: hint.highlightColor,
      anchorX: p ? p.x * w + 10 : 10,
      anchorY: p ? p.y * h : 30,
      textW,
      y: 0, // resolved below
    };
  });

  // Sort by anchor Y so we can push downward on collision
  labels.sort((a, b) => a.anchorY - b.anchorY);

  let prevBottom = -Infinity;
  for (const lbl of labels) {
    const idealTop = lbl.anchorY - HINT_LINE_HEIGHT / 2;
    lbl.y = Math.max(idealTop, prevBottom + 4);
    prevBottom = lbl.y + HINT_LINE_HEIGHT;
  }

  for (const lbl of labels) {
    const bx = lbl.anchorX - HINT_PAD_X;
    const by = lbl.y - HINT_PAD_Y;
    const bw = lbl.textW + HINT_PAD_X * 2;
    const bh = HINT_LINE_HEIGHT;
    const radius = 5;

    const clampedX = Math.max(2, Math.min(bx, w - bw - 2));
    const clampedY = Math.max(2, Math.min(by, h - bh - 2));

    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.beginPath();
    ctx.roundRect(clampedX, clampedY, bw, bh, radius);
    ctx.fill();

    ctx.fillStyle = lbl.color;
    ctx.font = HINT_FONT;
    ctx.fillText(lbl.label, clampedX + HINT_PAD_X, clampedY + HINT_LINE_HEIGHT - HINT_PAD_Y - 2);
    ctx.restore();
  }
}

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  frame?: PoseFrame;
  faults: Fault[];
  hints: RenderHint[];
  ghostArms?: GhostArms | null;
  onlinePhase?: number;
}

export const OverlayCanvas: React.FC<Props> = ({ videoRef, frame, faults, hints, ghostArms, onlinePhase }) => {
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

    if (ghostArms) {
      drawGhostArms(ctx, ghostArms);
    }

    drawSkeleton(ctx, frame.imageLandmarks);
    const highlightJoints = Array.from(new Set(faults.flatMap((f) => f.joints)));
    drawJoints(ctx, frame.imageLandmarks, highlightJoints);

    drawHintLabels(ctx, hints, frame.imageLandmarks);

    if (typeof onlinePhase === "number") {
      ctx.save();
      ctx.fillStyle = "rgba(0, 230, 180, 0.85)";
      ctx.font = "bold 20px 'Inter', sans-serif";
      ctx.fillText(`Phase: ${onlinePhase}`, 18, canvas.height - 24);
      ctx.restore();
    }
  }, [frame, faults, hints, ghostArms, onlinePhase]);

  return <canvas ref={canvasRef} />;
};
