import React, { useRef } from "react";
import { useCamera } from "./hooks/useCamera";
import { usePose } from "./hooks/usePose";
import { useRealtimeCoach } from "./hooks/useRealtimeCoach";
import { IdealOverlay } from "./components/IdealOverlay";
import { CorrectionOverlay } from "./components/CorrectionOverlay";

const statusColor: Record<string, string> = {
  idle: "#94a3b8",
  loading: "#f59e0b",
  ready: "#22c55e",
  error: "#f87171",
};

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const { ready, error } = useCamera(videoRef);
  const { status: poseStatus, frames, lastFrame } = usePose(videoRef, { enabled: ready });
  const coach = useRealtimeCoach(frames);

  return (
    <div className="split-layout">
      {/* Left panel: Module 1 — Ideal overlay */}
      <div className="split-panel">
        <div className="video-wrap">
          <video ref={videoRef} playsInline muted />
          <IdealOverlay
            videoRef={videoRef}
            frame={lastFrame}
            ghostUpperBody={coach.ghostUpperBody}
            hints={coach.hints}
            faults={coach.top3Faults}
            onlinePhase={coach.onlinePhase}
          />
          <div className="panel-label ideal-label">Ideal Form</div>
        </div>
        <div className="info-bar">
          <span className="status-dot" style={{ background: statusColor[ready ? "ready" : "idle"] }} />
          <span>Cam: {ready ? "✓" : "…"}</span>
          <span className="status-dot" style={{ background: statusColor[poseStatus] }} />
          <span>Pose: {poseStatus === "ready" ? "✓" : poseStatus}</span>
          {error && <span style={{ color: "#f87171" }}>{error}</span>}
          <span className="sep">|</span>
          <span className="score-group">
            Score: <strong>{coach.frameScore.toFixed(0)}</strong>
            <span className="score-bar-track">
              <span className="score-bar-fill" style={{ width: `${coach.frameScore}%` }} />
            </span>
          </span>
          <span className="sep">|</span>
          <span>Phase: <strong>{coach.onlinePhase}</strong></span>
          {coach.top3Faults.map((f, i) => (
            <span key={i} className={`fault-badge ${f.severity}`}>{f.message}</span>
          ))}
        </div>
      </div>

      {/* Right panel: Module 2 — ML Correction */}
      <div className="split-panel">
        <div className="video-wrap">
          <video ref={videoRef} playsInline muted />
          <CorrectionOverlay
            videoRef={videoRef}
            frame={lastFrame}
            correctionLandmarks={coach.correctionLandmarks}
          />
          <div className="panel-label correction-label">ML Correction</div>
        </div>
        <div className="info-bar">
          <span>Model: <strong>{coach.correctorReady ? "unload" : "mock (no model)"}</strong></span>
        </div>
      </div>
    </div>
  );
}
