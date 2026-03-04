import { useEffect, useMemo, useRef, useState } from 'react';

const API_DEFAULT = '/api/analyze';

const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
];

function parseScore(payload) {
  const candidates = [
    payload?.score,
    payload?.similarity,
    payload?.overall_score,
    payload?.metrics?.score,
    payload?.result?.score,
  ];
  const value = candidates.find((x) => Number.isFinite(Number(x)));
  return value == null ? null : Number(value);
}

function toXY(p) {
  if (Array.isArray(p) && p.length >= 2) return { x: Number(p[0]), y: Number(p[1]) };
  if (p && Number.isFinite(Number(p.x)) && Number.isFinite(Number(p.y))) {
    return { x: Number(p.x), y: Number(p.y) };
  }
  return null;
}

function parseLandmarks(payload) {
  const raw =
    payload?.ghost_skeleton ||
    payload?.ghostSkeleton ||
    payload?.ideal_skeleton ||
    payload?.idealSkeleton ||
    payload?.ideal_landmarks ||
    payload?.ghost_landmarks ||
    payload?.landmarks ||
    payload?.result?.ghost_skeleton ||
    payload?.result?.ideal_skeleton ||
    payload?.result?.ghost_landmarks;

  const arr = Array.isArray(raw)
    ? raw
    : raw?.landmarks || (raw && typeof raw === 'object' ? Object.keys(raw).sort((a, b) => Number(a) - Number(b)).map((k) => raw[k]) : null);
  if (!Array.isArray(arr)) return null;

  const points = arr.map(toXY);
  if (!points.some(Boolean)) return null;
  return points;
}

function parseActualLandmarks(payload) {
  const raw = payload?.actual_landmarks || payload?.result?.actual_landmarks;
  if (!raw || typeof raw !== 'object') return null;
  const arr = Object.keys(raw)
    .sort((a, b) => Number(a) - Number(b))
    .map((k) => raw[k]);
  return arr.map(toXY);
}

function pickConnections(payload, length) {
  const c = payload?.connections || payload?.ghost_connections || payload?.ideal_connections;
  if (!Array.isArray(c)) return POSE_CONNECTIONS;
  const parsed = c
    .map((x) => (Array.isArray(x) && x.length === 2 ? [Number(x[0]), Number(x[1])] : null))
    .filter((x) => x && Number.isInteger(x[0]) && Number.isInteger(x[1]) && x[0] < length && x[1] < length);
  return parsed.length ? parsed : POSE_CONNECTIONS;
}

function scoreClass(score) {
  if (score == null) return '';
  if (score >= 80) return 'score-good';
  if (score >= 60) return 'score-mid';
  return 'score-low';
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const intervalRef = useRef(null);
  const inFlightRef = useRef(false);

  const [apiUrl, setApiUrl] = useState(import.meta.env.VITE_API_URL || API_DEFAULT);
  const [sourceMode, setSourceMode] = useState('camera');
  const [uploadedUrl, setUploadedUrl] = useState('');
  const [uploadedName, setUploadedName] = useState('');
  const [fps, setFps] = useState(4);
  const [jpegQuality, setJpegQuality] = useState(0.7);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState('idle');
  const [score, setScore] = useState(null);
  const [latency, setLatency] = useState(null);
  const [error, setError] = useState('');
  const [note, setNote] = useState('');

  const streamRef = useRef(null);

  useEffect(() => {
    return () => {
      window.removeEventListener('resize', resizeOverlay);
      stopPipeline();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    return () => {
      if (uploadedUrl) URL.revokeObjectURL(uploadedUrl);
    };
  }, [uploadedUrl]);

  const apiEndpoint = useMemo(() => {
    const v = apiUrl.trim().replace(/\/+$/, '');
    if (!v) return '';
    return v;
  }, [apiUrl]);

  const endpointHint = useMemo(() => {
    if (!apiEndpoint) return '';
    if (apiEndpoint.startsWith('/')) return '';
    try {
      const u = new URL(apiEndpoint);
      if (!u.hostname.includes('run.app')) return '';
      if (!u.hostname.includes('.a.run.app') && !u.hostname.includes('.run.app')) {
        return 'Cloud Run host may be incorrect. Typical host contains .a.run.app';
      }
      return '';
    } catch {
      return 'Endpoint URL is invalid.';
    }
  }, [apiEndpoint]);

  async function startCamera() {
    if (streamRef.current) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    streamRef.current = stream;
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
    resizeOverlay();
  }

  async function startUploadedVideo() {
    const video = videoRef.current;
    if (!video) return;
    if (!uploadedUrl) {
      throw new Error('Please upload a video first.');
    }
    video.srcObject = null;
    video.src = uploadedUrl;
    await video.play();
    resizeOverlay();
  }

  function stopCamera() {
    const s = streamRef.current;
    if (s) {
      s.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }

  function resizeOverlay() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const w = video.videoWidth || 1280;
    const h = video.videoHeight || 720;
    canvas.width = w;
    canvas.height = h;
  }

  function drawGhost(points, connections) {
    const canvas = canvasRef.current;
    if (!canvas || !points) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const isNormalized = points.some((p) => p && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
    const mapPoint = (p) => {
      if (!p) return null;
      return {
        x: isNormalized ? p.x * canvas.width : p.x,
        y: isNormalized ? p.y * canvas.height : p.y,
      };
    };

    const mapped = points.map(mapPoint);

    ctx.lineWidth = 3;
    ctx.strokeStyle = '#ff4be3';
    ctx.fillStyle = '#ffffff';

    for (const [a, b] of connections) {
      const p1 = mapped[a];
      const p2 = mapped[b];
      if (!p1 || !p2) continue;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    for (const p of mapped) {
      if (!p) continue;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function clearOverlay() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function stopPipeline() {
    stopFrameLoop();
    setStatus('stopped');
    window.removeEventListener('resize', resizeOverlay);
    if (sourceMode === 'camera') {
      stopCamera();
      clearOverlay();
      return;
    }
    if (videoRef.current) {
      videoRef.current.pause();
    }
  }

  function startFrameLoop() {
    if (intervalRef.current) return;
    const intervalMs = Math.max(100, Math.floor(1000 / Number(fps || 4)));
    intervalRef.current = setInterval(sendFrame, intervalMs);
    setRunning(true);
  }

  function stopFrameLoop() {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    inFlightRef.current = false;
    setRunning(false);
  }

  async function sendFrame() {
    const video = videoRef.current;
    const snap = captureCanvasRef.current;
    if (!video || !snap || inFlightRef.current || !apiEndpoint) return;
    if (sourceMode === 'upload' && video.paused) return;

    if (!video.videoWidth || !video.videoHeight) return;

    inFlightRef.current = true;
    setStatus('processing');

    try {
      snap.width = video.videoWidth;
      snap.height = video.videoHeight;
      const sctx = snap.getContext('2d');
      sctx.drawImage(video, 0, 0, snap.width, snap.height);

      const blob = await new Promise((resolve) => snap.toBlob(resolve, 'image/jpeg', jpegQuality));
      if (!blob) throw new Error('Could not capture frame');

      const form = new FormData();
      form.append('file', blob, 'frame.jpg');
      form.append('timestamp_ms', String(Date.now()));

      const started = performance.now();
      const res = await fetch(apiEndpoint, {
        method: 'POST',
        body: form,
      });
      const ended = performance.now();
      setLatency(Math.round(ended - started));

      if (!res.ok) {
        throw new Error(`API ${res.status}: ${await res.text()}`);
      }

      const payload = await res.json();
      let lm = parseLandmarks(payload);
      const actual = parseActualLandmarks(payload);
      const connections = pickConnections(payload, lm?.length || 0);
      const nextScore = parseScore(payload);

      if (!lm && actual?.length) {
        // Backend currently returns ghost_landmarks=null for some frames.
        // Fallback to actual landmarks so user can still verify drawing pipeline.
        lm = actual;
      }

      if (lm) {
        drawGhost(lm, connections);
        if (parseLandmarks(payload)) {
          setNote('');
        } else {
          setNote('ghost_landmarks is null. Showing actual_landmarks as fallback overlay.');
        }
      } else {
        clearOverlay();
        if (actual?.length) {
          setNote('API returned actual_landmarks but ghost_landmarks is null. Backend did not generate ghost skeleton for this frame.');
        } else {
          setNote('No drawable skeleton in API response.');
        }
      }
      if (nextScore != null) setScore(nextScore);

      setError('');
      setStatus('streaming');
    } catch (e) {
      const msg = String(e?.message || e);
      if (msg === 'Failed to fetch') {
        setError(
          `Failed to fetch (${apiEndpoint}). Possible causes: 1) wrong host/path, 2) CORS blocked, 3) service not publicly reachable.`
        );
      } else {
        setError(msg);
      }
      setStatus('error');
    } finally {
      inFlightRef.current = false;
    }
  }

  async function startPipeline() {
    try {
      setError('');
      setNote('');
      setStatus('starting');
      if (sourceMode === 'camera') {
        await startCamera();
      } else {
        await startUploadedVideo();
      }
      setStatus('streaming');
      startFrameLoop();
      window.addEventListener('resize', resizeOverlay);
    } catch (e) {
      setError(String(e.message || e));
      setStatus('error');
      stopPipeline();
    }
  }

  function handleSourceModeChange(next) {
    if (running) stopPipeline();
    setSourceMode(next);
  }

  function handleUploadFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (uploadedUrl) URL.revokeObjectURL(uploadedUrl);
    const url = URL.createObjectURL(file);
    setUploadedUrl(url);
    setUploadedName(file.name);
    setError('');
    setNote('');
    setStatus('ready');
    clearOverlay();
  }

  function handleVideoLoadedMetadata() {
    resizeOverlay();
  }

  function handleVideoEnded() {
    if (sourceMode === 'upload') {
      stopFrameLoop();
      setStatus('finished');
    }
  }

  function handleVideoPlay() {
    if (sourceMode !== 'upload') return;
    if (!running) {
      startFrameLoop();
      setStatus('streaming');
    }
  }

  function handleVideoPause() {
    if (sourceMode !== 'upload') return;
    if (running) {
      stopFrameLoop();
      setStatus('paused');
    }
  }

  async function handleReplay() {
    if (sourceMode !== 'upload') return;
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = 0;
    await video.play();
    handleVideoPlay();
  }

  function handleStop() {
    if (sourceMode === 'upload' && videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
      stopFrameLoop();
      setStatus('stopped');
      return;
    }
    stopPipeline();
  }

  return (
    <div className="app">
      <h1>Ghost Skeleton Live Viewer</h1>
      <div className="sub">MacBook camera -&gt; Cloud Run model API -&gt; Ghost Skeleton + score overlay</div>

      <div className="controls">
        <select value={sourceMode} onChange={(e) => handleSourceModeChange(e.target.value)}>
          <option value="camera">Camera</option>
          <option value="upload">Upload Video</option>
        </select>

        <input
          value={apiUrl}
          onChange={(e) => setApiUrl(e.target.value)}
          placeholder="Cloud Run endpoint"
        />

        <select value={fps} onChange={(e) => setFps(Number(e.target.value))}>
          <option value={2}>2 fps</option>
          <option value={4}>4 fps</option>
          <option value={6}>6 fps</option>
          <option value={8}>8 fps</option>
        </select>

        <select value={jpegQuality} onChange={(e) => setJpegQuality(Number(e.target.value))}>
          <option value={0.55}>JPEG 55%</option>
          <option value={0.7}>JPEG 70%</option>
          <option value={0.85}>JPEG 85%</option>
        </select>

        {!running ? (
          <button onClick={startPipeline}>{sourceMode === 'camera' ? 'Start Live' : 'Start Upload Analysis'}</button>
        ) : (
          <button onClick={handleStop}>Stop</button>
        )}
      </div>

      {sourceMode === 'upload' ? (
        <div className="controls" style={{ gridTemplateColumns: '1fr' }}>
          <input type="file" accept="video/*" onChange={handleUploadFileChange} />
          {uploadedName ? <div className="legend">Loaded: {uploadedName}</div> : null}
          {uploadedName ? <button onClick={handleReplay}>Replay From Start</button> : null}
        </div>
      ) : null}

      {endpointHint ? <div className="error">{endpointHint}</div> : null}

      <div className="stack">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          controls={sourceMode === 'upload'}
          onLoadedMetadata={handleVideoLoadedMetadata}
          onEnded={handleVideoEnded}
          onPlay={handleVideoPlay}
          onPause={handleVideoPause}
        />
        <canvas ref={canvasRef} />
      </div>

      <div className="hud">
        <div className="card">
          <div className="label">Status</div>
          <div className="value">{status}</div>
        </div>
        <div className="card">
          <div className="label">Score</div>
          <div className={`value ${scoreClass(score)}`}>{score == null ? '--' : score.toFixed(1)}</div>
        </div>
        <div className="card">
          <div className="label">Latency</div>
          <div className="value">{latency == null ? '--' : `${latency} ms`}</div>
        </div>
      </div>

      {error ? <div className="error">{error}</div> : null}
      {note ? <div className="error">{note}</div> : null}

      <div className="legend">
        Ghost skeleton: <span style={{ color: '#ff4be3' }}>magenta</span>. API expected to return JSON with score and landmarks.
      </div>

      <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
    </div>
  );
}
