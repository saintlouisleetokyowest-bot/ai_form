import type { IPoseCorrector, CorrectorInput, CorrectorOutput } from "@ai-form/core";

/** Mock corrector: passthrough landmarks, zero confidence. */
export class MockPoseCorrector implements IPoseCorrector {
  ready = true;

  async load(): Promise<void> { /* no-op */ }

  run(input: CorrectorInput): CorrectorOutput {
    return MockPoseCorrector.passthrough(input);
  }

  static passthrough(input: CorrectorInput): CorrectorOutput {
    return {
      correctedLandmarks: new Float32Array(input.bodyLandmarks),
      confidence: new Float32Array(33).fill(0),
    };
  }
}

/** ONNX corrector: input landmarks [1,33,3] + phase [1,1]; output corrected [1,33,3], confidence [1,33]. */
export class OnnxPoseCorrector implements IPoseCorrector {
  private session: unknown = null;
  private cachedOutput: CorrectorOutput | null = null;
  ready = false;

  constructor(private modelUrl: string) {}

  async load(): Promise<void> {
    // #region agent log
    fetch("http://127.0.0.1:7259/ingest/88bfe024-b363-441f-ace3-aa21efba9052", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        location: "corrector.ts:load",
        message: "OnnxPoseCorrector.load start",
        data: { modelUrl: this.modelUrl },
        timestamp: Date.now(),
        hypothesisId: "A",
      }),
    }).catch(() => {});
    // #endregion
    try {
      const ort = await import("onnxruntime-web");
      // Vite serves .wasm with wrong MIME / 404 HTML. Use CDN for WASM binaries.
      // Must match installed version (1.24.2) - 1.17.1 lacks .mjs on CDN.
      const ortVersion = "1.24.2";
      ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/`;
      this.session = await ort.InferenceSession.create(this.modelUrl, {
        executionProviders: ["wasm"],
      });
      this.ready = true;
      // #region agent log
      fetch("http://127.0.0.1:7259/ingest/88bfe024-b363-441f-ace3-aa21efba9052", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          location: "corrector.ts:load",
          message: "OnnxPoseCorrector.load success",
          data: { ready: this.ready },
          timestamp: Date.now(),
          hypothesisId: "A",
        }),
      }).catch(() => {});
      // #endregion
    } catch (e) {
      // #region agent log
      fetch("http://127.0.0.1:7259/ingest/88bfe024-b363-441f-ace3-aa21efba9052", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          location: "corrector.ts:load",
          message: "OnnxPoseCorrector.load failed",
          data: { error: String(e) },
          timestamp: Date.now(),
          hypothesisId: "A",
        }),
      }).catch(() => {});
      // #endregion
      console.warn("Failed to load corrector model, falling back to mock:", e);
      this.ready = false;
    }
  }

  run(input: CorrectorInput): CorrectorOutput {
    if (!this.session) return MockPoseCorrector.passthrough(input);
    return this.cachedOutput ?? MockPoseCorrector.passthrough(input);
  }

  /** Async inference; call this and use the result to update UI state. */
  async runAsync(input: CorrectorInput): Promise<CorrectorOutput> {
    if (!this.session) return MockPoseCorrector.passthrough(input);

    const ort = await import("onnxruntime-web");
    const landmarksTensor = new ort.Tensor("float32", input.bodyLandmarks, [1, 33, 3]);
    const phaseNorm = Math.min(1, Math.max(0, input.phase / 99));
    const phaseTensor = new ort.Tensor("float32", new Float32Array([phaseNorm]), [1, 1]);

    const feeds = { landmarks: landmarksTensor, phase: phaseTensor };
    const results = await (this.session as { run: (f: typeof feeds) => Promise<Record<string, { data: Float32Array }>> }).run(feeds);

    const corrected = results.corrected;
    const confidence = results.confidence;
    const correctedData = corrected?.data as Float32Array;
    const confidenceData = confidence?.data as Float32Array;

    const output: CorrectorOutput = {
      correctedLandmarks: correctedData ? new Float32Array(correctedData) : new Float32Array(input.bodyLandmarks),
      confidence: confidenceData ? new Float32Array(confidenceData) : new Float32Array(33).fill(0),
    };
    this.cachedOutput = output;
    return output;
  }
}

/**
 * Factory: provide a model URL to get OnnxPoseCorrector, otherwise MockPoseCorrector.
 */
export function createCorrector(modelUrl?: string): IPoseCorrector {
  if (modelUrl) return new OnnxPoseCorrector(modelUrl);
  return new MockPoseCorrector();
}
