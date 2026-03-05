import type { IPoseCorrector, CorrectorInput, CorrectorOutput } from "@ai-form/core";

/**
 * Mock corrector: returns the user's original landmarks unchanged.
 * confidence is all-zero so the UI knows there is no real correction.
 * Use this while the ML model has not been trained yet.
 */
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

/**
 * ONNX-based corrector. Loads a trained .onnx model and runs per-frame
 * inference to produce corrected body-frame landmarks.
 *
 * Expected model I/O (opset 15+):
 *   input  "landmarks"  : float32 [1, 33, 3]
 *   input  "phase"      : float32 [1, 1]      (normalised 0-1)
 *   output "corrected"  : float32 [1, 33, 3]
 *   output "confidence" : float32 [1, 33]
 */
export class OnnxPoseCorrector implements IPoseCorrector {
  private session: any = null;
  ready = false;

  constructor(private modelUrl: string) {}

  async load(): Promise<void> {
    try {
      const ort = await import("onnxruntime-web");
      this.session = await ort.InferenceSession.create(this.modelUrl, {
        executionProviders: ["wasm"],
      });
      this.ready = true;
    } catch (e) {
      console.warn("Failed to load corrector model, falling back to mock:", e);
      this.ready = false;
    }
  }

  run(input: CorrectorInput): CorrectorOutput {
    if (!this.session) return MockPoseCorrector.passthrough(input);

    // Synchronous fallback — real async inference would need a different
    // calling pattern (e.g. double-buffer). For now, use the mock when
    // the model is loaded but inference is async.
    // TODO: implement async run() when the real model is available.
    return MockPoseCorrector.passthrough(input);
  }
}

/**
 * Factory: provide a model URL to get OnnxPoseCorrector, otherwise MockPoseCorrector.
 */
export function createCorrector(modelUrl?: string): IPoseCorrector {
  if (modelUrl) return new OnnxPoseCorrector(modelUrl);
  return new MockPoseCorrector();
}
