import * as ort from "onnxruntime-web";

export interface DenoiserInput {
  data: Float32Array; // shape: [T, 33, 3]
  timesteps: number;
}

export class PoseDenoiser {
  private session: ort.InferenceSession | null = null;

  constructor(private modelUrl: string) {}

  async load(): Promise<void> {
    if (this.session) return;
    this.session = await ort.InferenceSession.create(this.modelUrl, {
      executionProviders: ["wasm"],
    });
  }

  async run(input: DenoiserInput): Promise<Float32Array> {
    if (!this.session) throw new Error("Denoiser session not loaded");
    const { data, timesteps } = input;
    const tensor = new ort.Tensor("float32", data, [timesteps, 33, 3]);
    const feeds: Record<string, ort.Tensor> = { input: tensor };
    const outputs = await this.session.run(feeds);
    const first = outputs[Object.keys(outputs)[0]];
    return first.data as Float32Array;
  }
}
