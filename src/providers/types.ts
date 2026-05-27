/** Shared contract for every upscaling engine in the failover chain. */
export interface UpscaleProvider {
  /** Human-readable engine name (shown subtly in status, e.g. "NVIDIA PiD"). */
  readonly name: string;
  /** Lower tiers are tried first. 1 = Modal, 2 = HF Space, 3 = in-browser. */
  readonly tier: number;
  /** Whether this engine has the config it needs to run (skipped if false). */
  isConfigured(): boolean;
  /**
   * Upscale `input` and resolve with the result image blob.
   * Must reject promptly when `signal` aborts (timeout / failover).
   */
  run(input: Blob, ctx: RunContext): Promise<Blob>;
}

export interface RunContext {
  signal: AbortSignal;
  /** Neutral, user-facing progress copy. Never surfaces "busy"/"not ready". */
  onProgress?: (message: string) => void;
}

/** Result of a successful upscale, with which engine produced it. */
export interface UpscaleResult {
  blob: Blob;
  engine: string;
  tier: number;
}
