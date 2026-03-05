import { RealtimeFault } from "./types";
/**
 * Sliding-window smoother that suppresses flickering faults.
 * A fault must appear in >= MIN_HITS of the last WINDOW frames to surface.
 * Returns at most TOP_N faults sorted by average zScore.
 */
export declare class FaultSmoother {
    private history;
    update(frameFaults: RealtimeFault[]): RealtimeFault[];
    reset(): void;
}
//# sourceMappingURL=faultSmoother.d.ts.map