const WINDOW = 6;
const MIN_HITS = 4;
const TOP_N = 3;
/**
 * Sliding-window smoother that suppresses flickering faults.
 * A fault must appear in >= MIN_HITS of the last WINDOW frames to surface.
 * Returns at most TOP_N faults sorted by average zScore.
 */
export class FaultSmoother {
    constructor() {
        this.history = [];
    }
    update(frameFaults) {
        this.history.push(frameFaults);
        if (this.history.length > WINDOW)
            this.history.shift();
        const counts = new Map();
        for (const faults of this.history) {
            for (const f of faults) {
                const entry = counts.get(f.featureId);
                if (entry) {
                    entry.hits++;
                    entry.totalZ += f.zScore;
                    if (f.zScore > entry.fault.zScore)
                        entry.fault = f;
                }
                else {
                    counts.set(f.featureId, { hits: 1, totalZ: f.zScore, fault: f });
                }
            }
        }
        const stable = [];
        for (const entry of counts.values()) {
            if (entry.hits >= MIN_HITS) {
                stable.push({ avgZ: entry.totalZ / entry.hits, fault: entry.fault });
            }
        }
        stable.sort((a, b) => b.avgZ - a.avgZ);
        return stable.slice(0, TOP_N).map((s) => s.fault);
    }
    reset() {
        this.history = [];
    }
}
