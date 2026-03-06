/**
 * Estimate online phase (0..99) using normalized raise angle progress.
 */
export function estimatePhase(progress) {
    const clamped = Math.max(0, Math.min(1, progress));
    return Math.round(clamped * 99);
}
