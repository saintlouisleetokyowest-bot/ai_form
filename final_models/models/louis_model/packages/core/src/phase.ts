/** Phase 0..99 from normalized raise angle progress. */
export function estimatePhase(progress: number): number {
  const clamped = Math.max(0, Math.min(1, progress));
  return Math.round(clamped * 99);
}
