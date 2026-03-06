export const EPS = 1e-6;
export function add(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
export function sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
export function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
export function cross(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}
export function norm(a) {
    return Math.sqrt(dot(a, a));
}
export function normalize(a) {
    const n = norm(a);
    if (n < EPS)
        return [0, 0, 0];
    return [a[0] / n, a[1] / n, a[2] / n];
}
export function scale(a, k) {
    return [a[0] * k, a[1] * k, a[2] * k];
}
export function project(v, axis) {
    const k = dot(v, axis) / (norm(axis) ** 2 + EPS);
    return scale(axis, k);
}
export function angleBetween(a, b) {
    const cos = dot(a, b) / ((norm(a) * norm(b)) + EPS);
    const clamped = Math.min(1, Math.max(-1, cos));
    return Math.acos(clamped); // radians
}
export function degrees(rad) {
    return (rad * 180) / Math.PI;
}
export function median(values) {
    if (values.length === 0)
        return 0;
    const sorted = [...values].sort((x, y) => x - y);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
export function lerp(a, b, t) {
    return a + (b - a) * t;
}
export function clamp(v, min, max) {
    return Math.min(max, Math.max(min, v));
}
