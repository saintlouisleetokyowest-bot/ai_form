import { Vec3 } from "./types.js";
export declare const EPS = 0.000001;
export declare function add(a: Vec3, b: Vec3): Vec3;
export declare function sub(a: Vec3, b: Vec3): Vec3;
export declare function dot(a: Vec3, b: Vec3): number;
export declare function cross(a: Vec3, b: Vec3): Vec3;
export declare function norm(a: Vec3): number;
export declare function normalize(a: Vec3): Vec3;
export declare function scale(a: Vec3, k: number): Vec3;
export declare function project(v: Vec3, axis: Vec3): Vec3;
export declare function angleBetween(a: Vec3, b: Vec3): number;
export declare function degrees(rad: number): number;
export declare function median(values: number[]): number;
export declare function lerp(a: number, b: number, t: number): number;
export declare function clamp(v: number, min: number, max: number): number;
//# sourceMappingURL=math.d.ts.map