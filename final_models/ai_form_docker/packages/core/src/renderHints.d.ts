import { Fault, RealtimeFault } from "./types";
export interface RenderHint {
    joints: number[];
    highlightColor: string;
    label: string;
}
export declare function renderHintsForFaults(faults: Fault[]): RenderHint[];
export declare function renderHintsForRealtimeFaults(faults: RealtimeFault[]): RenderHint[];
//# sourceMappingURL=renderHints.d.ts.map