import { BodyFrame, CorrectorOutput, Landmark, GHOST_JOINT_IDS, Vec3 } from "./types";
import { add, scale as vecScale } from "./math";

/**
 * Project ML-corrected body-frame landmarks back to image space.
 *
 * A-class joints (upper body) use ML output; B-class joints passthrough
 * from the user's actual image landmarks.
 */
export function projectCorrectionToImage(
  correction: CorrectorOutput,
  bodyFrame: BodyFrame,
  userImageLandmarks: Landmark[],
): Array<{ x: number; y: number }> {
  const result: Array<{ x: number; y: number }> = new Array(33);

  // Build a simple affine mapping from world→image using the user's
  // shoulder midpoint as an anchor. This is approximate but avoids
  // a full camera calibration.
  const ls = userImageLandmarks[11];
  const rs = userImageLandmarks[12];
  const lh = userImageLandmarks[23];
  const rh = userImageLandmarks[24];

  if (!ls || !rs) {
    for (let i = 0; i < 33; i++) {
      const lm = userImageLandmarks[i];
      result[i] = lm ? { x: lm.x, y: lm.y } : { x: 0, y: 0 };
    }
    return result;
  }

  // Approximate world-to-image scale: compare body-frame shoulder
  // distance (normalised) with image shoulder distance.
  const imgShoulderDist = Math.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2);
  const bfLS = bodyFrame.bodyLandmarks[11];
  const bfRS = bodyFrame.bodyLandmarks[12];
  const bfShoulderDist = bfLS && bfRS
    ? Math.sqrt((bfLS[0] - bfRS[0]) ** 2 + (bfLS[1] - bfRS[1]) ** 2 + (bfLS[2] - bfRS[2]) ** 2)
    : 1;
  const w2i = bfShoulderDist > 1e-6 ? imgShoulderDist / bfShoulderDist : 1;

  // Image anchor (shoulder midpoint)
  const imgAnchorX = (ls.x + rs.x) / 2;
  const imgAnchorY = (ls.y + rs.y) / 2;

  // Body-frame anchor
  const bfAnchorX = bfLS && bfRS ? (bfLS[0] + bfRS[0]) / 2 : 0;
  const bfAnchorY = bfLS && bfRS ? (bfLS[1] + bfRS[1]) / 2 : 0;

  const aClassSet = new Set<number>(GHOST_JOINT_IDS);

  for (let i = 0; i < 33; i++) {
    if (aClassSet.has(i) && correction.confidence[i] > 0.1) {
      // Use ML-corrected position
      const cx = correction.correctedLandmarks[i * 3];
      const cy = correction.correctedLandmarks[i * 3 + 1];
      // Project: relative to body-frame anchor, scaled to image space
      const dx = (cx - bfAnchorX) * w2i;
      const dy = (cy - bfAnchorY) * w2i;
      // body-frame y is up, image y is down → negate dy
      result[i] = { x: imgAnchorX + dx, y: imgAnchorY - dy };
    } else {
      // Passthrough user's actual position
      const lm = userImageLandmarks[i];
      result[i] = lm ? { x: lm.x, y: lm.y } : { x: 0, y: 0 };
    }
  }

  return result;
}
