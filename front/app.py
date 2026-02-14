import time
import cv2
import mediapipe as mp


def main():
    # ---- Camera settings ----
    camera_index = 0  # 0 is usually the built-in webcam; try 1 if you have an external cam
    desired_width = 1280
    desired_height = 720

    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)  # AVFoundation works well on macOS
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing camera_index or permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # # ---- MediaPipe Pose ----
    # mp_pose = mp.solutions.pose
    # mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles

    # Model complexity: 0 (light/fast), 1 (default), 2 (heavy/accurate)
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    prev_t = time.time()
    fps = 0.0

    window_name = "MediaPipe Pose (Press q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read frame from camera.")
                break

            # MediaPipe expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # For performance, mark as not writeable while processing
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # Draw pose landmarks on the original BGR frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            # FPS calculation
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)  # smoothed FPS

            # Overlay text
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "q: quit | s: screenshot",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")

    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
