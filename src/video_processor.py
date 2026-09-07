from pathlib import Path

import cv2

from src.ppe_association import (
    extract_detections,
    remove_duplicate_persons,
    split_detections,
    associate_ppe_to_persons,
    build_ppe_status,
    draw_ppe_association
)

from src.database import PPEDatabase


class VideoProcessor:

    def __init__(
        self,
        output_dir: str = "outputs/videos"
    ):
        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ====================================================
        # DATABASE
        # ====================================================

        self.db = PPEDatabase()

    def save_tracking_ppe_video(
        self,
        results,
        input_video: str,
        class_names,
        min_score: float = 0.45
    ):
        """
        Process YOLO + BoT-SORT results frame-by-frame.

        Pipeline:

            YOLO
              ↓
            BoT-SORT
              ↓
            Detection extraction
              ↓
            Duplicate Person filtering
              ↓
            PPE association
              ↓
            PPE status
              ↓
            Database
              ↓
            Visualization
              ↓
            Output video
        """

        input_path = Path(
            input_video
        )

        # ====================================================
        # READ VIDEO PROPERTIES
        # ====================================================

        cap = cv2.VideoCapture(
            str(input_path)
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Could not open video: "
                f"{input_path}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        cap.release()

        if fps <= 0:
            fps = 25.0

        # ====================================================
        # OUTPUT PATH
        # ====================================================

        output_path = (
            self.output_dir
            /
            f"{input_path.stem}"
            "_ppe_tracking.mp4"
        )

        # ====================================================
        # VIDEO WRITER
        # ====================================================

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():

            raise RuntimeError(
                "Could not create output video."
            )

        frame_count = 0

        print(
            "\nStarting PPE association "
            "validation video..."
        )

        # ====================================================
        # PROCESS EVERY RESULT
        # ====================================================

        for result in results:

            # ------------------------------------------------
            # 1. Extract detections
            # ------------------------------------------------

            detections = (
                extract_detections(
                    result,
                    class_names
                )
            )

            # ------------------------------------------------
            # 2. Remove duplicate Person boxes
            # ------------------------------------------------

            detections = (
                remove_duplicate_persons(
                    detections
                )
            )

            # ------------------------------------------------
            # 3. Print current Person IDs
            # ------------------------------------------------

            person_ids = [
                detection["track_id"]
                for detection in detections
                if detection["class_name"]
                == "Person"
            ]

            print(
                f"Frame {frame_count}: "
                f"Person IDs = {person_ids}"
            )

            # ------------------------------------------------
            # 4. Separate ALL 9 classes
            # ------------------------------------------------

            (
                persons,
                helmets,
                no_helmets,
                vests,
                no_vests,
                safety_shoes,
                no_safety_shoes,
                without_safety_goggles,
                with_safety_goggles
            ) = split_detections(
                detections
            )

            # ------------------------------------------------
            # 5. Refined PPE association
            # ------------------------------------------------

            associations = (
                associate_ppe_to_persons(
                    persons=persons,

                    helmets=helmets,
                    no_helmets=no_helmets,

                    vests=vests,
                    no_vests=no_vests,

                    safety_shoes=safety_shoes,
                    no_safety_shoes=no_safety_shoes,

                    without_safety_goggles=(
                        without_safety_goggles
                    ),

                    with_safety_goggles=(
                        with_safety_goggles
                    ),

                    min_score=min_score
                )
            )

            # ------------------------------------------------
            # 6. Build clean PPE status
            # ------------------------------------------------

            ppe_status = (
                build_ppe_status(
                    associations
                )
            )

            # ------------------------------------------------
            # 7. Print PPE status
            # ------------------------------------------------

            for status in ppe_status:

                track_id = (
                    status["track_id"]
                )

                print(
                    f"  ID {track_id} -> "
                    f"Helmet="
                    f"{status['helmet_status']} "
                    f"(score="
                    f"{status['helmet_score']:.3f}) | "
                    f"Vest="
                    f"{status['vest_status']} "
                    f"(score="
                    f"{status['vest_score']:.3f}) | "
                    f"Shoes="
                    f"{status['safety_shoes_status']} "
                    f"(score="
                    f"{status['safety_shoes_score']:.3f}) | "
                    f"Goggles="
                    f"{status['safety_goggles_status']} "
                    f"(score="
                    f"{status['safety_goggles_score']:.3f})"
                )

            # ------------------------------------------------
            # 8. Get original frame
            # ------------------------------------------------

            annotated_frame = (
                result.orig_img.copy()
            )

            # ------------------------------------------------
            # 9. Draw final visualization
            # ------------------------------------------------

            annotated_frame = (
                draw_ppe_association(
                    annotated_frame,
                    persons,
                    ppe_status,
                    associations
                )
            )

            # ------------------------------------------------
            # 10. Write frame
            # ------------------------------------------------

            writer.write(
                annotated_frame
            )

            frame_count += 1

            if frame_count % 25 == 0:

                print(
                    f"Processed "
                    f"{frame_count} frames..."
                )

        # ====================================================
        # RELEASE
        # ====================================================

        writer.release()

        print()

        print(
            "✅ PPE association validation "
            "video saved"
        )

        print(
            f"Total frames processed: "
            f"{frame_count}"
        )

        print(
            f"Output: {output_path}"
        )

        return output_path
    