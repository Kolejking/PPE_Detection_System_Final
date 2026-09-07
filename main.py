from pathlib import Path

import cv2

from src.detector import PPEDetector
from src.tracker import PPETracker
from src.video_processor import VideoProcessor
from src.database import PPEDatabase

from src.ppe_association import (
    extract_detections,
    remove_duplicate_persons,
    split_detections,
    associate_ppe_to_persons,
    build_ppe_status,
    draw_ppe_association,
    MIN_ASSOCIATION_SCORE
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

MODEL_PATH = "models/best1.pt"

TRACKER_PATH = "configs/bootsort.yaml"


# Choose:
#
# "image" -> YOLO + refined PPE association
#
# "video" -> YOLO + BoT-SORT + refined PPE association

MODE = "image"


IMAGE_PATH = (
    "inputs/images/plantimage17.jpg"
)


VIDEO_PATH = (
    "inputs/videos/ppe_video.mp4"
)


CONFIDENCE = 0.25


MIN_SCORE = (
    MIN_ASSOCIATION_SCORE
)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Load YOLO11n model
    # --------------------------------------------------------

    detector = PPEDetector(
        MODEL_PATH,
        device="cpu"
    )

    print(
        "✅ Model loaded"
    )

    print(
        "Classes:",
        detector.model.names
    )

    print(
        "Minimum association score:",
        MIN_SCORE
    )


    # ========================================================
    # DATABASE
    # ========================================================

    database = PPEDatabase()

    print(
        "✅ SQLite database connected"
    )


    # ========================================================
    # IMAGE MODE
    # ========================================================

    if MODE == "image":

        print(
            "\n🖼️ Running refined "
            "image association..."
        )


        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------

        results = detector.predict(
            IMAGE_PATH,
            conf=CONFIDENCE
        )

        result = results[0]


        # ----------------------------------------------------
        # Extract detections
        # ----------------------------------------------------

        detections = extract_detections(
            result,
            detector.model.names
        )


        # ----------------------------------------------------
        # Remove duplicate Person boxes
        # ----------------------------------------------------

        detections = remove_duplicate_persons(
            detections
        )


        # ----------------------------------------------------
        # Split all 9 classes
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Detection summary
        # ----------------------------------------------------

        print(
            "\n========== DETECTION SUMMARY =========="
        )

        print(
            "Persons              :",
            len(persons)
        )

        print(
            "Helmets              :",
            len(helmets)
        )

        print(
            "No-Helmets           :",
            len(no_helmets)
        )

        print(
            "Vests                :",
            len(vests)
        )

        print(
            "No-Vests             :",
            len(no_vests)
        )

        print(
            "Safety-Shoes         :",
            len(safety_shoes)
        )

        print(
            "No-Safety-Shoes      :",
            len(no_safety_shoes)
        )

        print(
            "Without-Safety-Goggles:",
            len(without_safety_goggles)
        )

        print(
            "With-Safety-Goggles  :",
            len(with_safety_goggles)
        )

        print(
            "======================================="
        )


        # ----------------------------------------------------
        # Refined PPE association
        # ----------------------------------------------------

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

                min_score=MIN_SCORE
            )
        )


        # ----------------------------------------------------
        # Build PPE status
        # ----------------------------------------------------

        ppe_status = (
            build_ppe_status(
                associations
            )
        )


        # ----------------------------------------------------
        # Print person-level PPE status
        # ----------------------------------------------------

        print(
            "\n========== PERSON PPE STATUS =========="
        )

        for person in ppe_status:

            print(
                f'\nPerson '
                f'{person["person_index"]}'
            )

            print(
                "Track ID:",
                person.get("track_id")
            )

            print(
                "Helmet:",
                person["helmet_status"],
                f'| score='
                f'{person["helmet_score"]:.3f}'
            )

            print(
                "Vest:",
                person["vest_status"],
                f'| score='
                f'{person["vest_score"]:.3f}'
            )

            print(
                "Safety-Shoes:",
                person["safety_shoes_status"],
                f'| score='
                f'{person["safety_shoes_score"]:.3f}'
            )

            print(
                "Safety-Goggles:",
                person["safety_goggles_status"],
                f'| score='
                f'{person["safety_goggles_score"]:.3f}'
            )

            print(
                "Overall Status:",
                person.get("overall_status")
            )

        print(
            "\n======================================="
        )


        # ====================================================
        # SAVE PPE RESULTS TO DATABASE
        # ====================================================

        if ppe_status:

            database.insert_detections(
                statuses=ppe_status,
                source=IMAGE_PATH
            )

            print(
                f"\n✅ Saved "
                f"{len(ppe_status)} PPE record(s) "
                f"to SQLite database"
            )

        else:

            print(
                "\n⚠️ No persons detected. "
                "Nothing saved to database."
            )


        # ----------------------------------------------------
        # Read original image
        # ----------------------------------------------------

        image = cv2.imread(
            IMAGE_PATH
        )

        if image is None:

            raise RuntimeError(
                f"Could not load image: "
                f"{IMAGE_PATH}"
            )


        # ----------------------------------------------------
        # Draw refined association
        # ----------------------------------------------------

        association_image = (
            draw_ppe_association(
                image,
                persons,
                ppe_status,
                associations
            )
        )


        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        output_dir = Path(
            "outputs/images"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        input_name = Path(
            IMAGE_PATH
        ).stem

        output_path = (
            output_dir
            /
            f"{input_name}"
            "_refined_ppe_association.jpg"
        )

        success = cv2.imwrite(
            str(output_path),
            association_image
        )

        if not success:

            raise RuntimeError(
                "Could not save image output."
            )


        print(
            "\n✅ Refined image "
            "association completed"
        )

        print(
            "Input :",
            IMAGE_PATH
        )

        print(
            "Output:",
            output_path
        )


        # ----------------------------------------------------
        # Database record count
        # ----------------------------------------------------

        print(
            "Database records:",
            database.count_records()
        )


    # ========================================================
    # VIDEO MODE
    # ========================================================

    elif MODE == "video":

        print(
            "\n🎥 Starting refined "
            "video tracking + PPE association..."
        )


        # ----------------------------------------------------
        # Create BoT-SORT tracker
        # ----------------------------------------------------

        tracker = PPETracker(
            detector.model,
            TRACKER_PATH,
            device="cpu"
        )

        print(
            "✅ Tracker loaded"
        )


        # ----------------------------------------------------
        # Generate tracking results
        # ----------------------------------------------------

        results = tracker.track(
            VIDEO_PATH,
            conf=CONFIDENCE
        )


        # ----------------------------------------------------
        # Process video
        # ----------------------------------------------------

        processor = (
            VideoProcessor()
        )

        output_path = (
            processor.save_tracking_ppe_video(
                results=results,
                input_video=VIDEO_PATH,
                class_names=detector.model.names,
                min_score=MIN_SCORE
            )
        )


        print(
            "\n✅ Refined video "
            "tracking + PPE association completed"
        )

        print(
            "Input :",
            VIDEO_PATH
        )

        print(
            "Output:",
            output_path
        )


    # ========================================================
    # INVALID MODE
    # ========================================================

    else:

        raise ValueError(
            "MODE must be either "
            "'image' or 'video'"
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
