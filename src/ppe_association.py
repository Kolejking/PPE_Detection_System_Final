from typing import Dict, List, Tuple
import math

import cv2


# ============================================================
# ASSOCIATION CONFIGURATION
# ============================================================

OVERLAP_WEIGHT = 0.50
POSITION_WEIGHT = 0.25
DISTANCE_WEIGHT = 0.15
CONFIDENCE_WEIGHT = 0.10


# Expected vertical position inside a Person box
#
# 0.0 = top of person
# 1.0 = bottom of person
#
EXPECTED_POSITION = {
    "head": 0.18,
    "goggles": 0.18,
    "torso": 0.50,
    "feet": 0.90
}


POSITION_TOLERANCE = {
    "head": 0.20,
    "goggles": 0.18,
    "torso": 0.30,
    "feet": 0.20
}


MIN_ASSOCIATION_SCORE = 0.45


# ============================================================
# DUPLICATE PERSON CONFIGURATION
# ============================================================

DUPLICATE_CONTAINMENT_THRESHOLD = 0.80
DUPLICATE_CENTER_DISTANCE_THRESHOLD = 0.20


# ============================================================
# 1. EXTRACT YOLO / BOT-SORT DETECTIONS
# ============================================================

def extract_detections(
    result,
    class_names
) -> List[Dict]:
    """
    Convert YOLO / BoT-SORT results into dictionaries.

    Each detection contains:
        class_id
        class_name
        confidence
        box
        track_id
    """

    detections = []

    track_ids = None

    if result.boxes.id is not None:
        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

    for index, box in enumerate(result.boxes):

        class_id = int(
            box.cls[0].cpu()
        )

        confidence = float(
            box.conf[0].cpu()
        )

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .tolist()
        )

        track_id = None

        if track_ids is not None:
            track_id = track_ids[index]

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": confidence,
                "box": [
                    x1,
                    y1,
                    x2,
                    y2
                ],
                "track_id": track_id
            }
        )

    return detections


# ============================================================
# 2. SPLIT DETECTIONS BY CLASS
# ============================================================

def split_detections(
    detections: List[Dict]
) -> Tuple[
    List[Dict],
    List[Dict],
    List[Dict],
    List[Dict],
    List[Dict],
    List[Dict],
    List[Dict],
    List[Dict],
    List[Dict]
]:
    """
    Separate detections into all 9 PPE classes:

        Person
        Helmet
        No-Helmet
        Vest
        No-Vest
        Safety-Shoes
        No-Safety-Shoes
        Without-Safety-Goggles
        With-Safety-Goggles
    """

    persons = []
    helmets = []
    no_helmets = []
    vests = []
    no_vests = []
    safety_shoes = []
    no_safety_shoes = []
    without_safety_goggles = []
    with_safety_goggles = []

    for detection in detections:

        class_name = detection["class_name"]

        if class_name == "Person":
            persons.append(detection)

        elif class_name == "Helmet":
            helmets.append(detection)

        elif class_name == "No-Helmet":
            no_helmets.append(detection)

        elif class_name == "Vest":
            vests.append(detection)

        elif class_name == "No-Vest":
            no_vests.append(detection)

        elif class_name == "Safety-Shoes":
            safety_shoes.append(detection)

        elif class_name == "No-Safety-Shoes":
            no_safety_shoes.append(detection)

        elif class_name == "Without-Safety-Goggles":
            without_safety_goggles.append(detection)

        elif class_name == "With-Safety-Goggles":
            with_safety_goggles.append(detection)

    return (
        persons,
        helmets,
        no_helmets,
        vests,
        no_vests,
        safety_shoes,
        no_safety_shoes,
        without_safety_goggles,
        with_safety_goggles
    )


# ============================================================
# 3. BOX CENTER
# ============================================================

def get_box_center(
    box: List[float]
) -> Tuple[float, float]:
    """
    Return center of [x1, y1, x2, y2].
    """

    x1, y1, x2, y2 = box

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    return (
        center_x,
        center_y
    )


# ============================================================
# 4. PERSON REGIONS
# ============================================================

def get_person_regions(
    person_box: List[float]
) -> Dict[str, List[float]]:
    """
    Approximate PPE regions inside a Person box.

    Regions:

        head:
            top 35%

        goggles:
            top 30%

        torso:
            25% -> 75%

        feet:
            bottom 25%
    """

    x1, y1, x2, y2 = person_box

    height = (
        y2 - y1
    )

    # --------------------------------------------------------
    # HEAD
    # --------------------------------------------------------

    head_y2 = (
        y1 +
        0.35 * height
    )

    head_region = [
        x1,
        y1,
        x2,
        head_y2
    ]

    # --------------------------------------------------------
    # GOGGLES
    # --------------------------------------------------------

    goggles_y2 = (
        y1 +
        0.30 * height
    )

    goggles_region = [
        x1,
        y1,
        x2,
        goggles_y2
    ]

    # --------------------------------------------------------
    # TORSO
    # --------------------------------------------------------

    torso_y1 = (
        y1 +
        0.25 * height
    )

    torso_y2 = (
        y1 +
        0.75 * height
    )

    torso_region = [
        x1,
        torso_y1,
        x2,
        torso_y2
    ]

    # --------------------------------------------------------
    # FEET
    # --------------------------------------------------------

    feet_y1 = (
        y1 +
        0.75 * height
    )

    feet_region = [
        x1,
        feet_y1,
        x2,
        y2
    ]

    return {
        "head": head_region,
        "goggles": goggles_region,
        "torso": torso_region,
        "feet": feet_region
    }


# ============================================================
# 5. BOX AREA
# ============================================================

def calculate_box_area(
    box: List[float]
) -> float:
    """
    Calculate area of a bounding box.
    """

    x1, y1, x2, y2 = box

    width = max(
        0.0,
        x2 - x1
    )

    height = max(
        0.0,
        y2 - y1
    )

    return (
        width *
        height
    )


# ============================================================
# 6. INTERSECTION AREA
# ============================================================

def calculate_intersection_area(
    box_a: List[float],
    box_b: List[float]
) -> float:
    """
    Calculate intersection area between two boxes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(
        ax1,
        bx1
    )

    intersection_y1 = max(
        ay1,
        by1
    )

    intersection_x2 = min(
        ax2,
        bx2
    )

    intersection_y2 = min(
        ay2,
        by2
    )

    if (
        intersection_x2 <= intersection_x1
        or
        intersection_y2 <= intersection_y1
    ):
        return 0.0

    return (
        (intersection_x2 - intersection_x1)
        *
        (intersection_y2 - intersection_y1)
    )


# ============================================================
# 7. PPE OVERLAP SCORE
# ============================================================

def calculate_overlap_ratio(
    ppe_box: List[float],
    region_box: List[float]
) -> float:
    """
    Fraction of PPE box that lies inside the region.

    0.0 -> no overlap
    1.0 -> entire PPE box inside region
    """

    ppe_area = calculate_box_area(
        ppe_box
    )

    if ppe_area <= 0:
        return 0.0

    intersection_area = (
        calculate_intersection_area(
            ppe_box,
            region_box
        )
    )

    return (
        intersection_area /
        ppe_area
    )


# ============================================================
# 8. CENTER DISTANCE SCORE
# ============================================================

def calculate_center_distance_score(
    ppe_box: List[float],
    region_box: List[float],
    person_box: List[float]
) -> float:
    """
    Normalized distance between PPE center
    and region center.

    1.0 -> very close
    0.0 -> far away
    """

    ppe_center_x, ppe_center_y = (
        get_box_center(
            ppe_box
        )
    )

    region_center_x, region_center_y = (
        get_box_center(
            region_box
        )
    )

    distance = math.sqrt(
        (
            ppe_center_x -
            region_center_x
        ) ** 2
        +
        (
            ppe_center_y -
            region_center_y
        ) ** 2
    )

    person_x1, person_y1, person_x2, person_y2 = (
        person_box
    )

    person_width = (
        person_x2 -
        person_x1
    )

    person_height = (
        person_y2 -
        person_y1
    )

    person_diagonal = math.sqrt(
        person_width ** 2
        +
        person_height ** 2
    )

    if person_diagonal <= 0:
        return 0.0

    normalized_distance = (
        distance /
        person_diagonal
    )

    score = (
        1.0 -
        normalized_distance
    )

    return max(
        0.0,
        min(
            1.0,
            score
        )
    )


# ============================================================
# 9. EXPECTED POSITION SCORE
# ============================================================

def calculate_position_score(
    ppe_box: List[float],
    person_box: List[float],
    region_type: str
) -> float:
    """
    Measure how close PPE center is to expected
    vertical position.
    """

    if region_type not in EXPECTED_POSITION:
        return 0.0

    person_x1, person_y1, person_x2, person_y2 = (
        person_box
    )

    person_height = (
        person_y2 -
        person_y1
    )

    if person_height <= 0:
        return 0.0

    _, ppe_center_y = (
        get_box_center(
            ppe_box
        )
    )

    normalized_y = (
        ppe_center_y -
        person_y1
    ) / person_height

    normalized_y = max(
        0.0,
        min(
            1.0,
            normalized_y
        )
    )

    expected_y = (
        EXPECTED_POSITION[
            region_type
        ]
    )

    tolerance = (
        POSITION_TOLERANCE[
            region_type
        ]
    )

    difference = abs(
        normalized_y -
        expected_y
    )

    if difference >= tolerance:
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            1.0 -
            difference /
            tolerance
        )
    )


# ============================================================
# 10. COMBINED ASSOCIATION SCORE
# ============================================================

def calculate_association_score(
    ppe: Dict,
    person: Dict,
    region_type: str
) -> Dict[str, float]:
    """
    Final score:

        50% overlap
        25% position
        15% distance
        10% confidence
    """

    person_box = person["box"]
    ppe_box = ppe["box"]

    regions = get_person_regions(
        person_box
    )

    region = regions[
        region_type
    ]

    overlap_score = (
        calculate_overlap_ratio(
            ppe_box,
            region
        )
    )

    position_score = (
        calculate_position_score(
            ppe_box,
            person_box,
            region_type
        )
    )

    distance_score = (
        calculate_center_distance_score(
            ppe_box,
            region,
            person_box
        )
    )

    confidence_score = max(
        0.0,
        min(
            1.0,
            float(
                ppe["confidence"]
            )
        )
    )

    final_score = (
        OVERLAP_WEIGHT *
        overlap_score
        +
        POSITION_WEIGHT *
        position_score
        +
        DISTANCE_WEIGHT *
        distance_score
        +
        CONFIDENCE_WEIGHT *
        confidence_score
    )

    return {
        "final": final_score,
        "overlap": overlap_score,
        "position": position_score,
        "distance": distance_score,
        "confidence": confidence_score
    }


# ============================================================
# 11. DUPLICATE PERSON FUNCTIONS
# ============================================================

def calculate_containment_ratio(
    box_a: List[float],
    box_b: List[float]
) -> float:
    """
    Fraction of the smaller box contained inside
    the larger box.
    """

    area_a = calculate_box_area(
        box_a
    )

    area_b = calculate_box_area(
        box_b
    )

    if (
        area_a <= 0
        or
        area_b <= 0
    ):
        return 0.0

    intersection_area = (
        calculate_intersection_area(
            box_a,
            box_b
        )
    )

    smaller_area = min(
        area_a,
        area_b
    )

    return (
        intersection_area /
        smaller_area
    )


def calculate_center_distance_normalized(
    box_a: List[float],
    box_b: List[float]
) -> float:
    """
    Normalize center distance using the diagonal
    of the larger box.
    """

    center_a = get_box_center(
        box_a
    )

    center_b = get_box_center(
        box_b
    )

    distance = math.sqrt(
        (
            center_a[0] -
            center_b[0]
        ) ** 2
        +
        (
            center_a[1] -
            center_b[1]
        ) ** 2
    )

    area_a = calculate_box_area(
        box_a
    )

    area_b = calculate_box_area(
        box_b
    )

    reference_box = (
        box_a
        if area_a >= area_b
        else box_b
    )

    x1, y1, x2, y2 = (
        reference_box
    )

    width = (
        x2 -
        x1
    )

    height = (
        y2 -
        y1
    )

    diagonal = math.sqrt(
        width ** 2
        +
        height ** 2
    )

    if diagonal <= 0:
        return float("inf")

    return (
        distance /
        diagonal
    )


def are_duplicate_persons(
    person_a: Dict,
    person_b: Dict
) -> bool:
    """
    Determine whether two Person boxes likely represent
    the same visible worker.
    """

    containment = (
        calculate_containment_ratio(
            person_a["box"],
            person_b["box"]
        )
    )

    if (
        containment <
        DUPLICATE_CONTAINMENT_THRESHOLD
    ):
        return False

    center_distance = (
        calculate_center_distance_normalized(
            person_a["box"],
            person_b["box"]
        )
    )

    return (
        center_distance <=
        DUPLICATE_CENTER_DISTANCE_THRESHOLD
    )


def choose_better_person(
    person_a: Dict,
    person_b: Dict
) -> Dict:
    """
    Prefer the larger/full person box.

    Confidence is used only when the areas are nearly equal.
    """

    area_a = calculate_box_area(
        person_a["box"]
    )

    area_b = calculate_box_area(
        person_b["box"]
    )

    larger_area = max(
        area_a,
        area_b
    )

    if larger_area > 0:

        area_difference_ratio = (
            abs(
                area_a -
                area_b
            )
            /
            larger_area
        )

        if area_difference_ratio > 0.05:

            if area_a > area_b:
                return person_a

            return person_b

    confidence_a = float(
        person_a["confidence"]
    )

    confidence_b = float(
        person_b["confidence"]
    )

    if confidence_a >= confidence_b:
        return person_a

    return person_b


def remove_duplicate_persons(
    detections: List[Dict]
) -> List[Dict]:
    """
    Remove highly overlapping duplicate Person boxes.

    This operates only on the current frame.
    It does not modify BoT-SORT's internal IDs.
    """

    persons = [
        detection
        for detection in detections
        if detection["class_name"] == "Person"
    ]

    non_person_detections = [
        detection
        for detection in detections
        if detection["class_name"] != "Person"
    ]

    if len(persons) <= 1:
        return detections

    kept_persons = []

    for current_person in persons:

        duplicate_found = False

        for index, existing_person in enumerate(
            kept_persons
        ):

            if are_duplicate_persons(
                current_person,
                existing_person
            ):

                better_person = (
                    choose_better_person(
                        current_person,
                        existing_person
                    )
                )

                if (
                    better_person
                    is
                    current_person
                ):
                    removed_person = (
                        existing_person
                    )

                else:
                    removed_person = (
                        current_person
                    )

                kept_persons[index] = (
                    better_person
                )

                print(
                    "  Duplicate Person filtered: "
                    f"kept ID "
                    f"{better_person.get('track_id')} "
                    f"| removed ID "
                    f"{removed_person.get('track_id')}"
                )

                duplicate_found = True
                break

        if not duplicate_found:
            kept_persons.append(
                current_person
            )

    return (
        non_person_detections
        +
        kept_persons
    )


# ============================================================
# 12. ONE-TO-ONE PPE ASSIGNMENT
# ============================================================

def assign_ppe_group(
    persons: List[Dict],
    ppe_items: List[Dict],
    region_type: str,
    min_score: float
) -> List[Tuple[int, int, Dict]]:
    """
    One PPE object can be assigned to only one person,
    and each person can receive at most one item from
    this PPE group.
    """

    candidate_matches = []

    for ppe_index, ppe in enumerate(
        ppe_items
    ):

        for person_index, person in enumerate(
            persons
        ):

            scores = (
                calculate_association_score(
                    ppe,
                    person,
                    region_type
                )
            )

            if (
                scores["final"] >=
                min_score
            ):

                candidate_matches.append(
                    (
                        scores["final"],
                        person_index,
                        ppe_index,
                        scores
                    )
                )

    candidate_matches.sort(
        key=lambda item: item[0],
        reverse=True
    )

    assigned_persons = set()
    assigned_ppe = set()

    assignments = []

    for (
        final_score,
        person_index,
        ppe_index,
        scores
    ) in candidate_matches:

        if person_index in assigned_persons:
            continue

        if ppe_index in assigned_ppe:
            continue

        assigned_persons.add(
            person_index
        )

        assigned_ppe.add(
            ppe_index
        )

        assignments.append(
            (
                person_index,
                ppe_index,
                scores
            )
        )

    return assignments


# ============================================================
# 13. ASSOCIATE PPE TO PERSONS
# ============================================================

def associate_ppe_to_persons(
    persons: List[Dict],
    helmets: List[Dict],
    no_helmets: List[Dict],
    vests: List[Dict],
    no_vests: List[Dict],
    safety_shoes: List[Dict],
    no_safety_shoes: List[Dict],
    without_safety_goggles: List[Dict],
    with_safety_goggles: List[Dict],
    min_score: float = MIN_ASSOCIATION_SCORE
) -> List[Dict]:
    """
    Associate all PPE classes with persons.

    PPE groups:

        Helmet / No-Helmet
        Vest / No-Vest
        Safety-Shoes / No-Safety-Shoes
        With-Safety-Goggles / Without-Safety-Goggles
    """

    person_results = []

    for person_index, person in enumerate(
        persons,
        start=1
    ):

        person_results.append(
            {
                "person_index": person_index,
                "person": person,

                "helmet": None,
                "helmet_score": 0.0,
                "helmet_metrics": None,

                "no_helmet": None,
                "no_helmet_score": 0.0,
                "no_helmet_metrics": None,

                "vest": None,
                "vest_score": 0.0,
                "vest_metrics": None,

                "no_vest": None,
                "no_vest_score": 0.0,
                "no_vest_metrics": None,

                "safety_shoes": None,
                "safety_shoes_score": 0.0,
                "safety_shoes_metrics": None,

                "no_safety_shoes": None,
                "no_safety_shoes_score": 0.0,
                "no_safety_shoes_metrics": None,

                "without_safety_goggles": None,
                "without_safety_goggles_score": 0.0,
                "without_safety_goggles_metrics": None,

                "with_safety_goggles": None,
                "with_safety_goggles_score": 0.0,
                "with_safety_goggles_metrics": None
            }
        )

    # ========================================================
    # HELMET GROUP
    # ========================================================

    helmet_candidates = (
        helmets +
        no_helmets
    )

    helmet_assignments = (
        assign_ppe_group(
            persons,
            helmet_candidates,
            "head",
            min_score
        )
    )

    for (
        person_index,
        ppe_index,
        scores
    ) in helmet_assignments:

        ppe = (
            helmet_candidates[
                ppe_index
            ]
        )

        record = (
            person_results[
                person_index
            ]
        )

        if ppe["class_name"] == "Helmet":

            record["helmet"] = ppe

            record["helmet_score"] = (
                scores["final"]
            )

            record["helmet_metrics"] = (
                scores
            )

        elif (
            ppe["class_name"]
            == "No-Helmet"
        ):

            record["no_helmet"] = ppe

            record["no_helmet_score"] = (
                scores["final"]
            )

            record["no_helmet_metrics"] = (
                scores
            )

    # ========================================================
    # VEST GROUP
    # ========================================================

    vest_candidates = (
        vests +
        no_vests
    )

    vest_assignments = (
        assign_ppe_group(
            persons,
            vest_candidates,
            "torso",
            min_score
        )
    )

    for (
        person_index,
        ppe_index,
        scores
    ) in vest_assignments:

        ppe = (
            vest_candidates[
                ppe_index
            ]
        )

        record = (
            person_results[
                person_index
            ]
        )

        if ppe["class_name"] == "Vest":

            record["vest"] = ppe

            record["vest_score"] = (
                scores["final"]
            )

            record["vest_metrics"] = (
                scores
            )

        elif (
            ppe["class_name"]
            == "No-Vest"
        ):

            record["no_vest"] = ppe

            record["no_vest_score"] = (
                scores["final"]
            )

            record["no_vest_metrics"] = (
                scores
            )

    # ========================================================
    # SAFETY-SHOES GROUP
    # ========================================================

    shoe_candidates = (
        safety_shoes +
        no_safety_shoes
    )

    shoe_assignments = (
        assign_ppe_group(
            persons,
            shoe_candidates,
            "feet",
            min_score
        )
    )

    for (
        person_index,
        ppe_index,
        scores
    ) in shoe_assignments:

        ppe = (
            shoe_candidates[
                ppe_index
            ]
        )

        record = (
            person_results[
                person_index
            ]
        )

        if (
            ppe["class_name"]
            == "Safety-Shoes"
        ):

            record["safety_shoes"] = ppe

            record["safety_shoes_score"] = (
                scores["final"]
            )

            record["safety_shoes_metrics"] = (
                scores
            )

        elif (
            ppe["class_name"]
            == "No-Safety-Shoes"
        ):

            record["no_safety_shoes"] = ppe

            record["no_safety_shoes_score"] = (
                scores["final"]
            )

            record["no_safety_shoes_metrics"] = (
                scores
            )

    # ========================================================
    # SAFETY-GOGGLES GROUP
    # ========================================================

    goggles_candidates = (
        without_safety_goggles +
        with_safety_goggles
    )

    goggles_assignments = (
        assign_ppe_group(
            persons,
            goggles_candidates,
            "goggles",
            min_score
        )
    )

    for (
        person_index,
        ppe_index,
        scores
    ) in goggles_assignments:

        ppe = (
            goggles_candidates[
                ppe_index
            ]
        )

        record = (
            person_results[
                person_index
            ]
        )

        if (
            ppe["class_name"]
            == "Without-Safety-Goggles"
        ):

            record["without_safety_goggles"] = ppe

            record["without_safety_goggles_score"] = (
                scores["final"]
            )

            record["without_safety_goggles_metrics"] = (
                scores
            )

        elif (
            ppe["class_name"]
            == "With-Safety-Goggles"
        ):

            record["with_safety_goggles"] = ppe

            record["with_safety_goggles_score"] = (
                scores["final"]
            )

            record["with_safety_goggles_metrics"] = (
                scores
            )

    return person_results


# ============================================================
# 14. BUILD CLEAN PPE STATUS
# ============================================================

def build_ppe_status(
    person_results: List[Dict]
) -> List[Dict]:
    """
    Convert associations into person-level status:

        HELMET / NO-HELMET / UNKNOWN
        VEST / NO-VEST / UNKNOWN
        SAFETY-SHOES / NO-SAFETY-SHOES / UNKNOWN
        WITH-SAFETY-GOGGLES /
        WITHOUT-SAFETY-GOGGLES / UNKNOWN
    """

    status_results = []

    for record in person_results:

        # ====================================================
        # HELMET
        # ====================================================

        helmet = record["helmet"]
        no_helmet = record["no_helmet"]

        if (
            helmet is not None
            and
            no_helmet is not None
        ):

            if (
                record["helmet_score"]
                >=
                record["no_helmet_score"]
            ):

                helmet_status = "HELMET"
                helmet_score = (
                    record["helmet_score"]
                )

            else:

                helmet_status = "NO-HELMET"
                helmet_score = (
                    record["no_helmet_score"]
                )

        elif helmet is not None:

            helmet_status = "HELMET"
            helmet_score = (
                record["helmet_score"]
            )

        elif no_helmet is not None:

            helmet_status = "NO-HELMET"
            helmet_score = (
                record["no_helmet_score"]
            )

        else:

            helmet_status = "UNKNOWN"
            helmet_score = 0.0

        # ====================================================
        # VEST
        # ====================================================

        vest = record["vest"]
        no_vest = record["no_vest"]

        if (
            vest is not None
            and
            no_vest is not None
        ):

            if (
                record["vest_score"]
                >=
                record["no_vest_score"]
            ):

                vest_status = "VEST"
                vest_score = (
                    record["vest_score"]
                )

            else:

                vest_status = "NO-VEST"
                vest_score = (
                    record["no_vest_score"]
                )

        elif vest is not None:

            vest_status = "VEST"
            vest_score = (
                record["vest_score"]
            )

        elif no_vest is not None:

            vest_status = "NO-VEST"
            vest_score = (
                record["no_vest_score"]
            )

        else:

            vest_status = "UNKNOWN"
            vest_score = 0.0

        # ====================================================
        # SAFETY-SHOES
        # ====================================================

        safety_shoes = record["safety_shoes"]
        no_safety_shoes = record["no_safety_shoes"]

        if (
            safety_shoes is not None
            and
            no_safety_shoes is not None
        ):

            if (
                record["safety_shoes_score"]
                >=
                record["no_safety_shoes_score"]
            ):

                safety_shoes_status = "SAFETY-SHOES"

                safety_shoes_score = (
                    record["safety_shoes_score"]
                )

            else:

                safety_shoes_status = "NO-SAFETY-SHOES"

                safety_shoes_score = (
                    record["no_safety_shoes_score"]
                )

        elif safety_shoes is not None:

            safety_shoes_status = "SAFETY-SHOES"

            safety_shoes_score = (
                record["safety_shoes_score"]
            )

        elif no_safety_shoes is not None:

            safety_shoes_status = "NO-SAFETY-SHOES"

            safety_shoes_score = (
                record["no_safety_shoes_score"]
            )

        else:

            safety_shoes_status = "UNKNOWN"
            safety_shoes_score = 0.0

        # ====================================================
        # SAFETY-GOGGLES
        # ====================================================

        with_safety_goggles = (
            record["with_safety_goggles"]
        )

        without_safety_goggles = (
            record["without_safety_goggles"]
        )

        if (
            with_safety_goggles is not None
            and
            without_safety_goggles is not None
        ):

            if (
                record["with_safety_goggles_score"]
                >=
                record["without_safety_goggles_score"]
            ):

                safety_goggles_status = (
                    "WITH-SAFETY-GOGGLES"
                )

                safety_goggles_score = (
                    record["with_safety_goggles_score"]
                )

            else:

                safety_goggles_status = (
                    "WITHOUT-SAFETY-GOGGLES"
                )

                safety_goggles_score = (
                    record["without_safety_goggles_score"]
                )

        elif with_safety_goggles is not None:

            safety_goggles_status = (
                "WITH-SAFETY-GOGGLES"
            )

            safety_goggles_score = (
                record["with_safety_goggles_score"]
            )

        elif without_safety_goggles is not None:

            safety_goggles_status = (
                "WITHOUT-SAFETY-GOGGLES"
            )

            safety_goggles_score = (
                record["without_safety_goggles_score"]
            )

        else:

            safety_goggles_status = "UNKNOWN"
            safety_goggles_score = 0.0

        # ====================================================
        # FINAL PERSON STATUS
        # ====================================================

        status_results.append(
            {
                "person_index":
                    record["person_index"],

                "track_id":
                    record["person"].get(
                        "track_id"
                    ),

                "helmet_status":
                    helmet_status,

                "helmet_score":
                    helmet_score,

                "vest_status":
                    vest_status,

                "vest_score":
                    vest_score,

                "safety_shoes_status":
                    safety_shoes_status,

                "safety_shoes_score":
                    safety_shoes_score,

                "safety_goggles_status":
                    safety_goggles_status,

                "safety_goggles_score":
                    safety_goggles_score
            }
        )

    return status_results


# ============================================================
# 15. DRAW PPE ASSOCIATION
# ============================================================

def draw_ppe_association(
    image,
    persons: List[Dict],
    ppe_status: List[Dict],
    associations=None
):
    """
    Draw:

        Person boxes
        Track IDs
        Helmet
        Vest
        Safety-Shoes
        Safety-Goggles
        Person-level PPE status
    """

    output = image.copy()

    # ========================================================
    # PPE BOXES
    # ========================================================

    if associations is not None:

        for record in associations:

            matched_items = [
                record.get("helmet"),
                record.get("no_helmet"),

                record.get("vest"),
                record.get("no_vest"),

                record.get("safety_shoes"),
                record.get("no_safety_shoes"),

                record.get("without_safety_goggles"),
                record.get("with_safety_goggles")
            ]

            for ppe in matched_items:

                if ppe is None:
                    continue

                x1, y1, x2, y2 = [
                    int(value)
                    for value in ppe["box"]
                ]

                label = (
                    f"{ppe['class_name']} "
                    f"{ppe['confidence']:.2f}"
                )

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

                (
                    text_width,
                    text_height
                ), baseline = (
                    cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        1
                    )
                )

                label_y1 = max(
                    0,
                    y1 -
                    text_height -
                    baseline -
                    4
                )

                label_y2 = y1

                cv2.rectangle(
                    output,
                    (x1, label_y1),
                    (
                        x1 +
                        text_width +
                        6,
                        label_y2
                    ),
                    (255, 255, 255),
                    -1
                )

                cv2.putText(
                    output,
                    label,
                    (
                        x1 + 3,
                        label_y2 -
                        baseline -
                        2
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

    # ========================================================
    # PERSON BOXES
    # ========================================================

    for person, status in zip(
        persons,
        ppe_status
    ):

        x1, y1, x2, y2 = [
            int(value)
            for value in person["box"]
        ]

        track_id = (
            status.get("track_id")
        )

        if track_id is not None:

            identity = (
                f"ID: {track_id}"
            )

        else:

            identity = (
                f"Person "
                f"{status['person_index']}"
            )

        label_lines = [
            identity,

            (
                f"Helmet: "
                f"{status['helmet_status']}"
            ),

            (
                f"Vest: "
                f"{status['vest_status']}"
            ),

            (
                f"Shoes: "
                f"{status['safety_shoes_status']}"
            ),

            (
                f"Goggles: "
                f"{status['safety_goggles_status']}"
            )
        ]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        font = (
            cv2.FONT_HERSHEY_SIMPLEX
        )

        font_scale = 0.5
        thickness = 1
        line_height = 18

        max_text_width = 0

        for line in label_lines:

            (
                text_size,
                _
            ) = cv2.getTextSize(
                line,
                font,
                font_scale,
                thickness
            )

            max_text_width = max(
                max_text_width,
                text_size[0]
            )

        label_width = (
            max_text_width +
            10
        )

        label_height = (
            line_height *
            len(label_lines)
            +
            6
        )

        label_y2 = y1

        label_y1 = max(
            0,
            y1 -
            label_height
        )

        cv2.rectangle(
            output,
            (x1, label_y1),
            (
                x1 +
                label_width,
                label_y2
            ),
            (0, 255, 0),
            -1
        )

        for line_index, line in enumerate(
            label_lines
        ):

            text_y = (
                label_y1
                +
                15
                +
                line_index *
                line_height
            )

            cv2.putText(
                output,
                line,
                (x1 + 4, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

    return output 
