from typing import Dict, List, Tuple, Optional
import math

import cv2


# ============================================================
# CONFIGURATION
# ============================================================

MIN_ASSOCIATION_SCORE = 0.45


# ============================================================
# MODEL CLASSES
# ============================================================
#
# 0  Helmet
# 1  No-Helmet
# 2  No-Vest
# 3  Person
# 4  Vest
# 5  Safety-Shoes
# 6  No-Safety-Shoes
# 7  Without-Safety-Goggles
# 8  With-Safety-Goggles
#
# ============================================================


# ============================================================
# PPE ASSOCIATION CONFIGURATION
# ============================================================

PPE_CONFIG = {

    "helmet": {
        "region": "head",

        "overlap": 0.40,
        "position": 0.35,
        "distance": 0.15,
        "confidence": 0.10,

        "expected_y": 0.15,
        "tolerance": 0.25,

        "min_y": 0.00,
        "max_y": 0.42,

        "max_items": 1,
    },

    "goggles": {
        "region": "goggles",

        "overlap": 0.25,
        "position": 0.50,
        "distance": 0.15,
        "confidence": 0.10,

        "expected_y": 0.18,
        "tolerance": 0.18,

        "min_y": 0.00,
        "max_y": 0.42,

        "max_items": 1,
    },

    "vest": {
        "region": "torso",

        "overlap": 0.50,
        "position": 0.25,
        "distance": 0.15,
        "confidence": 0.10,

        "expected_y": 0.50,
        "tolerance": 0.32,

        "min_y": 0.15,
        "max_y": 0.88,

        "max_items": 1,
    },

    "shoes": {
        "region": "feet",

        "overlap": 0.30,
        "position": 0.40,
        "distance": 0.20,
        "confidence": 0.10,

        "expected_y": 0.90,
        "tolerance": 0.25,

        "min_y": 0.62,
        "max_y": 1.05,

        "max_items": 2,
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):

    try:
        return int(round(float(value)))
    except Exception:
        return default


# ============================================================
# BOX UTILITIES
# ============================================================

def get_box_center(
    box: List[float],
) -> Tuple[float, float]:

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def calculate_box_area(
    box: List[float],
) -> float:

    x1, y1, x2, y2 = box

    width = max(
        0.0,
        x2 - x1,
    )

    height = max(
        0.0,
        y2 - y1,
    )

    return width * height


def calculate_intersection_area(
    box_a: List[float],
    box_b: List[float],
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    return (
        (ix2 - ix1)
        *
        (iy2 - iy1)
    )


def calculate_iou(
    box_a: List[float],
    box_b: List[float],
) -> float:

    area_a = calculate_box_area(box_a)
    area_b = calculate_box_area(box_b)

    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    intersection = calculate_intersection_area(
        box_a,
        box_b,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return intersection / union


def calculate_containment_ratio(
    inner_box: List[float],
    outer_box: List[float],
) -> float:

    inner_area = calculate_box_area(
        inner_box
    )

    if inner_area <= 0.0:
        return 0.0

    intersection = calculate_intersection_area(
        inner_box,
        outer_box,
    )

    return intersection / inner_area


# ============================================================
# 1. EXTRACT YOLO DETECTIONS
# ============================================================

def extract_detections(
    result,
    class_names,
) -> List[Dict]:

    detections = []

    if result is None:
        return detections

    if result.boxes is None:
        return detections

    boxes = result.boxes

    if len(boxes) == 0:
        return detections

    # --------------------------------------------------------
    # Track IDs
    # --------------------------------------------------------

    track_ids = None

    if boxes.id is not None:

        try:

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

        except Exception:

            track_ids = None

    # --------------------------------------------------------
    # Class name helper
    # --------------------------------------------------------

    def get_class_name(
        class_id: int,
    ) -> str:

        if isinstance(
            class_names,
            dict,
        ):

            return str(
                class_names.get(
                    class_id,
                    str(class_id),
                )
            )

        if (
            isinstance(
                class_names,
                list,
            )
            and
            0 <= class_id < len(class_names)
        ):

            return str(
                class_names[class_id]
            )

        return str(class_id)

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    for index in range(
        len(boxes)
    ):

        class_id = int(
            boxes.cls[index].item()
        )

        confidence = float(
            boxes.conf[index].item()
        )

        x1, y1, x2, y2 = (
            boxes.xyxy[index]
            .cpu()
            .tolist()
        )

        track_id = None

        if (
            track_ids is not None
            and
            index < len(track_ids)
        ):

            try:

                track_id = int(
                    track_ids[index]
                )

            except Exception:

                track_id = None

        detections.append(
            {
                "class_id": class_id,

                "class_name": get_class_name(
                    class_id
                ),

                "confidence": confidence,

                "box": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ],

                "track_id": track_id,
            }
        )

    return detections


# ============================================================
# 2. SPLIT 9 CLASSES
# ============================================================

def split_detections(
    detections: List[Dict],
):

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

        class_name = detection.get(
            "class_name",
            "",
        )

        if class_name == "Person":

            persons.append(
                detection
            )

        elif class_name == "Helmet":

            helmets.append(
                detection
            )

        elif class_name == "No-Helmet":

            no_helmets.append(
                detection
            )

        elif class_name == "Vest":

            vests.append(
                detection
            )

        elif class_name == "No-Vest":

            no_vests.append(
                detection
            )

        elif class_name == "Safety-Shoes":

            safety_shoes.append(
                detection
            )

        elif class_name == "No-Safety-Shoes":

            no_safety_shoes.append(
                detection
            )

        elif class_name == "Without-Safety-Goggles":

            without_safety_goggles.append(
                detection
            )

        elif class_name == "With-Safety-Goggles":

            with_safety_goggles.append(
                detection
            )

    return (
        persons,
        helmets,
        no_helmets,
        vests,
        no_vests,
        safety_shoes,
        no_safety_shoes,
        without_safety_goggles,
        with_safety_goggles,
    )


# ============================================================
# 3. PERSON REGIONS
# ============================================================

def get_person_regions(
    person_box: List[float],
):

    x1, y1, x2, y2 = person_box

    width = max(
        1.0,
        x2 - x1,
    )

    height = max(
        1.0,
        y2 - y1,
    )

    return {

        # Head
        "head": [
            x1,
            y1,
            x2,
            y1 + 0.38 * height,
        ],

        # Face / eyes
        "goggles": [
            x1,
            y1 + 0.05 * height,
            x2,
            y1 + 0.32 * height,
        ],

        # Torso
        "torso": [
            x1,
            y1 + 0.22 * height,
            x2,
            y1 + 0.78 * height,
        ],

        # Legs / feet
        "feet": [
            x1,
            y1 + 0.62 * height,
            x2,
            y2,
        ],
    }


# ============================================================
# 4. NORMALIZED CENTER DISTANCE
# ============================================================

def calculate_center_distance_normalized(
    ppe_box: List[float],
    person_box: List[float],
) -> float:

    ppe_cx, ppe_cy = get_box_center(
        ppe_box
    )

    person_cx, person_cy = get_box_center(
        person_box
    )

    person_width = max(
        1.0,
        person_box[2] - person_box[0],
    )

    person_height = max(
        1.0,
        person_box[3] - person_box[1],
    )

    dx = (
        ppe_cx - person_cx
    ) / person_width

    dy = (
        ppe_cy - person_cy
    ) / person_height

    return math.sqrt(
        dx * dx + dy * dy
    )


def calculate_center_distance_score(
    ppe_box: List[float],
    person_box: List[float],
) -> float:

    distance = (
        calculate_center_distance_normalized(
            ppe_box,
            person_box,
        )
    )

    return max(
        0.0,
        min(
            1.0,
            1.0 - distance,
        ),
    )


# ============================================================
# 5. POSITION SCORE
# ============================================================

def calculate_position_score(
    ppe_box: List[float],
    person_box: List[float],
    region_type: str,
) -> float:

    config = PPE_CONFIG[
        region_type
    ]

    _, ppe_center_y = get_box_center(
        ppe_box
    )

    person_y1 = person_box[1]

    person_height = max(
        1.0,
        person_box[3] - person_box[1],
    )

    relative_y = (
        ppe_center_y - person_y1
    ) / person_height

    expected_y = config[
        "expected_y"
    ]

    tolerance = config[
        "tolerance"
    ]

    difference = abs(
        relative_y - expected_y
    )

    if difference >= tolerance:
        return 0.0

    return max(
        0.0,
        1.0 - (
            difference / tolerance
        ),
    )


# ============================================================
# 6. OVERLAP SCORE
# ============================================================

def calculate_overlap_ratio(
    ppe_box: List[float],
    region_box: List[float],
) -> float:

    ppe_area = calculate_box_area(
        ppe_box
    )

    if ppe_area <= 0.0:
        return 0.0

    intersection = (
        calculate_intersection_area(
            ppe_box,
            region_box,
        )
    )

    return max(
        0.0,
        min(
            1.0,
            intersection / ppe_area,
        ),
    )


# ============================================================
# 7. SPATIAL VALIDATION
# ============================================================

def passes_spatial_validation(
    ppe_box: List[float],
    person_box: List[float],
    region_type: str,
) -> bool:

    config = PPE_CONFIG[
        region_type
    ]

    _, ppe_center_y = get_box_center(
        ppe_box
    )

    person_y1 = person_box[1]

    person_height = max(
        1.0,
        person_box[3] - person_box[1],
    )

    relative_y = (
        ppe_center_y - person_y1
    ) / person_height

    return (
        config["min_y"]
        <= relative_y
        <= config["max_y"]
    )


# ============================================================
# 8. ASSOCIATION SCORE
# ============================================================

def calculate_association_score(
    ppe: Dict,
    person: Dict,
    region_type: str,
) -> Dict:

    config = PPE_CONFIG[
        region_type
    ]

    ppe_box = ppe["box"]
    person_box = person["box"]

    regions = get_person_regions(
        person_box
    )

    region = regions[
        config["region"]
    ]

    overlap = calculate_overlap_ratio(
        ppe_box,
        region,
    )

    position = calculate_position_score(
        ppe_box,
        person_box,
        region_type,
    )

    distance = calculate_center_distance_score(
        ppe_box,
        person_box,
    )

    confidence = safe_float(
        ppe.get(
            "confidence",
            0.0,
        )
    )

    spatial_valid = (
        passes_spatial_validation(
            ppe_box,
            person_box,
            region_type,
        )
    )

    final_score = (

        config["overlap"]
        * overlap

        +

        config["position"]
        * position

        +

        config["distance"]
        * distance

        +

        config["confidence"]
        * confidence
    )

    # Strongly penalize impossible spatial
    # associations.
    if not spatial_valid:

        final_score *= 0.10

    return {

        "final": max(
            0.0,
            min(
                1.0,
                float(final_score),
            ),
        ),

        "overlap": float(
            overlap
        ),

        "position": float(
            position
        ),

        "distance": float(
            distance
        ),

        "confidence": float(
            confidence
        ),

        "spatial_valid": bool(
            spatial_valid
        ),
    }


# ============================================================
# 9. DUPLICATE PERSON REMOVAL
# ============================================================

DUPLICATE_IOU_THRESHOLD = 0.80

DUPLICATE_CONTAINMENT_THRESHOLD = 0.90


def are_duplicate_persons(
    person_a: Dict,
    person_b: Dict,
) -> bool:

    box_a = person_a["box"]
    box_b = person_b["box"]

    iou = calculate_iou(
        box_a,
        box_b,
    )

    if iou >= DUPLICATE_IOU_THRESHOLD:
        return True

    containment_a = (
        calculate_containment_ratio(
            box_a,
            box_b,
        )
    )

    containment_b = (
        calculate_containment_ratio(
            box_b,
            box_a,
        )
    )

    return (
        containment_a
        >= DUPLICATE_CONTAINMENT_THRESHOLD
        or
        containment_b
        >= DUPLICATE_CONTAINMENT_THRESHOLD
    )


def choose_better_person(
    person_a: Dict,
    person_b: Dict,
) -> Dict:

    track_a = person_a.get(
        "track_id"
    )

    track_b = person_b.get(
        "track_id"
    )

    # Prefer a tracked detection.
    if (
        track_a is not None
        and track_b is None
    ):

        return person_a

    if (
        track_b is not None
        and track_a is None
    ):

        return person_b

    confidence_a = safe_float(
        person_a.get(
            "confidence",
            0.0,
        )
    )

    confidence_b = safe_float(
        person_b.get(
            "confidence",
            0.0,
        )
    )

    if confidence_a >= confidence_b:
        return person_a

    return person_b


def remove_duplicate_persons(
    detections: List[Dict],
) -> List[Dict]:

    persons = [
        d
        for d in detections
        if d.get("class_name")
        == "Person"
    ]

    others = [
        d
        for d in detections
        if d.get("class_name")
        != "Person"
    ]

    persons.sort(
        key=lambda d: safe_float(
            d.get(
                "confidence",
                0.0,
            )
        ),
        reverse=True,
    )

    kept = []

    for current in persons:

        duplicate = False

        for index, existing in enumerate(
            kept
        ):

            if are_duplicate_persons(
                current,
                existing,
            ):

                kept[index] = (
                    choose_better_person(
                        current,
                        existing,
                    )
                )

                duplicate = True
                break

        if not duplicate:
            kept.append(
                current
            )

    return others + kept


# ============================================================
# 10. GLOBAL PPE ASSIGNMENT
# ============================================================

def assign_ppe_group(
    persons: List[Dict],
    ppe_items: List[Dict],
    region_type: str,
    min_score: float,
    max_items_per_person: int = 1,
):

    if not persons:
        return []

    if not ppe_items:
        return []

    candidates = []

    # --------------------------------------------------------
    # Generate all possible associations.
    # --------------------------------------------------------

    for ppe_index, ppe in enumerate(
        ppe_items
    ):

        for person_index, person in enumerate(
            persons
        ):

            metrics = (
                calculate_association_score(
                    ppe,
                    person,
                    region_type,
                )
            )

            if (
                metrics["final"]
                >= min_score
            ):

                candidates.append(
                    (
                        metrics["final"],
                        person_index,
                        ppe_index,
                        metrics,
                    )
                )

    # Highest score first.
    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    assigned_ppe = set()

    person_counts = {}

    assignments = []

    for (
        score,
        person_index,
        ppe_index,
        metrics,
    ) in candidates:

        # One PPE detection cannot belong
        # to multiple people.
        if ppe_index in assigned_ppe:
            continue

        count = person_counts.get(
            person_index,
            0,
        )

        if count >= max_items_per_person:
            continue

        assigned_ppe.add(
            ppe_index
        )

        person_counts[
            person_index
        ] = count + 1

        assignments.append(
            (
                person_index,
                ppe_index,
                metrics,
            )
        )

    return assignments


# ============================================================
# 11. APPLY SINGLE PPE ASSIGNMENT
# ============================================================

def apply_single_assignments(
    records,
    ppe_items,
    assignments,
    object_key,
    score_key,
    metrics_key,
):

    for (
        person_index,
        ppe_index,
        metrics,
    ) in assignments:

        if not (
            0
            <= person_index
            < len(records)
        ):
            continue

        if not (
            0
            <= ppe_index
            < len(ppe_items)
        ):
            continue

        records[
            person_index
        ][object_key] = (
            ppe_items[
                ppe_index
            ]
        )

        records[
            person_index
        ][score_key] = float(
            metrics["final"]
        )

        records[
            person_index
        ][metrics_key] = metrics


# ============================================================
# 12. APPLY MULTIPLE PPE ASSIGNMENTS
# ============================================================

def apply_multiple_assignments(
    records,
    ppe_items,
    assignments,
    object_key,
    score_key,
    metrics_key,
):

    for (
        person_index,
        ppe_index,
        metrics,
    ) in assignments:

        if not (
            0
            <= person_index
            < len(records)
        ):
            continue

        if not (
            0
            <= ppe_index
            < len(ppe_items)
        ):
            continue

        records[
            person_index
        ][object_key].append(
            ppe_items[
                ppe_index
            ]
        )

        current_score = safe_float(
            records[
                person_index
            ].get(
                score_key,
                0.0,
            )
        )

        records[
            person_index
        ][score_key] = max(
            current_score,
            safe_float(
                metrics["final"]
            ),
        )

        records[
            person_index
        ].setdefault(
            metrics_key,
            [],
        )

        records[
            person_index
        ][metrics_key].append(
            metrics
        )


# ============================================================
# 13. ASSOCIATE ALL PPE
# ============================================================

def associate_ppe_to_persons(
    persons: List[Dict],
    helmets: List[Dict],
    no_helmets: List[Dict],
    vests: List[Dict],
    no_vests: List[Dict],
    safety_shoes: Optional[List[Dict]] = None,
    no_safety_shoes: Optional[List[Dict]] = None,
    without_safety_goggles: Optional[List[Dict]] = None,
    with_safety_goggles: Optional[List[Dict]] = None,
    min_score: float = MIN_ASSOCIATION_SCORE,
):

    safety_shoes = (
        safety_shoes or []
    )

    no_safety_shoes = (
        no_safety_shoes or []
    )

    without_safety_goggles = (
        without_safety_goggles or []
    )

    with_safety_goggles = (
        with_safety_goggles or []
    )

    records = []

    # --------------------------------------------------------
    # Create person records.
    # --------------------------------------------------------

    for person_index, person in enumerate(
        persons,
        start=1,
    ):

        records.append(
            {

                "person_index":
                    person_index,

                "person":
                    person,

                # Helmet
                "helmet": None,
                "helmet_score": 0.0,
                "helmet_metrics": None,

                "no_helmet": None,
                "no_helmet_score": 0.0,
                "no_helmet_metrics": None,

                # Vest
                "vest": None,
                "vest_score": 0.0,
                "vest_metrics": None,

                "no_vest": None,
                "no_vest_score": 0.0,
                "no_vest_metrics": None,

                # Shoes
                "safety_shoes": [],
                "safety_shoes_score": 0.0,
                "safety_shoes_metrics": [],

                "no_safety_shoes": [],
                "no_safety_shoes_score": 0.0,
                "no_safety_shoes_metrics": [],

                # Goggles
                "with_safety_goggles": None,
                "with_safety_goggles_score": 0.0,
                "with_safety_goggles_metrics": None,

                "without_safety_goggles": None,
                "without_safety_goggles_score": 0.0,
                "without_safety_goggles_metrics": None,
            }
        )

    if not records:
        return records

    # ========================================================
    # HELMET
    # ========================================================

    assignments = assign_ppe_group(
        persons,
        helmets,
        "helmet",
        min_score,
        1,
    )

    apply_single_assignments(
        records,
        helmets,
        assignments,
        "helmet",
        "helmet_score",
        "helmet_metrics",
    )

    # ========================================================
    # NO HELMET
    # ========================================================

    assignments = assign_ppe_group(
        persons,
        no_helmets,
        "helmet",
        min_score,
        1,
    )

    apply_single_assignments(
        records,
        no_helmets,
        assignments,
        "no_helmet",
        "no_helmet_score",
        "no_helmet_metrics",
    )

    # ========================================================
    # VEST
    # ========================================================

    assignments = assign_ppe_group(
        persons,
        vests,
        "vest",
        min_score,
        1,
    )

    apply_single_assignments(
        records,
        vests,
        assignments,
        "vest",
        "vest_score",
        "vest_metrics",
    )

    # ========================================================
    # NO VEST
    # ========================================================

    assignments = assign_ppe_group(
        persons,
        no_vests,
        "vest",
        min_score,
        1,
    )

    apply_single_assignments(
        records,
        no_vests,
        assignments,
        "no_vest",
        "no_vest_score",
        "no_vest_metrics",
    )

    # ========================================================
    # SAFETY SHOES
    # ========================================================

    assignments = assign_ppe_group(
        persons,
        safety_shoes,
        "shoes",
        min_score,
        2,
    )

    apply_multiple_assignments(
        records,
        safety_shoes,
        assignments,
        "safety_shoes",
        "safety_shoes_score",
        "safety_shoes_metrics",
    )

    # ========================================================
    # NO SAFETY SHOES
    # ========================================================

    assignments = assign_ppe_group(
        persons,
        no_safety_shoes,
        "shoes",
        min_score,
        2,
    )

    apply_multiple_assignments(
        records,
        no_safety_shoes,
        assignments,
        "no_safety_shoes",
        "no_safety_shoes_score",
        "no_safety_shoes_metrics",
    )

    # ========================================================
    # WITH GOGGLES
    # ========================================================

    assignments = assign_ppe_group(
        persons,
        with_safety_goggles,
        "goggles",
        min_score,
        1,
    )

    apply_single_assignments(
        records,
        with_safety_goggles,
        assignments,
        "with_safety_goggles",
        "with_safety_goggles_score",
        "with_safety_goggles_metrics",
    )

    # ========================================================
    # WITHOUT GOGGLES
    # ========================================================

    assignments = assign_ppe_group(
        persons,
        without_safety_goggles,
        "goggles",
        min_score,
        1,
    )

    apply_single_assignments(
        records,
        without_safety_goggles,
        assignments,
        "without_safety_goggles",
        "without_safety_goggles_score",
        "without_safety_goggles_metrics",
    )

    return records


# ============================================================
# 14. BINARY CONFLICT RESOLUTION
# ============================================================

def resolve_binary_conflict(
    record: Dict,
    positive_key: str,
    positive_score_key: str,
    negative_key: str,
    negative_score_key: str,
):

    positive = record.get(
        positive_key
    )

    negative = record.get(
        negative_key
    )

    positive_score = safe_float(
        record.get(
            positive_score_key,
            0.0,
        )
    )

    negative_score = safe_float(
        record.get(
            negative_score_key,
            0.0,
        )
    )

    if (
        positive is not None
        and
        negative is not None
    ):

        if positive_score >= negative_score:

            record[
                negative_key
            ] = None

            record[
                negative_score_key
            ] = 0.0

        else:

            record[
                positive_key
            ] = None

            record[
                positive_score_key
            ] = 0.0


# ============================================================
# 15. BUILD FINAL PPE STATUS
# ============================================================

def build_ppe_status(
    associations: List[Dict],
) -> List[Dict]:

    results = []

    for record in associations:

        # ----------------------------------------------------
        # Resolve helmet conflict
        # ----------------------------------------------------

        resolve_binary_conflict(
            record,
            "helmet",
            "helmet_score",
            "no_helmet",
            "no_helmet_score",
        )

        # ----------------------------------------------------
        # Resolve vest conflict
        # ----------------------------------------------------

        resolve_binary_conflict(
            record,
            "vest",
            "vest_score",
            "no_vest",
            "no_vest_score",
        )

        # ----------------------------------------------------
        # Resolve goggles conflict
        # ----------------------------------------------------

        resolve_binary_conflict(
            record,
            "with_safety_goggles",
            "with_safety_goggles_score",
            "without_safety_goggles",
            "without_safety_goggles_score",
        )

        # ====================================================
        # HELMET
        # ====================================================

        if record.get(
            "helmet"
        ) is not None:

            helmet_status = "HELMET"

            helmet_score = safe_float(
                record.get(
                    "helmet_score",
                    0.0,
                )
            )

        elif record.get(
            "no_helmet"
        ) is not None:

            helmet_status = "NO-HELMET"

            helmet_score = safe_float(
                record.get(
                    "no_helmet_score",
                    0.0,
                )
            )

        else:

            helmet_status = "UNKNOWN"
            helmet_score = 0.0

        # ====================================================
        # VEST
        # ====================================================

        if record.get(
            "vest"
        ) is not None:

            vest_status = "VEST"

            vest_score = safe_float(
                record.get(
                    "vest_score",
                    0.0,
                )
            )

        elif record.get(
            "no_vest"
        ) is not None:

            vest_status = "NO-VEST"

            vest_score = safe_float(
                record.get(
                    "no_vest_score",
                    0.0,
                )
            )

        else:

            vest_status = "UNKNOWN"
            vest_score = 0.0

        # ====================================================
        # SAFETY SHOES
        # ====================================================

        safety_shoes = record.get(
            "safety_shoes",
            [],
        )

        no_safety_shoes = record.get(
            "no_safety_shoes",
            [],
        )

        safety_shoes_score = safe_float(
            record.get(
                "safety_shoes_score",
                0.0,
            )
        )

        no_safety_shoes_score = safe_float(
            record.get(
                "no_safety_shoes_score",
                0.0,
            )
        )

        # ----------------------------------------------------
        # If both sides are detected, use the stronger side.
        # ----------------------------------------------------

        if (
            safety_shoes
            and
            no_safety_shoes
        ):

            if (
                safety_shoes_score
                >=
                no_safety_shoes_score
            ):

                safety_shoes = (
                    safety_shoes
                )

                no_safety_shoes = []

                record[
                    "no_safety_shoes"
                ] = []

                no_safety_shoes_score = 0.0

            else:

                no_safety_shoes = (
                    no_safety_shoes
                )

                safety_shoes = []

                record[
                    "safety_shoes"
                ] = []

                safety_shoes_score = 0.0

        if safety_shoes:

            shoes_status = (
                "SAFETY-SHOES"
            )

            shoes_score = (
                safety_shoes_score
            )

        elif no_safety_shoes:

            shoes_status = (
                "NO-SAFETY-SHOES"
            )

            shoes_score = (
                no_safety_shoes_score
            )

        else:

            shoes_status = "UNKNOWN"
            shoes_score = 0.0

        # ====================================================
        # SAFETY GOGGLES
        # ====================================================

        with_goggles = record.get(
            "with_safety_goggles"
        )

        without_goggles = record.get(
            "without_safety_goggles"
        )

        with_goggles_score = safe_float(
            record.get(
                "with_safety_goggles_score",
                0.0,
            )
        )

        without_goggles_score = safe_float(
            record.get(
                "without_safety_goggles_score",
                0.0,
            )
        )

        if with_goggles is not None:

            goggles_status = (
                "WITH-GOGGLES"
            )

            goggles_score = (
                with_goggles_score
            )

        elif without_goggles is not None:

            goggles_status = (
                "WITHOUT-GOGGLES"
            )

            goggles_score = (
                without_goggles_score
            )

        else:

            goggles_status = "UNKNOWN"
            goggles_score = 0.0

        # ====================================================
        # OVERALL STATUS
        # ====================================================

        statuses = [
            helmet_status,
            vest_status,
            shoes_status,
            goggles_status,
        ]

        violations = [
            helmet_status == "NO-HELMET",
            vest_status == "NO-VEST",
            shoes_status == "NO-SAFETY-SHOES",
            goggles_status == "WITHOUT-GOGGLES",
        ]

        compliant = [
            helmet_status == "HELMET",
            vest_status == "VEST",
            shoes_status == "SAFETY-SHOES",
            goggles_status == "WITH-GOGGLES",
        ]

        known_count = sum(
            status != "UNKNOWN"
            for status in statuses
        )

        violation_count = sum(
            violations
        )

        compliant_count = sum(
            compliant
        )

        if known_count == 0:

            overall_status = "UNKNOWN"

        elif violation_count > 0:

            overall_status = (
                "NON-COMPLIANT"
            )

        elif compliant_count == 4:

            overall_status = "COMPLIANT"

        else:

            overall_status = "PARTIAL"

        # ====================================================
        # FINAL RESULT
        # ====================================================

        track_id = (
            record
            .get("person", {})
            .get("track_id")
        )

        results.append(
            {

                "person_index":
                    record[
                        "person_index"
                    ],

                "track_id":
                    track_id,

                # Helmet
                "helmet_status":
                    helmet_status,

                "helmet_score":
                    helmet_score,

                # Vest
                "vest_status":
                    vest_status,

                "vest_score":
                    vest_score,

                # ------------------------------------------------
                # IMPORTANT:
                # Both naming conventions are returned.
                # ------------------------------------------------

                # Shoes
                "safety_shoes_status":
                    shoes_status,

                "safety_shoes_score":
                    shoes_score,

                "shoes_status":
                    shoes_status,

                "shoes_score":
                    shoes_score,

                # Goggles
                "safety_goggles_status":
                    goggles_status,

                "safety_goggles_score":
                    goggles_score,

                "goggles_status":
                    goggles_status,

                "goggles_score":
                    goggles_score,

                # Overall
                "overall_status":
                    overall_status,
            }
        )

    return results


# ============================================================
# 16. DRAW PERSON BOX
# ============================================================

def _draw_person_box(
    image,
    box,
):

    if image is None:
        return

    if box is None:
        return

    if len(box) != 4:
        return

    x1, y1, x2, y2 = [
        safe_int(v)
        for v in box
    ]

    h, w = image.shape[:2]

    x1 = max(
        0,
        min(x1, w - 1),
    )

    x2 = max(
        0,
        min(x2, w - 1),
    )

    y1 = max(
        0,
        min(y1, h - 1),
    )

    y2 = max(
        0,
        min(y2, h - 1),
    )

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )


# ============================================================
# 17. DRAW PPE BOXES
# ============================================================

def _draw_associated_ppe(
    image,
    associations,
):

    if associations is None:
        return

    for record in associations:

        items = []

        # Single objects
        items.extend(
            [
                record.get("helmet"),
                record.get("no_helmet"),

                record.get("vest"),
                record.get("no_vest"),

                record.get(
                    "with_safety_goggles"
                ),

                record.get(
                    "without_safety_goggles"
                ),
            ]
        )

        # Shoes
        items.extend(
            record.get(
                "safety_shoes",
                [],
            )
        )

        items.extend(
            record.get(
                "no_safety_shoes",
                [],
            )
        )

        for ppe in items:

            if ppe is None:
                continue

            box = ppe.get(
                "box"
            )

            if box is None:
                continue

            x1, y1, x2, y2 = [
                safe_int(v)
                for v in box
            ]

            h, w = image.shape[:2]

            x1 = max(
                0,
                min(x1, w - 1),
            )

            x2 = max(
                0,
                min(x2, w - 1),
            )

            y1 = max(
                0,
                min(y1, h - 1),
            )

            y2 = max(
                0,
                min(y2, h - 1),
            )

            if x2 <= x1 or y2 <= y1:
                continue

            # Only the PPE box is drawn.
            #
            # We intentionally DO NOT draw a label
            # such as:
            #
            # Safety-Shoes 0.58
            #
            # because that creates visual clutter.
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                1,
            )


# ============================================================
# 18. STATUS PANEL TEXT
# ============================================================

def _status_text(
    status: Dict,
):

    track_id = status.get(
        "track_id"
    )

    if track_id is None:

        identity = (
            f"Person {status.get('person_index', '-')}"
        )

    else:

        identity = (
            f"ID: {track_id}"
        )

    return [

        identity,

        (
            "Helmet: "
            f"{status.get('helmet_status', 'UNKNOWN')}"
        ),

        (
            "Vest: "
            f"{status.get('vest_status', 'UNKNOWN')}"
        ),

        (
            "Shoes: "
            f"{status.get('shoes_status', 'UNKNOWN')}"
        ),

        (
            "Goggles: "
            f"{status.get('goggles_status', 'UNKNOWN')}"
        ),

        (
            "Overall: "
            f"{status.get('overall_status', 'UNKNOWN')}"
        ),
    ]


# ============================================================
# 19. PANEL RECTANGLE
# ============================================================

def _panel_rect(
    x,
    y,
    width,
    height,
):

    return [
        x,
        y,
        x + width,
        y + height,
    ]


def _rectangles_overlap(
    rect_a,
    rect_b,
    margin=5,
):

    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b

    ax1 -= margin
    ay1 -= margin
    ax2 += margin
    ay2 += margin

    bx1 -= margin
    by1 -= margin
    bx2 += margin
    by2 += margin

    return not (
        ax2 <= bx1
        or
        bx2 <= ax1
        or
        ay2 <= by1
        or
        by2 <= ay1
    )


# ============================================================
# 20. FIND NON-OVERLAPPING PANEL POSITION
# ============================================================

def _find_panel_position(
    image_shape,
    person_box,
    panel_width,
    panel_height,
    existing_panels,
):

    image_height, image_width = (
        image_shape[:2]
    )

    x1, y1, x2, y2 = [
        safe_int(v)
        for v in person_box
    ]

    margin = 10

    # --------------------------------------------------------
    # Candidate locations.
    #
    # We try:
    #
    # 1. right
    # 2. left
    # 3. below
    # 4. above
    #
    # For each candidate we check whether it overlaps another
    # status panel.
    # --------------------------------------------------------

    candidates = [

        (
            x2 + margin,
            y1,
        ),

        (
            x1 - panel_width - margin,
            y1,
        ),

        (
            x1,
            y2 + margin,
        ),

        (
            x1,
            y1 - panel_height - margin,
        ),
    ]

    # --------------------------------------------------------
    # First pass: fully inside image + no panel collision.
    # --------------------------------------------------------

    for px, py in candidates:

        rect = _panel_rect(
            px,
            py,
            panel_width,
            panel_height,
        )

        rx1, ry1, rx2, ry2 = rect

        if (
            rx1 < 5
            or ry1 < 5
            or rx2 > image_width - 5
            or ry2 > image_height - 5
        ):
            continue

        collision = False

        for existing in existing_panels:

            if _rectangles_overlap(
                rect,
                existing,
                margin=5,
            ):

                collision = True
                break

        if not collision:

            return (
                px,
                py,
            )

    # --------------------------------------------------------
    # Second pass:
    # Find a position with minimum collision.
    # --------------------------------------------------------

    best_position = None
    best_collision_count = 10**9

    for px, py in candidates:

        px = max(
            5,
            min(
                px,
                image_width
                - panel_width
                - 5,
            ),
        )

        py = max(
            5,
            min(
                py,
                image_height
                - panel_height
                - 5,
            ),
        )

        rect = _panel_rect(
            px,
            py,
            panel_width,
            panel_height,
        )

        collision_count = 0

        for existing in existing_panels:

            if _rectangles_overlap(
                rect,
                existing,
                margin=3,
            ):

                collision_count += 1

        if (
            collision_count
            <
            best_collision_count
        ):

            best_collision_count = (
                collision_count
            )

            best_position = (
                px,
                py,
            )

    if best_position is not None:
        return best_position

    # --------------------------------------------------------
    # Absolute fallback.
    # --------------------------------------------------------

    return (
        max(
            5,
            min(
                x1,
                image_width
                - panel_width
                - 5,
            ),
        ),

        max(
            5,
            min(
                y1,
                image_height
                - panel_height
                - 5,
            ),
        ),
    )


# ============================================================
# 21. DRAW STATUS PANEL
# ============================================================

def _draw_status_panel(
    image,
    status,
    person_box,
    existing_panels,
):

    lines = _status_text(
        status
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.42
    thickness = 1

    line_height = 17

    padding_x = 8
    padding_y = 7

    widths = []

    for line in lines:

        (
            size,
            _,
        ) = cv2.getTextSize(
            line,
            font,
            scale,
            thickness,
        )

        widths.append(
            size[0]
        )

    panel_width = (
        max(widths)
        + 2 * padding_x
    )

    panel_height = (
        len(lines)
        * line_height
        +
        2 * padding_y
    )

    image_height, image_width = (
        image.shape[:2]
    )

    # Prevent oversized panels.
    panel_width = min(
        panel_width,
        image_width - 10,
    )

    panel_height = min(
        panel_height,
        image_height - 10,
    )

    panel_x, panel_y = (
        _find_panel_position(
            image.shape,
            person_box,
            panel_width,
            panel_height,
            existing_panels,
        )
    )

    panel_rect = _panel_rect(
        panel_x,
        panel_y,
        panel_width,
        panel_height,
    )

    # --------------------------------------------------------
    # Semi-transparent background.
    # --------------------------------------------------------

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (
            panel_x,
            panel_y,
        ),
        (
            panel_x + panel_width,
            panel_y + panel_height,
        ),
        (0, 0, 0),
        -1,
    )

    image[:] = cv2.addWeighted(
        overlay,
        0.70,
        image,
        0.30,
        0,
    )

    # Border
    cv2.rectangle(
        image,
        (
            panel_x,
            panel_y,
        ),
        (
            panel_x + panel_width,
            panel_y + panel_height,
        ),
        (255, 255, 255),
        1,
    )

    # --------------------------------------------------------
    # Text.
    # --------------------------------------------------------

    for index, line in enumerate(
        lines
    ):

        # Prevent unexpectedly long strings.
        if len(line) > 60:

            line = (
                line[:57]
                + "..."
            )

        cv2.putText(
            image,
            line,
            (
                panel_x + padding_x,
                panel_y
                + padding_y
                + 13
                + index
                * line_height,
            ),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Leader line.
    # --------------------------------------------------------

    x1, y1, x2, y2 = [
        safe_int(v)
        for v in person_box
    ]

    person_center = (
        int(
            (x1 + x2) / 2
        ),
        int(
            (y1 + y2) / 2
        ),
    )

    panel_center = (
        int(
            panel_x
            + panel_width / 2
        ),
        int(
            panel_y
            + panel_height / 2
        ),
    )

    cv2.line(
        image,
        person_center,
        panel_center,
        (255, 255, 255),
        1,
    )

    cv2.circle(
        image,
        person_center,
        3,
        (255, 255, 255),
        -1,
    )

    return panel_rect


# ============================================================
# 22. DRAW PPE ASSOCIATION
# ============================================================

def draw_ppe_association(
    image,
    persons: List[Dict],
    ppe_status: List[Dict],
    associations=None,
):

    if image is None:
        return image

    output = image.copy()

    if not persons:
        return output

    if not ppe_status:
        return output

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Draw PPE boxes first.
    # Draw persons/panels afterwards.
    #
    # This prevents panels from being hidden by later
    # person/PPE drawing.
    # --------------------------------------------------------

    _draw_associated_ppe(
        output,
        associations,
    )

    # --------------------------------------------------------
    # Match only the available records.
    # --------------------------------------------------------

    count = min(
        len(persons),
        len(ppe_status),
    )

    existing_panels = []

    # --------------------------------------------------------
    # Draw persons.
    # --------------------------------------------------------

    for index in range(
        count
    ):

        person = persons[index]

        box = person.get(
            "box"
        )

        if box is None:
            continue

        _draw_person_box(
            output,
            box,
        )

    # --------------------------------------------------------
    # Draw status panels.
    #
    # They are placed sequentially while remembering previous
    # panel positions, which prevents the major overlap issue
    # seen in the screenshot.
    # --------------------------------------------------------

    for index in range(
        count
    ):

        person = persons[index]

        status = ppe_status[index]

        box = person.get(
            "box"
        )

        if box is None:
            continue

        panel_rect = (
            _draw_status_panel(
                output,
                status,
                box,
                existing_panels,
            )
        )

        existing_panels.append(
            panel_rect
        )

    return output
