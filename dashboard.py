import sqlite3
from pathlib import Path

import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = Path(
    "database/ppe_detection.db"
)

OUTPUT_IMAGE_DIR = Path(
    "outputs/images"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="wide"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    return sqlite3.connect(
        DB_PATH
    )


# ============================================================
# DATABASE SUMMARY
# ============================================================

def get_summary():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*) AS total,

            SUM(
                CASE
                    WHEN overall_status = 'COMPLIANT'
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN overall_status = 'NON-COMPLIANT'
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN overall_status = 'PARTIAL'
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN overall_status = 'UNKNOWN'
                    THEN 1
                    ELSE 0
                END
            )

        FROM ppe_detections
        """
    )

    row = cursor.fetchone()

    connection.close()

    return {
        "total": row[0] or 0,
        "compliant": row[1] or 0,
        "non_compliant": row[2] or 0,
        "partial": row[3] or 0,
        "unknown": row[4] or 0,
    }


# ============================================================
# PPE VIOLATIONS
# ============================================================

def get_violations():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT

            SUM(
                CASE
                    WHEN helmet_status = 'NO-HELMET'
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN vest_status = 'NO-VEST'
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN safety_shoes_status =
                         'NO-SAFETY-SHOES'
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN safety_goggles_status =
                         'WITHOUT-GOGGLES'
                    THEN 1
                    ELSE 0
                END
            )

        FROM ppe_detections
        """
    )

    row = cursor.fetchone()

    connection.close()

    return {
        "helmet": row[0] or 0,
        "vest": row[1] or 0,
        "shoes": row[2] or 0,
        "goggles": row[3] or 0,
    }


# ============================================================
# RECENT RECORDS
# ============================================================

def get_recent_records(
    limit=20
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            timestamp,
            source,
            person_index,
            helmet_status,
            vest_status,
            safety_shoes_status,
            safety_goggles_status,
            overall_status

        FROM ppe_detections

        ORDER BY id DESC

        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


# ============================================================
# HEADER
# ============================================================

st.title(
    "🦺 PPE Detection System"
)

st.caption(
    "YOLO11-based Personal Protective Equipment Monitoring"
)

st.divider()


# ============================================================
# CHECK DATABASE
# ============================================================

if not DB_PATH.exists():

    st.error(
        "Database not found: "
        f"{DB_PATH}"
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

summary = get_summary()

violations = get_violations()

records = get_recent_records()


# ============================================================
# OVERALL STATUS
# ============================================================

st.subheader(
    "Overall Status"
)

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Total Records",
        summary["total"]
    )


with col2:

    st.metric(
        "Compliant",
        summary["compliant"]
    )


with col3:

    st.metric(
        "Non-Compliant",
        summary["non_compliant"]
    )


with col4:

    st.metric(
        "Partial",
        summary["partial"]
    )


with col5:

    st.metric(
        "Unknown",
        summary["unknown"]
    )


# ============================================================
# PPE VIOLATIONS
# ============================================================

st.divider()

st.subheader(
    "⚠️ PPE Violations"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "No Helmet",
        violations["helmet"]
    )


with col2:

    st.metric(
        "No Vest",
        violations["vest"]
    )


with col3:

    st.metric(
        "No Safety Shoes",
        violations["shoes"]
    )


with col4:

    st.metric(
        "Without Goggles",
        violations["goggles"]
    )


# ============================================================
# PROCESSED IMAGE VIEWER
# ============================================================

st.divider()

st.subheader(
    "🖼️ Processed PPE Images"
)


processed_images = sorted(
    OUTPUT_IMAGE_DIR.glob(
        "*_refined_ppe_association.jpg"
    )
)


if processed_images:

    image_names = [
        image.name
        for image in processed_images
    ]

    selected_image = st.selectbox(
        "Select a processed image",
        image_names
    )

    selected_path = (
        OUTPUT_IMAGE_DIR
        /
        selected_image
    )

    st.image(
        str(selected_path),
        caption=selected_image,
        use_container_width=True
    )

else:

    st.info(
        "No processed PPE images found."
    )


# ============================================================
# RECENT RECORDS
# ============================================================

st.divider()

st.subheader(
    "📋 Recent PPE Records"
)


if records:

    table_data = []

    for record in records:

        table_data.append(
            {
                "ID": record[0],
                "Timestamp": record[1],
                "Source": record[2],
                "Person": record[3],
                "Helmet": record[4],
                "Vest": record[5],
                "Safety Shoes": record[6],
                "Safety Goggles": record[7],
                "Overall Status": record[8],
            }
        )

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No PPE detection records found."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PPE Detection System • Local SQLite Database"
)
