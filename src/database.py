import sqlite3
from pathlib import Path
from datetime import datetime


class PPEDatabase:

    def __init__(self, db_path: str = "database/ppe_detection.db"):

        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.create_table()

    def create_table(self):

        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ppe_detections (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,
                source TEXT,

                track_id INTEGER,
                person_index INTEGER,

                helmet_status TEXT,
                helmet_score REAL,

                vest_status TEXT,
                vest_score REAL,

                safety_shoes_status TEXT,
                safety_shoes_score REAL,

                safety_goggles_status TEXT,
                safety_goggles_score REAL,

                overall_status TEXT
            )
            """
        )

        connection.commit()
        connection.close()

    def insert_detection(
        self,
        status: dict,
        source: str = "",
    ):

        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        cursor.execute(
            """
            INSERT INTO ppe_detections (

                timestamp,
                source,

                track_id,
                person_index,

                helmet_status,
                helmet_score,

                vest_status,
                vest_score,

                safety_shoes_status,
                safety_shoes_score,

                safety_goggles_status,
                safety_goggles_score,

                overall_status
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                source,

                status.get("track_id"),
                status.get("person_index"),

                status.get("helmet_status"),
                status.get("helmet_score", 0.0),

                status.get("vest_status"),
                status.get("vest_score", 0.0),

                status.get(
                    "safety_shoes_status",
                    status.get("shoes_status")
                ),

                status.get(
                    "safety_shoes_score",
                    status.get("shoes_score", 0.0)
                ),

                status.get(
                    "safety_goggles_status",
                    status.get("goggles_status")
                ),

                status.get(
                    "safety_goggles_score",
                    status.get("goggles_score", 0.0)
                ),

                status.get("overall_status"),
            )
        )

        connection.commit()
        connection.close()

    def insert_detections(
        self,
        statuses: list,
        source: str = "",
    ):

        for status in statuses:
            self.insert_detection(
                status=status,
                source=source,
            )

    def get_recent_detections(
        self,
        limit: int = 20,
    ):

        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM ppe_detections
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()

        connection.close()

        return [
            dict(row)
            for row in rows
        ]

    def count_records(self) -> int:

        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM ppe_detections
            """
        )

        count = cursor.fetchone()[0]

        connection.close()

        return count