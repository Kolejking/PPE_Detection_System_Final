import sqlite3

from src.database import PPEDatabase


class PPEReport:

    def __init__(self):
        self.database = PPEDatabase()

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    def get_summary(self):

        conn = sqlite3.connect(
            self.database.db_path
        )

        cursor = conn.cursor()

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
                ) AS compliant,

                SUM(
                    CASE
                        WHEN overall_status = 'NON-COMPLIANT'
                        THEN 1
                        ELSE 0
                    END
                ) AS non_compliant,

                SUM(
                    CASE
                        WHEN overall_status = 'PARTIAL'
                        THEN 1
                        ELSE 0
                    END
                ) AS partial,

                SUM(
                    CASE
                        WHEN overall_status = 'UNKNOWN'
                        THEN 1
                        ELSE 0
                    END
                ) AS unknown

            FROM ppe_detections
            """
        )

        row = cursor.fetchone()

        conn.close()

        return {
            "total": row[0] or 0,
            "compliant": row[1] or 0,
            "non_compliant": row[2] or 0,
            "partial": row[3] or 0,
            "unknown": row[4] or 0,
        }

    # ========================================================
    # PPE VIOLATIONS SUMMARY
    # ========================================================

    def get_violations(self):

        conn = sqlite3.connect(
            self.database.db_path
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT

                SUM(
                    CASE
                        WHEN helmet_status = 'NO-HELMET'
                        THEN 1
                        ELSE 0
                    END
                ) AS helmet_violations,

                SUM(
                    CASE
                        WHEN vest_status = 'NO-VEST'
                        THEN 1
                        ELSE 0
                    END
                ) AS vest_violations,

                SUM(
                    CASE
                        WHEN safety_shoes_status =
                             'NO-SAFETY-SHOES'
                        THEN 1
                        ELSE 0
                    END
                ) AS shoe_violations,

                SUM(
                    CASE
                        WHEN safety_goggles_status =
                             'WITHOUT-GOGGLES'
                        THEN 1
                        ELSE 0
                    END
                ) AS goggles_violations

            FROM ppe_detections
            """
        )

        row = cursor.fetchone()

        conn.close()

        return {
            "helmet": row[0] or 0,
            "vest": row[1] or 0,
            "safety_shoes": row[2] or 0,
            "safety_goggles": row[3] or 0,
        }

    # ========================================================
    # GET RECENT RECORDS
    # ========================================================

    def get_records(
        self,
        limit: int = 20
    ):

        conn = sqlite3.connect(
            self.database.db_path
        )

        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

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

        conn.close()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # PRINT ALL RECORDS
    # ========================================================

    def print_records(
        self,
        limit: int = 20
    ):

        records = self.get_records(
            limit
        )

        print()
        print("=" * 100)
        print("                    PPE DETECTION RECORDS")
        print("=" * 100)

        if not records:

            print()
            print("No records found.")

            print()
            print("=" * 100)

            return

        for record in records:

            print()
            print("-" * 100)

            print(
                f"Record ID     : "
                f"{record['id']}"
            )

            print(
                f"Timestamp     : "
                f"{record['timestamp']}"
            )

            print(
                f"Source Image  : "
                f"{record['source']}"
            )

            print(
                f"Person        : "
                f"{record['person_index']}"
            )

            print(
                f"Helmet        : "
                f"{record['helmet_status']}"
            )

            print(
                f"Vest          : "
                f"{record['vest_status']}"
            )

            print(
                f"Safety Shoes  : "
                f"{record['safety_shoes_status']}"
            )

            print(
                f"Safety Goggles: "
                f"{record['safety_goggles_status']}"
            )

            print(
                f"Overall Status: "
                f"{record['overall_status']}"
            )

        print()
        print("=" * 100)

    # ========================================================
    # GET VIOLATION RECORDS ONLY
    # ========================================================

    def get_violation_records(self):

        conn = sqlite3.connect(
            self.database.db_path
        )

        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

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

            WHERE
                helmet_status = 'NO-HELMET'
                OR vest_status = 'NO-VEST'
                OR safety_shoes_status = 'NO-SAFETY-SHOES'
                OR safety_goggles_status = 'WITHOUT-GOGGLES'

            ORDER BY id DESC
            """
        )

        rows = cursor.fetchall()

        conn.close()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # PRINT VIOLATIONS ONLY
    # ========================================================

    def print_violations(self):

        records = self.get_violation_records()

        print()
        print("=" * 100)
        print("                    PPE VIOLATIONS")
        print("=" * 100)

        if not records:

            print()
            print("No PPE violations found.")

            print()
            print("=" * 100)

            return

        for record in records:

            print()
            print("-" * 100)

            print(
                f"Record ID     : "
                f"{record['id']}"
            )

            print(
                f"Timestamp     : "
                f"{record['timestamp']}"
            )

            print(
                f"Source Image  : "
                f"{record['source']}"
            )

            print(
                f"Person        : "
                f"{record['person_index']}"
            )

            # ------------------------------------------------
            # Show only actual violations
            # ------------------------------------------------

            if (
                record["helmet_status"]
                == "NO-HELMET"
            ):

                print(
                    "⚠ Helmet        : NO-HELMET"
                )

            if (
                record["vest_status"]
                == "NO-VEST"
            ):

                print(
                    "⚠ Vest          : NO-VEST"
                )

            if (
                record["safety_shoes_status"]
                == "NO-SAFETY-SHOES"
            ):

                print(
                    "⚠ Safety Shoes  : "
                    "NO-SAFETY-SHOES"
                )

            if (
                record["safety_goggles_status"]
                == "WITHOUT-GOGGLES"
            ):

                print(
                    "⚠ Safety Goggles: "
                    "WITHOUT-GOGGLES"
                )

            print(
                f"Overall Status: "
                f"{record['overall_status']}"
            )

        print()
        print("=" * 100)

    # ========================================================
    # COMPLETE REPORT
    # ========================================================

    def print_report(self):

        summary = self.get_summary()

        violations = self.get_violations()

        print()
        print("=" * 55)
        print("              PPE DATABASE REPORT")
        print("=" * 55)

        print()
        print("OVERALL STATUS")
        print("-" * 55)

        print(
            f"Total Records       : "
            f"{summary['total']}"
        )

        print(
            f"Compliant           : "
            f"{summary['compliant']}"
        )

        print(
            f"Non-Compliant       : "
            f"{summary['non_compliant']}"
        )

        print(
            f"Partial             : "
            f"{summary['partial']}"
        )

        print(
            f"Unknown             : "
            f"{summary['unknown']}"
        )

        print()
        print("PPE VIOLATIONS")
        print("-" * 55)

        print(
            f"Helmet Violations   : "
            f"{violations['helmet']}"
        )

        print(
            f"Vest Violations     : "
            f"{violations['vest']}"
        )

        print(
            f"Shoe Violations     : "
            f"{violations['safety_shoes']}"
        )

        print(
            f"Goggle Violations   : "
            f"{violations['safety_goggles']}"
        )

        print()
        print("=" * 55)
        