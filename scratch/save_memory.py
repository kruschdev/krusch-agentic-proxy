import os
import sys
import psycopg2


def insert_memory():
    db_password = os.environ.get("DB_PASSWORD", "")
    if not db_password:
        print("Error: DB_PASSWORD environment variable is required.", file=sys.stderr)
        return
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST", "10.10.10.2"),
            port=os.environ.get("DB_PORT", "5434"),
            database=os.environ.get("DB_NAME", "kruschdb"),
            user=os.environ.get("DB_USER", "openclaw"),
            password=db_password,
        )
        cur = conn.cursor()

        cur.execute("SELECT author_id FROM homelab_memory_v2 WHERE author_id IS NOT NULL LIMIT 1")
        author_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO homelab_memory_v2 (category, content, project, author_id)
            VALUES (%s, %s, %s, %s)
            """,
            (
                "activity",
                "[krusch-agentic-proxy] Completed v0.4.0 hardening and 5-GPU fleet synchronization.",
                "krusch-agentic-proxy",
                author_id,
            ),
        )

        conn.commit()
        cur.close()
        conn.close()
        print("Successfully saved memory state.")
    except Exception as e:
        print(f"Error querying memory database: {e}", file=sys.stderr)


if __name__ == "__main__":
    insert_memory()
