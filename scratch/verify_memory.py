import os
import sys
import psycopg2


def query_memory():
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

        print("--- Querying homelab_memory_v2 ---")
        cur.execute(
            """
            SELECT category, content, project, created_at 
            FROM homelab_memory_v2 
            WHERE content ILIKE '%krusch-agentic-proxy%' 
               OR content ILIKE '%krusch-agentic-mcp%' 
               OR project ILIKE '%krusch-agentic-proxy%' 
               OR project ILIKE '%krusch-agentic-mcp%' 
               OR content ILIKE '%gpu%'
            ORDER BY created_at DESC LIMIT 10;
            """
        )
        for r in cur.fetchall():
            print(f"[{r[3]}] Category: {r[0]} | Project: {r[2]}")
            print(f"Content: {r[1]}")
            print("-" * 50)

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error querying memory database: {e}", file=sys.stderr)


if __name__ == "__main__":
    query_memory()
