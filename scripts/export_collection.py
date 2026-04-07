"""Export a Qdrant collection to a local CSV file.

Usage:
    python scripts/export_collection_to_csv.py <collection_name> [options]

Examples:
    python scripts/export_collection_to_csv.py child_chunks
    python scripts/export_collection_to_csv.py parent_store --output exports/parent.csv
    python scripts/export_collection_to_csv.py child_chunks --no-vectors --batch-size 500
    python scripts/export_collection_to_csv.py child_chunks --sort document
    python scripts/export_collection_to_csv.py child_chunks --host localhost --port 6333

Sort modes (--sort):
    none      Keep Qdrant scroll order (default, streams directly — no memory buffering)
    document  Sort by filename → parent index → child index, preserving original doc order
"""

import argparse
import asyncio
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_CHUNK_ID_RE = re.compile(r"_p(\d+)_c(\d+)$")


def _document_sort_key(row: dict) -> tuple:
    """Sort key that reconstructs original document order from chunk_id.

    chunk_id format: {doc_id}_p{parent_index}_c{child_index}
    Falls back gracefully when the field is absent or unparseable.
    """
    raw_meta = row.get("metadata", "")
    try:
        meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
    except (json.JSONDecodeError, TypeError):
        meta = {}

    filename = str(meta.get("filename") or "")
    chunk_id = str(meta.get("chunk_id") or "")

    m = _CHUNK_ID_RE.search(chunk_id)
    parent_idx = int(m.group(1)) if m else 0
    child_idx = int(m.group(2)) if m else 0

    return (filename, parent_idx, child_idx)


async def _scroll_all(
    client,
    collection_name: str,
    batch_size: int,
    include_vectors: bool,
    total_points: int,
) -> list[dict]:
    """Fetch all points and return as a list of row dicts."""
    rows: list[dict] = []
    offset = None
    fieldnames: list[str] | None = None

    while True:
        result, next_offset = await client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=include_vectors,
        )

        if not result:
            break

        for point in result:
            payload = point.payload or {}
            row: dict = {"id": str(point.id)}

            for key, val in payload.items():
                if isinstance(val, (dict, list)):
                    row[key] = json.dumps(val, ensure_ascii=False)
                else:
                    row[key] = val

            if include_vectors and point.vector is not None:
                if isinstance(point.vector, list):
                    row["vector"] = json.dumps(point.vector)
                elif isinstance(point.vector, dict):
                    for vec_name, vec_vals in point.vector.items():
                        row[f"vector_{vec_name}"] = json.dumps(vec_vals)

            if fieldnames is None:
                fieldnames = list(row.keys())

            rows.append(row)

        print(f"  Fetched {len(rows)}/{total_points} points...", end="\r")

        if next_offset is None:
            break
        offset = next_offset

    print()  # newline after progress line
    return rows


async def export_collection(
    collection_name: str,
    output_path: Path,
    include_vectors: bool,
    batch_size: int,
    sort: str,
    host: str,
    port: int,
    api_key: str | None,
) -> None:
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(
        host=host,
        port=port,
        api_key=api_key or None,
        timeout=30.0,
    )

    try:
        info = await client.get_collection(collection_name)
    except Exception as exc:
        print(f"Error: could not access collection '{collection_name}': {exc}", file=sys.stderr)
        await client.close()
        sys.exit(1)

    total_points = info.points_count
    print(f"Collection '{collection_name}': {total_points} points")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sort == "document":
        # Buffer all rows in memory, then sort, then write
        rows = await _scroll_all(client, collection_name, batch_size, include_vectors, total_points)
        await client.close()

        print(f"Sorting {len(rows)} rows by document order...")
        rows.sort(key=_document_sort_key)

        if not rows:
            print("No rows to write.")
            return

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print(f"Done. {len(rows)} rows saved to: {output_path}")

    else:
        # Stream directly — no memory buffering
        offset = None
        rows_written = 0
        writer_initialized = False

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = None

            while True:
                result, next_offset = await client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=include_vectors,
                )

                if not result:
                    break

                for point in result:
                    payload = point.payload or {}
                    row: dict = {"id": str(point.id)}

                    for key, val in payload.items():
                        if isinstance(val, (dict, list)):
                            row[key] = json.dumps(val, ensure_ascii=False)
                        else:
                            row[key] = val

                    if include_vectors and point.vector is not None:
                        if isinstance(point.vector, list):
                            row["vector"] = json.dumps(point.vector)
                        elif isinstance(point.vector, dict):
                            for vec_name, vec_vals in point.vector.items():
                                row[f"vector_{vec_name}"] = json.dumps(vec_vals)

                    if not writer_initialized:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
                        writer.writeheader()
                        writer_initialized = True

                    writer.writerow(row)
                    rows_written += 1

                print(f"  Exported {rows_written}/{total_points} points...", end="\r")

                if next_offset is None:
                    break
                offset = next_offset

        await client.close()
        print(f"\nDone. {rows_written} rows saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Qdrant collection to a CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("collection", help="Qdrant collection name to export")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to exports/<collection>_<timestamp>.csv",
    )
    parser.add_argument(
        "--sort",
        choices=["none", "document"],
        default="none",
        help=(
            "Sort order. 'document' sorts by filename → parent index → child index "
            "using the chunk_id metadata field, preserving original document order. "
            "'none' streams directly from Qdrant (faster, no buffering)."
        ),
    )
    parser.add_argument(
        "--no-vectors",
        action="store_true",
        help="Exclude embedding vectors from the CSV (much smaller file)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="Number of points to fetch per Qdrant scroll request",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Qdrant host. Falls back to QDRANT_HOST env var, then 'localhost'",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Qdrant port. Falls back to QDRANT_PORT env var, then 6333",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Qdrant API key. Falls back to QDRANT_API_KEY env var",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    from app.core.config import settings

    host = args.host or settings.QDRANT_HOST
    port = args.port or settings.QDRANT_PORT
    api_key = args.api_key or settings.QDRANT_API_KEY or None

    output_path = args.output or Path("exports") / (
        f"{args.collection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    print(f"Connecting to Qdrant at {host}:{port}")
    print(f"Exporting collection: {args.collection}")
    print(f"Sort: {args.sort}")
    print(f"Include vectors: {not args.no_vectors}")
    print(f"Output: {output_path}\n")

    await export_collection(
        collection_name=args.collection,
        output_path=output_path,
        include_vectors=not args.no_vectors,
        batch_size=args.batch_size,
        sort=args.sort,
        host=host,
        port=port,
        api_key=api_key,
    )


if __name__ == "__main__":
    asyncio.run(main())
