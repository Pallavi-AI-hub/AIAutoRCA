"""
Dynamic semantic retrieval for changing incident/RCA schemas.

Main callable function:

    retrieval(query, top_k_cases=None)

Flow:
1. Search Chroma chunks using semantic search.
2. Extract unique internal case IDs from matched chunk metadata.
3. Fetch complete parent cases from the Parquet case store.
4. Return all available case fields dynamically, without assuming fixed columns.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path) -> Path:
    """Resolve absolute paths directly and relative paths from project root."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return BASE_DIR / path


VECTOR_STORE_DIR = resolve_path(
    os.getenv("CHROMA_PERSIST_DIR", ".chroma/vector_store")
)

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "retail_product_incidents",
)

CASE_STORE_FILE = os.getenv(
    "PLIM_CASE_STORE_FILE",
    "retail_product_case_store.parquet",
)

CASE_STORE_PATH = VECTOR_STORE_DIR / CASE_STORE_FILE

TOP_K_CASES = int(
    os.getenv(
        "RETRIEVAL_TOP_K",
        os.getenv("HYBRID_TOP_K", "5"),
    )
)

CANDIDATE_MULTIPLIER = int(
    os.getenv(
        "RETRIEVAL_CANDIDATE_MULTIPLIER",
        os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "4"),
    )
)

SOURCE_DATASET = os.getenv(
    "RAG_SOURCE_DATASET",
    "retail_product_incident_history",
)

CASE_ID_LOOKUP_COLUMNS = [
    "incident_id",
    "incident_number",
    "plim_incident_id",
    "case_id",
    "ticket_id",
    "issue_id",
    "id",
]


# -----------------------------------------------------------------------------
# Store loading helpers
# -----------------------------------------------------------------------------


def get_chroma_collection() -> Any:
    """Open the persisted Chroma collection."""
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    return chroma_client.get_collection(name=COLLECTION_NAME)


def load_case_store() -> pd.DataFrame:
    """Load the full-case parent store from Parquet."""
    if not CASE_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Case store not found: {CASE_STORE_PATH}. Run ingestion.py first."
        )

    return pd.read_parquet(CASE_STORE_PATH)


def count_matching_chunks(collection: Any) -> int:
    """Count chunks for the configured source dataset."""
    result = collection.get(
        where={"source_dataset": SOURCE_DATASET},
        include=[],
    )

    return len(result.get("ids", []))


# -----------------------------------------------------------------------------
# Conversion helpers
# -----------------------------------------------------------------------------


def is_missing(value: object) -> bool:
    """Return True when a value is pandas/null missing."""
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def safe_value(value: object) -> Any:
    """Convert Parquet values into JSON-serializable Python values."""
    if is_missing(value):
        return ""

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def parse_json_field(value: object) -> Any:
    """Parse a JSON string when possible, otherwise return an empty dict."""
    text = safe_value(value)

    if not isinstance(text, str) or not text.strip():
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# -----------------------------------------------------------------------------
# Parent case fetching
# -----------------------------------------------------------------------------


def get_unique_case_ids_from_chunks(
    ranked_chunks: list[dict[str, Any]],
) -> list[str]:
    """Extract unique case IDs from matched chunk results in ranking order."""
    case_ids = []

    for chunk in ranked_chunks:
        metadata = chunk.get("metadata", {}) or {}

        case_id = (
            metadata.get("incident_id")
            or metadata.get("incident_number")
            or metadata.get("plim_incident_id")
            or metadata.get("case_id")
            or metadata.get("ticket_id")
            or metadata.get("id")
        )

        if case_id and case_id not in case_ids:
            case_ids.append(str(case_id))

    return case_ids


def find_case_rows(case_df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    """Find rows in the parent store by any known case id column."""
    for column in CASE_ID_LOOKUP_COLUMNS:
        if column not in case_df.columns:
            continue

        matched_rows = case_df[case_df[column].astype(str) == str(case_id)]
        if not matched_rows.empty:
            return matched_rows

    return case_df.iloc[0:0]


def row_to_dynamic_case(row: pd.Series) -> dict[str, Any]:
    """Return a complete case dict without assuming fixed source columns."""
    case = {column: safe_value(row[column]) for column in row.index}

    if "raw_fields_json" in case:
        case["raw_fields"] = parse_json_field(case["raw_fields_json"])

    if "source_schema_json" in case:
        case["source_schema"] = parse_json_field(case["source_schema_json"])

    if "source_column_mapping_json" in case:
        case["source_column_mapping"] = parse_json_field(case["source_column_mapping_json"])

    return case


def fetch_full_cases_from_parquet(
    case_df: pd.DataFrame,
    case_ids: list[str],
) -> list[dict[str, Any]]:
    """Fetch full parent cases dynamically from Parquet."""
    full_cases = []

    for case_id in case_ids:
        matched_rows = find_case_rows(case_df, case_id)

        if matched_rows.empty:
            continue

        row = matched_rows.iloc[0]
        full_cases.append(row_to_dynamic_case(row))

    return full_cases


# -----------------------------------------------------------------------------
# Semantic retrieval
# -----------------------------------------------------------------------------


def retrieval(
    query: str,
    top_k_cases: int | None = None,
) -> dict[str, Any]:
    """Retrieve complete incident cases with semantic search."""
    top_k_cases = top_k_cases or TOP_K_CASES

    collection = get_chroma_collection()
    case_df = load_case_store()
    chunk_count = count_matching_chunks(collection)

    if chunk_count == 0:
        return {
            "query": query,
            "unique_case_ids": [],
            "full_cases": [],
            "matched_chunks": [],
            "message": "No chunks found in Chroma collection.",
            "config": {
                "retrieval_type": "semantic_dynamic_schema",
                "vector_store_dir": str(VECTOR_STORE_DIR),
                "collection_name": COLLECTION_NAME,
                "case_store_path": str(CASE_STORE_PATH),
                "top_k_cases": top_k_cases,
                "source_dataset": SOURCE_DATASET,
            },
        }

    candidate_count = min(
        top_k_cases * CANDIDATE_MULTIPLIER,
        chunk_count,
    )

    result = collection.query(
        query_texts=[query],
        n_results=candidate_count,
        where={"source_dataset": SOURCE_DATASET},
        include=["documents", "metadatas", "distances"],
    )

    matched_chunks = []

    for rank, chunk_data in enumerate(
        zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ),
        start=1,
    ):
        chunk_id, chunk_text, metadata, distance = chunk_data

        matched_chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "metadata": metadata or {},
                "hybrid_score": None,
                "semantic_rank": rank,
                "semantic_distance": distance,
                "bm25_rank": None,
                "bm25_score": None,
            }
        )

    case_ids = get_unique_case_ids_from_chunks(matched_chunks)
    case_ids = case_ids[:top_k_cases]

    full_cases = fetch_full_cases_from_parquet(
        case_df=case_df,
        case_ids=case_ids,
    )

    return {
        "query": query,
        "unique_case_ids": case_ids,
        "full_cases": full_cases,
        "matched_chunks": matched_chunks,
        "config": {
            "retrieval_type": "semantic_dynamic_schema",
            "vector_store_dir": str(VECTOR_STORE_DIR),
            "collection_name": COLLECTION_NAME,
            "case_store_path": str(CASE_STORE_PATH),
            "top_k_cases": top_k_cases,
            "source_dataset": SOURCE_DATASET,
        },
    }


# -----------------------------------------------------------------------------
# Example run
# -----------------------------------------------------------------------------


def main() -> None:
    """Run a simple dynamic semantic retrieval test."""
    query = "duplicate key value violates unique constraint"
    result = retrieval(query=query, top_k_cases=1)

    print("\nDYNAMIC SEMANTIC RETRIEVAL")
    print("=" * 80)
    print("Unique case IDs:", result["unique_case_ids"])

    for index, case in enumerate(result["full_cases"], start=1):
        print(f"\nCase Rank: {index}")
        print(f"Incident ID: {case.get('incident_id', '')}")
        print(f"Source dataset: {case.get('source_dataset', '')}")
        print(f"Schema signature: {case.get('schema_signature', '')}")
        print("-" * 80)
        print(case.get("case_document", ""))


if __name__ == "__main__":
    main()
