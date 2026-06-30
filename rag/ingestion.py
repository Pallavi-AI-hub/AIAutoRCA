
"""
Dynamic ingestion for incident/RCA CSV datasets.
 
Main callable function:
 
    ingest(input_dataset_csv_path)
 
Design goal:
- Accept CSV files whose columns change over time.
- Preserve every source column in the Parquet parent store.
- Build searchable case documents dynamically from whatever columns exist.
- Store Chroma chunks only for search, then retrieve complete parent cases later.
 
The only stable contract is the internal case id:
- If a known id column exists, it is used.
- Otherwise, a deterministic id is generated from the row content.
"""
 
from __future__ import annotations
 
import hashlib
import json
import os
import re
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
 
 
def get_default_csv_file_path() -> Path:
    """Read the default incident CSV path from .env."""
    csv_path = os.getenv("PLIM_INCIDENT_CSV_PATH")
 
    if not csv_path:
        raise ValueError(
            "Missing CSV path. Set PLIM_INCIDENT_CSV_PATH in your .env file "
            "or call ingest(input_dataset_csv_path) directly."
        )
 
    return resolve_path(csv_path)
 
 
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
 
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP_SIZE", "200"))
CHROMA_BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", "500"))
 
RESET_CHROMA_COLLECTION = os.getenv(
    "RESET_CHROMA_COLLECTION",
    "true",
).lower() == "true"
 
REPLACE_SOURCE_DATASET = os.getenv(
    "REPLACE_SOURCE_DATASET",
    "true",
).lower() == "true"
 
SOURCE_DATASET = os.getenv(
    "RAG_SOURCE_DATASET",
    "retail_product_incident_history",
)
 
# Optional. Set this when your CSV has a specific id column you want to use.
# Example: RAG_CASE_ID_COLUMN=ticket_id
CASE_ID_COLUMN_OVERRIDE = os.getenv("RAG_CASE_ID_COLUMN", "").strip()
 
# Optional comma-separated metadata fields. Only existing columns are used.
# Example: RAG_METADATA_COLUMNS=priority,status,dag_id,task_id,owner
METADATA_COLUMNS_OVERRIDE = os.getenv("RAG_METADATA_COLUMNS", "").strip()
 
MAX_METADATA_COLUMNS = int(os.getenv("RAG_MAX_METADATA_COLUMNS", "20"))
MAX_METADATA_VALUE_CHARS = int(os.getenv("RAG_MAX_METADATA_VALUE_CHARS", "300"))
 
DEFAULT_CASE_ID_CANDIDATES = [
    "incident_id",
    "incident_number",
    "plim_incident_id",
    "case_id",
    "ticket_id",
    "ticket_number",
    "issue_id",
    "jira_id",
    "id",
]
 
DEFAULT_METADATA_HINTS = [
    "priority",
    "severity",
    "status",
    "scenario_type",
    "category",
    "failure_pattern",
    "dag_id",
    "task_id",
    "source_system",
    "target_system",
    "target_table",
    "owner",
    "team",
    "environment",
    "service",
    "database",
    "table",
    "recurrence_flag",
]
 
INTERNAL_COLUMNS = {
    "case_document",
    "comment_length",
    "source_dataset",
    "source_schema_json",
    "source_column_mapping_json",
    "raw_fields_json",
    "rag_case_id_source_column",
    "schema_signature",
}
 
 
# -----------------------------------------------------------------------------
# Cleaning and schema helpers
# -----------------------------------------------------------------------------
 
 
def clean_text(value: object) -> str:
    """Convert any CSV cell into clean text without truncating content."""
    if value is None:
        return ""
 
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
 
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
 
    cleaned_lines = []
    for line in text.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(cleaned_line)
 
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
 
    return text.strip()
 
 
def normalize_column_name(column_name: object) -> str:
    """Normalize source CSV headers into safe snake_case column names."""
    cleaned = clean_text(column_name)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
 
    return cleaned or "column"
 
 
def make_unique_column_names(column_names: list[str]) -> list[str]:
    """Make normalized column names unique by adding numeric suffixes."""
    counts: dict[str, int] = {}
    unique_names = []
 
    for column_name in column_names:
        base_name = column_name
        count = counts.get(base_name, 0)
 
        if count == 0:
            unique_name = base_name
        else:
            unique_name = f"{base_name}_{count + 1}"
 
        counts[base_name] = count + 1
        unique_names.append(unique_name)
 
    return unique_names
 
 
def normalize_dataframe_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize column names and return original to normalized mapping."""
    original_columns = [str(column) for column in df.columns]
    normalized_columns = [normalize_column_name(column) for column in original_columns]
    unique_columns = make_unique_column_names(normalized_columns)
 
    column_mapping = {
        original: normalized
        for original, normalized in zip(original_columns, unique_columns)
    }
 
    df = df.copy()
    df.columns = unique_columns
 
    return df, column_mapping
 
 
def json_dumps(value: Any) -> str:
    """Serialize a value into stable JSON text."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
 
 
def build_schema_signature(columns: list[str]) -> str:
    """Create a short signature for the current normalized schema."""
    payload = json_dumps(columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
 
 
def build_row_hash(row: pd.Series, columns: list[str]) -> str:
    """Build a deterministic row hash from all source columns."""
    payload = {
        column: clean_text(row.get(column, ""))
        for column in columns
    }
    return hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()[:16]
 
 
def parse_csv_list(value: str) -> list[str]:
    """Parse comma-separated .env values."""
    if not value:
        return []
 
    return [
        normalize_column_name(item)
        for item in value.split(",")
        if normalize_column_name(item)
    ]
 
 
def choose_case_id_column(df: pd.DataFrame) -> str | None:
    """Choose a case id column dynamically."""
    normalized_override = normalize_column_name(CASE_ID_COLUMN_OVERRIDE)
 
    if CASE_ID_COLUMN_OVERRIDE and normalized_override in df.columns:
        return normalized_override
 
    for candidate in DEFAULT_CASE_ID_CANDIDATES:
        if candidate in df.columns and df[candidate].astype(str).str.strip().ne("").any():
            return candidate
 
    return None
 
 
def combine_non_empty_values(values: pd.Series) -> str:
    """Combine unique non-empty values from duplicate case rows."""
    cleaned_values = []
 
    for value in values.tolist():
        cleaned_value = clean_text(value)
        if cleaned_value and cleaned_value not in cleaned_values:
            cleaned_values.append(cleaned_value)
 
    return "\n\n".join(cleaned_values)
 
 
def prepare_dynamic_dataframe(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean, normalize, deduplicate, and assign internal case ids."""
    if raw_df.empty:
        raise ValueError("Input CSV has no rows.")
 
    stats: dict[str, Any] = {
        "raw_rows": len(raw_df),
        "raw_columns": len(raw_df.columns),
        "exact_duplicate_rows_removed": 0,
        "duplicate_case_ids_merged": 0,
        "generated_case_ids": 0,
        "final_case_rows": 0,
    }
 
    df, column_mapping = normalize_dataframe_columns(raw_df)
    source_columns = list(df.columns)
    schema_signature = build_schema_signature(source_columns)
 
    # Convert every source cell to cleaned text. No content is truncated.
    for column in source_columns:
        df[column] = df[column].apply(clean_text)
 
    # Preserve the original source row as JSON before adding internal columns.
    df["raw_fields_json"] = df.apply(
        lambda row: json_dumps({column: row.get(column, "") for column in source_columns}),
        axis=1,
    )
 
    before_drop_duplicates = len(df)
    df = df.drop_duplicates(subset=source_columns).copy()
    stats["exact_duplicate_rows_removed"] = before_drop_duplicates - len(df)
 
    case_id_source_column = choose_case_id_column(df)
    case_ids = []
 
    for index, row in df.iterrows():
        if case_id_source_column:
            case_id = clean_text(row.get(case_id_source_column, ""))
        else:
            case_id = ""
 
        if not case_id:
            case_id = f"GENERATED_CASE_{build_row_hash(row, source_columns)}"
            stats["generated_case_ids"] += 1
 
        case_ids.append(case_id)
 
    # If the source has incident_id but we choose a different id field, preserve it.
    if "incident_id" in df.columns and case_id_source_column != "incident_id":
        preserve_column = "source_original_incident_id"
        suffix = 2
        while preserve_column in df.columns:
            preserve_column = f"source_original_incident_id_{suffix}"
            suffix += 1
        df[preserve_column] = df["incident_id"]
 
    # Internal stable id contract for the rest of the RAG pipeline.
    df["incident_id"] = case_ids
    df["incident_number"] = df["incident_id"]
    df["plim_incident_id"] = df["incident_id"]
    df["rag_case_id_source_column"] = case_id_source_column or "generated_from_row_hash"
    df["source_dataset"] = SOURCE_DATASET
    df["source_schema_json"] = json_dumps(source_columns)
    df["source_column_mapping_json"] = json_dumps(column_mapping)
    df["schema_signature"] = schema_signature
 
    duplicate_case_count = int(df["incident_id"].duplicated().sum())
 
    if duplicate_case_count:
        stats["duplicate_case_ids_merged"] = duplicate_case_count
        df = (
            df.groupby("incident_id", sort=False, as_index=False)
            .agg(combine_non_empty_values)
            .copy()
        )
        df["incident_number"] = df["incident_id"]
        df["plim_incident_id"] = df["incident_id"]
 
    stats["final_case_rows"] = len(df)
    stats["case_id_source_column"] = case_id_source_column or "generated_from_row_hash"
    stats["schema_signature"] = schema_signature
    stats["normalized_columns"] = source_columns
 
    return df.reset_index(drop=True), stats
 
 
# -----------------------------------------------------------------------------
# Full case document creation
# -----------------------------------------------------------------------------
 
 
def humanize_column_name(column_name: str) -> str:
    """Convert snake_case column names into readable labels."""
    return column_name.replace("_", " ").strip().title()
 
 
def ordered_document_columns(row: pd.Series) -> list[str]:
    """Return columns in a useful display order without relying on a fixed schema."""
    all_columns = [
        column
        for column in row.index.tolist()
        if column not in INTERNAL_COLUMNS
        and column not in {"incident_number", "plim_incident_id"}
    ]
 
    priority_tokens = [
        "incident_id",
        "case_id",
        "ticket_id",
        "issue_title",
        "title",
        "summary",
        "scenario_type",
        "priority",
        "severity",
        "status",
        "dag_id",
        "task_id",
        "execution_date",
        "failure_time",
        "resolved_time",
        "duration",
        "error_message",
        "log",
        "root_cause",
        "fix",
        "resolution",
        "business_impact",
        "preventive_action",
    ]
 
    selected = []
    for token in priority_tokens:
        for column in all_columns:
            if column not in selected and (column == token or token in column):
                selected.append(column)
 
    for column in all_columns:
        if column not in selected:
            selected.append(column)
 
    return selected
 
 
def build_case_document(row: pd.Series) -> str:
    """Build one full case document from whatever columns exist."""
    lines = [f"Incident ID: {clean_text(row.get('incident_id', ''))}", ""]
 
    for column in ordered_document_columns(row):
        value = clean_text(row.get(column, ""))
        if not value:
            continue
 
        label = humanize_column_name(column)
 
        if "\n" in value or len(value) > 120:
            lines.append(f"{label}:")
            lines.append(value)
            lines.append("")
        else:
            lines.append(f"{label}: {value}")
 
    return "\n".join(lines).strip()
 
 
def build_case_store(df: pd.DataFrame) -> pd.DataFrame:
    """Create the parent case store while preserving all dynamic columns."""
    case_df = df.copy()
    case_df["case_document"] = case_df.apply(build_case_document, axis=1)
    case_df["comment_length"] = case_df["case_document"].str.len()
 
    preferred_first = [
        "incident_id",
        "incident_number",
        "plim_incident_id",
        "source_dataset",
        "schema_signature",
        "rag_case_id_source_column",
        "comment_length",
        "case_document",
        "raw_fields_json",
        "source_schema_json",
        "source_column_mapping_json",
    ]
 
    ordered_columns = [column for column in preferred_first if column in case_df.columns]
    ordered_columns.extend([column for column in case_df.columns if column not in ordered_columns])
 
    return case_df[ordered_columns].copy()
 
 
# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------
 
 
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a full case document into overlapping chunks."""
    if chunk_size <= overlap:
        raise ValueError("RAG_CHUNK_SIZE must be greater than RAG_CHUNK_OVERLAP_SIZE.")
 
    if len(text) <= chunk_size:
        return [text]
 
    chunks = []
    start = 0
 
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
 
        if chunk:
            chunks.append(chunk)
 
        start = end - overlap
 
    return chunks
 
 
def infer_metadata_columns(case_df: pd.DataFrame) -> list[str]:
    """Infer safe, useful metadata fields from the current schema."""
    if METADATA_COLUMNS_OVERRIDE:
        requested_columns = parse_csv_list(METADATA_COLUMNS_OVERRIDE)
        return [column for column in requested_columns if column in case_df.columns]
 
    metadata_columns = []
 
    for column in DEFAULT_METADATA_HINTS:
        if column in case_df.columns and column not in metadata_columns:
            metadata_columns.append(column)
 
    # Add short categorical/id-like columns discovered from the current schema.
    name_tokens = [
        "id",
        "status",
        "priority",
        "severity",
        "type",
        "category",
        "system",
        "table",
        "source",
        "target",
        "owner",
        "team",
        "env",
        "service",
    ]
 
    for column in case_df.columns:
        if column in metadata_columns or column in INTERNAL_COLUMNS:
            continue
        if column in {"case_document", "raw_fields_json"}:
            continue
        if any(token in column for token in name_tokens):
            metadata_columns.append(column)
        if len(metadata_columns) >= MAX_METADATA_COLUMNS:
            break
 
    return metadata_columns[:MAX_METADATA_COLUMNS]
 
 
def metadata_safe_value(value: object) -> str | int | float | bool | None:
    """Return a Chroma-safe metadata scalar.
 
    Long values are not included as metadata because metadata is for filtering,
    not for source-of-truth storage. The full value remains in Parquet and in
    case_document.
    """
    text = clean_text(value)
 
    if not text:
        return None
 
    if len(text) > MAX_METADATA_VALUE_CHARS:
        return None
 
    return text
 
 
def build_metadata(row: pd.Series, metadata_columns: list[str], chunk_index: int, total_chunks: int) -> dict[str, Any]:
    """Build dynamic Chroma metadata for one chunk."""
    metadata: dict[str, Any] = {
        "incident_id": clean_text(row.get("incident_id", "")),
        "incident_number": clean_text(row.get("incident_number", row.get("incident_id", ""))),
        "plim_incident_id": clean_text(row.get("plim_incident_id", row.get("incident_id", ""))),
        "chunk_index": int(chunk_index),
        "total_chunks": int(total_chunks),
        "source_dataset": SOURCE_DATASET,
        "schema_signature": clean_text(row.get("schema_signature", "")),
    }
 
    for column in metadata_columns:
        if column in metadata:
            continue
        value = metadata_safe_value(row.get(column, ""))
        if value is not None:
            metadata[column] = value
 
    return metadata
 
 
def build_chunk_rows(case_df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per searchable chunk."""
    chunk_rows = []
    metadata_columns = infer_metadata_columns(case_df)
 
    for _, row in case_df.iterrows():
        incident_id = clean_text(row.get("incident_id", ""))
        case_document = clean_text(row.get("case_document", ""))
 
        chunks = chunk_text(
            text=case_document,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
 
        total_chunks = len(chunks)
 
        for chunk_index, chunk in enumerate(chunks):
            chunk_rows.append(
                {
                    "chunk_id": f"{incident_id}_chunk_{chunk_index}",
                    "incident_id": incident_id,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "chunk_text": chunk,
                    "metadata": build_metadata(row, metadata_columns, chunk_index, total_chunks),
                }
            )
 
    return pd.DataFrame(chunk_rows)
 
 
# -----------------------------------------------------------------------------
# Chroma ingestion
# -----------------------------------------------------------------------------
 
 
def collection_exists(chroma_client: chromadb.PersistentClient, name: str) -> bool:
    """Check whether a Chroma collection exists."""
    for collection in chroma_client.list_collections():
        collection_name = getattr(collection, "name", collection)
        if collection_name == name:
            return True
    return False
 
 
def delete_existing_source_dataset(collection: Any) -> None:
    """Delete existing chunks for this source dataset when replacing data."""
    if not REPLACE_SOURCE_DATASET:
        return
 
    try:
        collection.delete(where={"source_dataset": SOURCE_DATASET})
    except Exception:
        # Chroma can raise when no records match. That should not block ingestion.
        return
 
 
def ingest_chunks_into_chroma(chunk_df: pd.DataFrame) -> None:
    """Store searchable chunks in ChromaDB."""
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
 
    if RESET_CHROMA_COLLECTION and collection_exists(chroma_client, COLLECTION_NAME):
        chroma_client.delete_collection(name=COLLECTION_NAME)
 
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
 
    if not RESET_CHROMA_COLLECTION:
        delete_existing_source_dataset(collection)
 
    ids = chunk_df["chunk_id"].tolist()
    documents = chunk_df["chunk_text"].tolist()
    metadatas = chunk_df["metadata"].tolist()
 
    for start in range(0, len(ids), CHROMA_BATCH_SIZE):
        end = start + CHROMA_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
 
    print(f"Chunks ingested into Chroma: {len(ids)}")
    print(f"Total Chroma records: {collection.count()}")
 
 
# -----------------------------------------------------------------------------
# Public ingestion entry point
# -----------------------------------------------------------------------------
 
 
def ingest(input_dataset_csv_path: str | Path) -> dict[str, object]:
    """Ingest any CSV dataset into ChromaDB and a Parquet parent store."""
    csv_file_path = resolve_path(input_dataset_csv_path)
 
    if not csv_file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
 
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
 
    raw_df = pd.read_csv(
        csv_file_path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )
 
    incident_df, stats = prepare_dynamic_dataframe(raw_df)
 
    # Parent store: complete dynamic case documents and all source columns.
    case_df = build_case_store(incident_df)
    case_df.to_parquet(CASE_STORE_PATH, index=False)
 
    # Child store: searchable chunks derived from complete case documents.
    chunk_df = build_chunk_rows(case_df)
    ingest_chunks_into_chroma(chunk_df)
 
    result = {
        "raw_rows": stats["raw_rows"],
        "raw_columns": stats["raw_columns"],
        "exact_duplicate_rows_removed": stats["exact_duplicate_rows_removed"],
        "duplicate_case_ids_merged": stats["duplicate_case_ids_merged"],
        "generated_case_ids": stats["generated_case_ids"],
        "cases_stored_in_parquet": len(case_df),
        "chunks_created": len(chunk_df),
        "case_id_source_column": stats["case_id_source_column"],
        "schema_signature": stats["schema_signature"],
        "normalized_columns": stats["normalized_columns"],
        "csv_file": str(csv_file_path),
        "case_store_path": str(CASE_STORE_PATH),
        "vector_store_path": str(VECTOR_STORE_DIR),
        "collection_name": COLLECTION_NAME,
        "source_dataset": SOURCE_DATASET,
    }
 
    print(f"Raw rows: {result['raw_rows']}")
    print(f"Raw columns: {result['raw_columns']}")
    print(f"Exact duplicate rows removed: {result['exact_duplicate_rows_removed']}")
    print(f"Duplicate case IDs merged: {result['duplicate_case_ids_merged']}")
    print(f"Generated case IDs: {result['generated_case_ids']}")
    print(f"Cases stored in Parquet: {result['cases_stored_in_parquet']}")
    print(f"Chunks created: {result['chunks_created']}")
    print(f"Case ID source column: {result['case_id_source_column']}")
    print(f"Schema signature: {result['schema_signature']}")
    print(f"CSV file: {result['csv_file']}")
    print(f"Case store path: {result['case_store_path']}")
    print(f"Vector store path: {result['vector_store_path']}")
    print(f"Collection name: {result['collection_name']}")
    print(f"Source dataset: {result['source_dataset']}")
 
    return result
 
 
# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------
 
 
def main() -> None:
    """Run ingestion using the CSV path configured in .env."""
    ingest(get_default_csv_file_path())
 
 
if __name__ == "__main__":
    main()
