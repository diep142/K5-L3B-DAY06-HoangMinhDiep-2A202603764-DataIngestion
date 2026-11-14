"""Persistent PostgreSQL + pgvector storage and exact cosine retrieval."""
import json
import math
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk

TABLE_NAME = "ingestion_chunks"
FILTER_FIELDS = {"chapter", "section"}


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page TEXT NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        embedding vector({self.dimensions}) NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    SELECT format_type(attribute.atttypid, attribute.atttypmod)
                    FROM pg_attribute AS attribute
                    WHERE attribute.attrelid = %s::regclass
                      AND attribute.attname = 'embedding'
                      AND NOT attribute.attisdropped
                    """,
                    (TABLE_NAME,),
                )
                column_type = cursor.fetchone()
                expected_type = f"vector({self.dimensions})"
                if not column_type or column_type[0] != expected_type:
                    actual = column_type[0] if column_type else "missing"
                    raise ValueError(
                        f"{TABLE_NAME}.embedding is {actual}; configured model requires {expected_type}. "
                        "Use a separate database/table when changing embedding dimensions."
                    )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace one strategy snapshot atomically, removing obsolete chunks."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique within a strategy")
        rows = []
        for chunk, vector in zip(chunks, vectors):
            vector_literal = self._vector_literal(vector)
            rows.append((
                strategy,
                chunk.chunk_id,
                chunk.document_id,
                chunk.source,
                str(chunk.page),
                chunk.text,
                Jsonb(chunk.metadata),
                vector_literal,
            ))

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {TABLE_NAME} WHERE strategy = %s", (strategy,))
                if rows:
                    cursor.executemany(
                        f"""
                        INSERT INTO {TABLE_NAME}
                            (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            source = EXCLUDED.source,
                            page = EXCLUDED.page,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        rows,
                    )

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter JSONB metadata in SQL and rank by exact cosine distance."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector_literal = self._vector_literal(query_vector)
        normalized_filters = filters or {}
        unsupported = set(normalized_filters) - FILTER_FIELDS
        if unsupported:
            raise ValueError(f"Unsupported metadata filter fields: {', '.join(sorted(unsupported))}")

        sql = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1.0 - (embedding <=> %s::vector) AS score
            FROM {TABLE_NAME}
            WHERE strategy = %s
        """
        parameters: list[Any] = [vector_literal, strategy]
        if normalized_filters:
            sql += " AND metadata @> %s::jsonb"
            parameters.append(json.dumps(normalized_filters, ensure_ascii=False))
        sql += " ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s"
        parameters.extend((vector_literal, top_k))

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, parameters)
                rows = cursor.fetchall()
        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]

    def backend_info(self) -> dict[str, str]:
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_setting('server_version')")
                postgres_version = str(cursor.fetchone()[0])
                cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                extension = cursor.fetchone()
        if extension is None:
            raise RuntimeError("The pgvector extension is not installed in the configured database")
        return {"postgres_version": postgres_version, "pgvector_version": str(extension[0])}

    def _vector_literal(self, vector: list[float]) -> str:
        if len(vector) != self.dimensions:
            raise ValueError(f"Vector has {len(vector)} dimensions; expected {self.dimensions}")
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Vector values must all be finite numbers")
        return "[" + ",".join(format(value, ".9g") for value in values) + "]"
