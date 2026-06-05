from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd


class DuckDBBase:
    def __init__(self, db_path: str | Path | None = None, *, read_only: bool = False):
        self.db_path = str(db_path or ":memory:")
        self.con = duckdb.connect(self.db_path, read_only=read_only)

    @staticmethod
    def quote_identifier(name: str) -> str:
        if not isinstance(name, str) or name == "":
            raise ValueError("identifier must be a non-empty string")
        return '"' + name.replace('"', '""') + '"'

    @classmethod
    def quote_identifiers(cls, names: Iterable[str]) -> list[str]:
        return [cls.quote_identifier(name) for name in names]

    def execute(self, sql: str, params: list[Any] | tuple[Any, ...] | None = None):
        return self.con.execute(sql, params or [])

    def df(
        self,
        sql: str,
        params: list[Any] | tuple[Any, ...] | None = None,
    ) -> pd.DataFrame:
        return self.execute(sql, params).df()

    def table_exists(self, table_name: str) -> bool:
        return (
            self.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'main'
                  AND table_name = ?
                """,
                [table_name],
            ).fetchone()[0]
            > 0
        )

    def list_tables(self) -> list[str]:
        rows = self.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        ).fetchall()

        return [row[0] for row in rows]

    def list_columns(self, table_name: str) -> list[str]:
        rows = self.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'main'
              AND table_name = ?
            ORDER BY ordinal_position
            """,
            [table_name],
        ).fetchall()

        return [row[0] for row in rows]

    def column_type(self, table_name: str, column_name: str) -> str | None:
        row = self.execute(
            """
            SELECT data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
              AND table_name = ?
              AND column_name = ?
            """,
            [table_name, column_name],
        ).fetchone()

        return None if row is None else row[0]

    def drop_table(self, table_name: str, *, if_exists: bool = True) -> None:
        table = self.quote_identifier(table_name)
        exists_sql = "IF EXISTS " if if_exists else ""
        self.execute(f"DROP TABLE {exists_sql}{table}")

    @contextmanager
    def registered(self, name: str, frame: pd.DataFrame):
        self.con.register(name, frame)
        try:
            yield name
        finally:
            self.con.unregister(name)

    @contextmanager
    def transaction(self):
        self.execute("BEGIN")
        try:
            yield self
        except Exception:
            self.execute("ROLLBACK")
            raise
        else:
            self.execute("COMMIT")

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()