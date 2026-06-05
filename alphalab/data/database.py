import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from .base import DuckDBBase


WriteMode = Literal["replace_table", "upsert", "replace_columns"]


class FactorDuckDB(DuckDBBase):
    """DuckDB-backed factor database.

    ``Universe`` is the canonical ``(date, code)`` key table. Factor tables
    should contain exactly the same ``(date, code)`` rows as ``Universe``.
    """

    key_cols = ("date", "code")
    universe_table = "Universe"
    dictionary_table = "factor_dictionary"

    def __init__(self, db_path: str | Path | None = None, *, read_only: bool = False):
        """Open a DuckDB factor database."""
        super().__init__(db_path, read_only=read_only)

    @staticmethod
    def _normalize_date_code(frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with normalized ``date`` and ``code`` columns."""
        df = frame.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["code"] = df["code"].astype(str)
        return df

    @staticmethod
    def infer_sql_type(series: pd.Series) -> str:
        """Infer a DuckDB SQL type from a pandas Series."""
        sample = series.dropna()

        if sample.empty:
            return "DOUBLE"
        if pd.api.types.is_bool_dtype(sample):
            return "BOOLEAN"
        if pd.api.types.is_integer_dtype(sample):
            return "BIGINT"
        if pd.api.types.is_float_dtype(sample):
            return "DOUBLE"
        if pd.api.types.is_datetime64_any_dtype(sample):
            return "TIMESTAMP"

        return "VARCHAR"

    def create_universe_table(self) -> None:
        """Create ``Universe`` if it does not exist."""
        table = self.quote_identifier(self.universe_table)
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                date DATE NOT NULL,
                code VARCHAR NOT NULL,
                PRIMARY KEY (date, code)
            )
            """
        )

    def update_universe(
        self,
        frame: pd.DataFrame,
        *,
        replace: bool = True,
        sync_factor_tables: bool = True,
    ) -> None:
        """Update ``Universe``.

        Parameters
        ----------
        frame:
            DataFrame containing ``date`` and ``code`` columns.
        replace:
            If true, rebuild ``Universe`` from ``frame``. If false, append
            missing rows.
        sync_factor_tables:
            If true, align existing factor tables to the updated ``Universe``.
        """
        if not {"date", "code"}.issubset(frame.columns):
            raise ValueError("frame must contain 'date' and 'code' columns")

        df = self._normalize_date_code(frame[["date", "code"]])
        table = self.quote_identifier(self.universe_table)

        with self.registered("_tmp_Universe", df):
            if replace:
                self.execute(f"DROP TABLE IF EXISTS {table}")
                self.execute(
                    f"""
                    CREATE TABLE {table} (
                        date DATE NOT NULL,
                        code VARCHAR NOT NULL,
                        PRIMARY KEY (date, code)
                    )
                    """
                )
                self.execute(
                    f"""
                    INSERT INTO {table} (date, code)
                    SELECT DISTINCT date, code
                    FROM _tmp_Universe
                    """
                )
            else:
                self.create_universe_table()
                self.execute(
                    f"""
                    INSERT INTO {table} (date, code)
                    SELECT DISTINCT date, code
                    FROM _tmp_Universe
                    ON CONFLICT (date, code) DO NOTHING
                    """
                )

        if sync_factor_tables:
            self.sync_all_factor_tables()

    def load_universe(self, date: str | datetime.date | pd.Timestamp) -> list[str]:
        """Return all codes in ``Universe`` on a given date."""
        date = pd.to_datetime(date).date()
        table = self.quote_identifier(self.universe_table)

        rows = self.execute(
            f"""
            SELECT code
            FROM {table}
            WHERE date = ?
            ORDER BY code
            """,
            [date],
        ).fetchall()

        return [row[0] for row in rows]

    def create_factor_table(
        self,
        table_name: str,
        *,
        sync_universe: bool = True,
    ) -> None:
        """Create a factor table and optionally align it to ``Universe``."""
        table = self.quote_identifier(table_name)

        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                date DATE NOT NULL,
                code VARCHAR NOT NULL,
                PRIMARY KEY (date, code)
            )
            """
        )

        if sync_universe:
            self.sync_table_universe(table_name)

    def sync_table_universe(self, table_name: str) -> None:
        """Make a factor table's ``(date, code)`` rows exactly match ``Universe``.

        Rows outside ``Universe`` are deleted. Missing ``Universe`` rows are
        inserted with NULL factor values.
        """
        if table_name == self.universe_table:
            return
        if not self.table_exists(self.universe_table):
            raise RuntimeError("Universe table does not exist")

        table = self.quote_identifier(table_name)
        universe_table = self.quote_identifier(self.universe_table)

        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                date DATE NOT NULL,
                code VARCHAR NOT NULL,
                PRIMARY KEY (date, code)
            )
            """
        )
        self.execute(
            f"""
            DELETE FROM {table} AS t
            WHERE NOT EXISTS (
                SELECT 1
                FROM {universe_table} AS u
                WHERE u.date = t.date
                  AND u.code = t.code
            )
            """
        )
        self.execute(
            f"""
            INSERT INTO {table} (date, code)
            SELECT date, code
            FROM {universe_table}
            ON CONFLICT (date, code) DO NOTHING
            """
        )

    def sync_all_factor_tables(self) -> None:
        """Align every factor table to ``Universe``."""
        if not self.table_exists(self.universe_table):
            return

        skip_tables = {self.universe_table, self.dictionary_table}
        for table_name in self.list_tables():
            if table_name not in skip_tables:
                self.sync_table_universe(table_name)

    def list_factors(self, table_name: str) -> list[str]:
        """List factor columns in a factor table."""
        if not self.table_exists(table_name):
            return []

        return [
            col
            for col in self.list_columns(table_name)
            if col not in self.key_cols
        ]

    def ensure_columns(
        self,
        table_name: str,
        dtypes: dict[str, str],
        *,
        drop_if_type_mismatch: bool = False,
        sync_universe: bool = True,
    ) -> None:
        """Ensure factor columns exist with the expected SQL types."""
        self.create_factor_table(table_name, sync_universe=sync_universe)
        table = self.quote_identifier(table_name)

        for col_name, sql_type in dtypes.items():
            current_type = self.column_type(table_name, col_name)
            col = self.quote_identifier(col_name)

            if current_type is None:
                self.execute(f"ALTER TABLE {table} ADD COLUMN {col} {sql_type}")
                continue

            if current_type != sql_type:
                if not drop_if_type_mismatch:
                    raise TypeError(
                        f"{table_name}.{col_name} type mismatch: "
                        f"current={current_type}, expected={sql_type}"
                    )
                self.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
                self.execute(f"ALTER TABLE {table} ADD COLUMN {col} {sql_type}")

    def drop_factor(
        self,
        table_name: str,
        factor_name: str,
        *,
        if_exists: bool = True,
    ) -> None:
        """Drop a factor column."""
        table = self.quote_identifier(table_name)
        col = self.quote_identifier(factor_name)
        exists_sql = "IF EXISTS " if if_exists else ""
        self.execute(f"ALTER TABLE {table} DROP COLUMN {exists_sql}{col}")

    def rename_factor(self, table_name: str, old_name: str, new_name: str) -> None:
        """Rename a factor column."""
        table = self.quote_identifier(table_name)
        old = self.quote_identifier(old_name)
        new = self.quote_identifier(new_name)
        self.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")

    def clear_factor(
        self,
        table_name: str,
        factor_name: str,
        *,
        start_date: str | datetime.date | None = None,
        end_date: str | datetime.date | None = None,
    ) -> None:
        """Set a factor column to NULL, optionally within a date range."""
        table = self.quote_identifier(table_name)
        col = self.quote_identifier(factor_name)

        clauses = []
        params = []
        if start_date is not None:
            clauses.append("date >= ?")
            params.append(pd.to_datetime(start_date).date())
        if end_date is not None:
            clauses.append("date <= ?")
            params.append(pd.to_datetime(end_date).date())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        self.execute(f"UPDATE {table} SET {col} = NULL {where_sql}", params)

    def _coerce_factor_frame(
        self,
        frame: pd.DataFrame | pd.Series,
        *,
        factor_name: str | None,
    ) -> pd.DataFrame:
        """Normalize supported factor inputs to long form."""
        if isinstance(frame, pd.Series):
            name = factor_name or frame.name
            if name is None:
                raise ValueError("factor_name is required for an unnamed Series")

            df = frame.rename(name).reset_index()
            if df.shape[1] != 3:
                raise ValueError("Series input must have a two-level index: date, code")
            df.columns = ["date", "code", name]
            return df

        if {"date", "code"}.issubset(frame.columns):
            df = frame.copy()
            if factor_name is not None:
                if factor_name not in df.columns:
                    raise ValueError(f"frame does not contain {factor_name!r}")
                df = df[["date", "code", factor_name]]
            return df

        if factor_name is None:
            raise ValueError("factor_name is required for a wide DataFrame")

        return (
            frame.copy()
            .rename_axis(index="date")
            .reset_index()
            .melt(id_vars="date", var_name="code", value_name=factor_name)
        )

    def write_factor_table(
        self,
        table_name: str,
        frame: pd.DataFrame | pd.Series,
        *,
        factor_name: str | None = None,
        dtypes: dict[str, str] | None = None,
        mode: WriteMode = "replace_table",
        drop_if_type_mismatch: bool = False,
        sync_universe: bool = True,
    ) -> None:
        """Write factor data to a factor table.

        Supported input:
        - long DataFrame: ``date``, ``code``, factor columns
        - wide DataFrame: index is date, columns are codes, one factor
        - Series with two-level index: date and code

        With ``sync_universe=True``, writes are restricted to ``Universe`` and
        the target table is aligned to ``Universe`` after writing.
        """
        if frame.empty:
            return

        df = self._coerce_factor_frame(frame, factor_name=factor_name)
        if not {"date", "code"}.issubset(df.columns):
            raise ValueError("input must contain or resolve to date/code columns")

        df = self._normalize_date_code(df)
        factor_cols = [col for col in df.columns if col not in self.key_cols]
        if not factor_cols:
            raise ValueError("no factor columns found")

        df = df.drop_duplicates(["date", "code"], keep="last")
        col_types = {
            col: (dtypes or {}).get(col, self.infer_sql_type(df[col]))
            for col in factor_cols
        }
        df = df[["date", "code", *factor_cols]]

        if mode == "replace_table":
            self._replace_factor_table(
                table_name,
                df,
                col_types,
                sync_universe=sync_universe,
            )
            return

        self.create_factor_table(table_name, sync_universe=sync_universe)

        if mode == "replace_columns":
            for col in factor_cols:
                if self.column_type(table_name, col) is not None:
                    self.drop_factor(table_name, col, if_exists=True)

        self.ensure_columns(
            table_name,
            col_types,
            drop_if_type_mismatch=drop_if_type_mismatch,
            sync_universe=sync_universe,
        )
        self._upsert_factor_rows(
            table_name,
            df,
            factor_cols,
            sync_universe=sync_universe,
        )

    def _replace_factor_table(
        self,
        table_name: str,
        frame: pd.DataFrame,
        dtypes: dict[str, str],
        *,
        sync_universe: bool = True,
    ) -> None:
        """Drop and rebuild a factor table.

        If ``sync_universe`` is true, the rebuilt table uses ``Universe`` as
        the driving table. Input keys outside ``Universe`` are ignored and
        missing factor values become NULL.
        """
        table = self.quote_identifier(table_name)
        factor_cols = [col for col in frame.columns if col not in self.key_cols]
        quoted_factor_cols = self.quote_identifiers(factor_cols)
        column_defs = [
            "date DATE NOT NULL",
            "code VARCHAR NOT NULL",
            *[
                f"{self.quote_identifier(col)} {dtypes[col]}"
                for col in factor_cols
            ],
            "PRIMARY KEY (date, code)",
        ]

        with self.registered("_tmp_factors", frame):
            self.execute(f"DROP TABLE IF EXISTS {table}")
            self.execute(
                f"""
                CREATE TABLE {table} (
                    {", ".join(column_defs)}
                )
                """
            )

            if sync_universe:
                if not self.table_exists(self.universe_table):
                    raise RuntimeError("Universe table does not exist")

                universe_table = self.quote_identifier(self.universe_table)
                insert_cols = ["date", "code", *quoted_factor_cols]
                select_cols = [
                    "u.date",
                    "u.code",
                    *[f"src.{col}" for col in quoted_factor_cols],
                ]
                self.execute(
                    f"""
                    INSERT INTO {table} ({", ".join(insert_cols)})
                    SELECT {", ".join(select_cols)}
                    FROM {universe_table} AS u
                    LEFT JOIN _tmp_factors AS src
                        ON src.date = u.date
                       AND src.code = u.code
                    """
                )
            else:
                insert_cols = ["date", "code", *quoted_factor_cols]
                cols_sql = ", ".join(insert_cols)
                self.execute(
                    f"""
                    INSERT INTO {table} ({cols_sql})
                    SELECT {cols_sql}
                    FROM _tmp_factors
                    """
                )

    def _upsert_factor_rows(
        self,
        table_name: str,
        frame: pd.DataFrame,
        factor_cols: list[str],
        *,
        sync_universe: bool = True,
    ) -> None:
        """Upsert factor values, optionally filtering source rows by ``Universe``."""
        table = self.quote_identifier(table_name)
        quoted_factor_cols = self.quote_identifiers(factor_cols)
        insert_cols = ["date", "code", *quoted_factor_cols]
        update_sql = ", ".join(
            f"{col} = EXCLUDED.{col}"
            for col in quoted_factor_cols
        )

        with self.registered("_tmp_factors", frame):
            if sync_universe:
                if not self.table_exists(self.universe_table):
                    raise RuntimeError("Universe table does not exist")

                universe_table = self.quote_identifier(self.universe_table)
                select_cols = ", ".join(f"src.{col}" for col in insert_cols)
                source_sql = f"""
                    SELECT {select_cols}
                    FROM _tmp_factors AS src
                    INNER JOIN {universe_table} AS u
                        ON src.date = u.date
                       AND src.code = u.code
                """
            else:
                cols_sql = ", ".join(insert_cols)
                source_sql = f"""
                    SELECT {cols_sql}
                    FROM _tmp_factors
                """

            self.execute(
                f"""
                INSERT INTO {table} ({", ".join(insert_cols)})
                {source_sql}
                ON CONFLICT (date, code) DO UPDATE SET {update_sql}
                """
            )

        if sync_universe:
            self.sync_table_universe(table_name)

    def write_factor(
        self,
        table_name: str,
        factor_name: str,
        frame: pd.DataFrame | pd.Series,
        **kwargs,
    ) -> None:
        """Compatibility wrapper for writing one factor."""
        self.write_factor_table(
            table_name,
            frame,
            factor_name=factor_name,
            **kwargs,
        )

    def write_factors(self, table_name: str, frame: pd.DataFrame, **kwargs) -> None:
        """Compatibility wrapper for writing one or more factors."""
        self.write_factor_table(table_name, frame, **kwargs)

    def query(
        self,
        table_name: str,
        factors: str | list[str] | None = None,
        codes: str | list[str] | None = None,
        start_date: str | datetime.date | None = None,
        end_date: str | datetime.date | None = None,
        *,
        drop_all_null: bool = False,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Query factor data in long form."""
        if not self.table_exists(table_name):
            raise RuntimeError(f"table does not exist: {table_name}")

        factor_cols = [factors] if isinstance(factors, str) else factors
        factor_cols = factor_cols or self.list_factors(table_name)
        columns = ["date", "code", *factor_cols]
        select_sql = ", ".join(self.quote_identifiers(columns))

        clauses = []
        params = []
        if codes is not None:
            code_list = [codes] if isinstance(codes, str) else list(codes)
            clauses.append(f"code IN ({', '.join(['?'] * len(code_list))})")
            params.extend(code_list)
        if start_date is not None:
            clauses.append("date >= ?")
            params.append(pd.to_datetime(start_date).date())
        if end_date is not None:
            clauses.append("date <= ?")
            params.append(pd.to_datetime(end_date).date())
        if drop_all_null and factor_cols:
            null_clause = " OR ".join(
                f"{self.quote_identifier(col)} IS NOT NULL"
                for col in factor_cols
            )
            clauses.append(f"({null_clause})")

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
        table = self.quote_identifier(table_name)

        return self.df(
            f"""
            SELECT {select_sql}
            FROM {table}
            {where_sql}
            ORDER BY date, code
            {limit_sql}
            """,
            params,
        )

    def query_wide(
        self,
        table_name: str,
        factor: str,
        *,
        codes: str | list[str] | None = None,
        start_date: str | datetime.date | None = None,
        end_date: str | datetime.date | None = None,
    ) -> pd.DataFrame:
        """Query one factor as a date-by-code matrix."""
        df = self.query(
            table_name,
            factors=factor,
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            drop_all_null=True,
        )

        if df.empty:
            return pd.DataFrame()

        return (
            df.pivot(index="date", columns="code", values=factor)
            .sort_index()
            .sort_index(axis=1)
        )

    def delete_rows(
        self,
        table_name: str,
        *,
        codes: str | list[str] | None = None,
        start_date: str | datetime.date | None = None,
        end_date: str | datetime.date | None = None,
    ) -> None:
        """Delete rows from a factor table. At least one filter is required."""
        clauses = []
        params = []
        if codes is not None:
            code_list = [codes] if isinstance(codes, str) else list(codes)
            clauses.append(f"code IN ({', '.join(['?'] * len(code_list))})")
            params.extend(code_list)
        if start_date is not None:
            clauses.append("date >= ?")
            params.append(pd.to_datetime(start_date).date())
        if end_date is not None:
            clauses.append("date <= ?")
            params.append(pd.to_datetime(end_date).date())
        if not clauses:
            raise ValueError("delete_rows requires at least one filter")

        table = self.quote_identifier(table_name)
        self.execute(
            f"""
            DELETE FROM {table}
            WHERE {' AND '.join(clauses)}
            """,
            params,
        )

        if self.table_exists(self.universe_table):
            self.sync_table_universe(table_name)

    def describe_table(self, table_name: str) -> pd.DataFrame:
        """Return column names, SQL types, and nullability."""
        return self.df(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'main'
              AND table_name = ?
            ORDER BY ordinal_position
            """,
            [table_name],
        )

    def table_stats(self, table_name: str) -> pd.DataFrame:
        """Return date range and row/code/date counts."""
        table = self.quote_identifier(table_name)
        return self.df(
            f"""
            SELECT
                MIN(date) AS start_date,
                MAX(date) AS end_date,
                COUNT(*) AS rows,
                COUNT(DISTINCT date) AS dates,
                COUNT(DISTINCT code) AS codes
            FROM {table}
            """
        )

    def universe_match_stats(self, table_name: str) -> pd.DataFrame:
        """Return key mismatch counts between a factor table and ``Universe``."""
        if not self.table_exists(self.universe_table):
            raise RuntimeError("Universe table does not exist")
        if not self.table_exists(table_name):
            raise RuntimeError(f"table does not exist: {table_name}")

        table = self.quote_identifier(table_name)
        universe_table = self.quote_identifier(self.universe_table)

        return self.df(
            f"""
            SELECT
                (
                    SELECT COUNT(*)
                    FROM {universe_table} AS u
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM {table} AS t
                        WHERE t.date = u.date
                          AND t.code = u.code
                    )
                ) AS missing_in_table,
                (
                    SELECT COUNT(*)
                    FROM {table} AS t
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM {universe_table} AS u
                        WHERE u.date = t.date
                          AND u.code = t.code
                    )
                ) AS extra_in_table
            """
        )


if __name__ == "__main__":
    with FactorDuckDB() as db:
        pass
