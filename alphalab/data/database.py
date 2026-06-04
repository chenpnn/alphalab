from __future__ import annotations

import datetime as dt
import logging
import duckdb
import pandas as pd
import re
from typing import Any, List, Optional, Union

LOGGER = logging.getLogger("update_duckdb")

class FactorDuckDB():
    def __init__(self, db_path=None):
        """
        基于 DuckDB 的因子数据库，所有表共享同一套 (date, code) 索引（仅包含在市的股票）
        
        db_path: str | Path
            DuckDB 数据库文件路径
        """
        self.db_path = str(db_path)
        self.con = duckdb.connect(self.db_path)
        self._known_tables = set()
    

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        """
        安全引用 SQL 标识符（表名、列名）
        将双引号转义为两个双引号，并用双引号包裹

        identifier: str
        """
        return f'"{identifier.replace('"', '""')}"'
    

    @staticmethod
    def _validate_identifier(identifier: str) -> bool:
        """
        验证标识符是否合法（仅包含字母、数字、下划线，可选中文）
        可根据需要调整正则表达式
        """
        return bool(re.fullmatch(r'[A-Za-z0-9_\u4e00-\u9fff]+', identifier))
    

    def _table_exists(self, table_name: str) -> bool:
        """
        检查表是否存在
        """
        if table_name in self._known_tables:
            return True
        result = self.con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0]
        exists = result > 0
        if exists:
            self._known_tables.add(table_name)
        return exists


    def _insert_universe_rows_into_table(self, table_name: str) -> None:
        """
        将 universe 中缺失的主键行插入到指定表中 (ON CONFLICT 忽略已存在)
        """
        table = self._quote_identifier(table_name)
        self.con.execute(
            f"""
            INSERT INTO {table} (date, code)
            SELECT date, code FROM universe
            ON CONFLICT (date, code) DO NOTHING
            """
        )
    
    
    def update_universe(self, frame: pd.DataFrame) -> None:
        """
        更新股票池/公共索引表 (date, code)
        通常为交易日以及相应的在市股票代码 
        """
        if not {"date", "code"}.issubset(frame.columns):
            LOGGER.error("更新 universe 失败：输入缺少 'date' 或 'code' 列")
            return
        
        df_universe = frame[["date", "code"]].copy()
        df_universe["date"] = pd.to_datetime(df_universe["date"]).dt.date

        self.con.register("_tmp_universe", df_universe)
        try:
            self.con.execute("CREATE OR REPLACE TABLE universe AS SELECT date, code FROM _tmp_universe")
        finally:
            self.con.unregister("_tmp_universe")
        
        LOGGER.info(f"更新股票池完成，共 {len(df_universe)} 条记录")

        # 可选：自动同步所有现有因子表（若需要立即生效，取消下面注释）
        # self.sync_all_tables()


    def load_universe(self, date: Union[dt.date, str]) -> List[str]:
        """
        获取指定日期的股票代码列表
        """
        res = self.con.execute(
            "SELECT code FROM universe WHERE date = ?", 
            [date]
        ).fetchall()
        return [r[0] for r in res]
    

    def sync_table_universe(self, table_name: str) -> None:
        """
        将指定表与当前 universe 同步（插入缺失的主键行）
        """
        if not self._table_exists(table_name):
            LOGGER.warning(f"表 {table_name} 不存在，无法同步")
            return
        self._insert_universe_rows_into_table(table_name)
        LOGGER.debug(f"表 {table_name} 已与 universe 同步")


    def sync_all_tables(self) -> None:
        """
        同步所有因子表与当前 universe
        """
        tables = self.list_tables()
        for tbl in tables:
            if tbl == "universe":
                continue
            self.sync_table_universe(tbl)
        LOGGER.info(f"已同步 {len(tables)-1} 个表")
    

    def create_table(self, table_name: str) -> None:
        """
        创建因子表（如果不存在），包含主键 (date, code)，并插入所有 universe 主键行
        """
        table = self._quote_identifier(table_name)
        self.con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                date DATE NOT NULL,
                code VARCHAR NOT NULL,
                PRIMARY KEY (date, code)
            )
            """
        )
        self._known_tables.add(table_name)
        # 插入当前 universe 的所有行
        self._insert_universe_rows_into_table(table_name)
        LOGGER.debug(f"表 {table_name} 已创建，主键行与 universe 同步")


    def ensure_factor_column(
        self, 
        table_name: str, 
        factor_name: str, 
        sql_type: str, 
        drop_if_type_mismatch: bool = False
    ) -> None:
        """
        确保表中存在指定因子列，若类型不匹配则根据参数决定是否重建列
        """
        if not self._validate_identifier(factor_name):
            raise ValueError(f"非法因子名: {factor_name}")

        # 确保表存在且包含主键行
        if not self._table_exists(table_name):
            self.create_table(table_name)
        else:
            # 表存在但可能缺少部分 universe 行
            self._insert_universe_rows_into_table(table_name)

        # 检查列是否存在及其类型
        current_type = self.con.execute(
            """
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = ? AND column_name = ?
            """,
            [table_name, factor_name],
        ).fetchone()

        if current_type:
            if current_type[0] != sql_type:
                msg = f"列 {table_name}.{factor_name} 类型不匹配: 当前 {current_type[0]}, 期望 {sql_type}"
                if drop_if_type_mismatch:
                    LOGGER.warning(msg + "，将删除原列并重建")
                    self.drop_factor_column(table_name, factor_name)
                else:
                    LOGGER.warning(msg + "，未执行任何操作")
                    return
            else:
                LOGGER.debug(f"列 {table_name}.{factor_name} 已存在且类型匹配")
                return

        # 添加新列
        table = self._quote_identifier(table_name)
        col = self._quote_identifier(factor_name)
        self.con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {sql_type}")


    def drop_table(self, table_name: str, if_exists: str = "fail") -> None:
        """
        删除整张因子表

        if_exists : str, {'fail', 'ignore'}
            - 'fail'   : 表不存在时抛出异常
            - 'ignore' : 表不存在时不作任何操作
        """
        if not self._validate_identifier(table_name):
            raise ValueError(f"非法表名: {table_name}")

        exists = self._table_exists(table_name)
        if not exists and if_exists == "fail":
            raise RuntimeError(f"表 {table_name} 不存在，无法删除")

        if exists:
            table = self._quote_identifier(table_name)
            self.con.execute(f"DROP TABLE {table}")
            self._known_tables.discard(table_name)
            LOGGER.info(f"已删除表 {table_name}")
        else:
            LOGGER.debug(f"表 {table_name} 不存在，忽略")


    def drop_factor_column(self, table_name: str, factor_name: str) -> None:
        """
        删除因子列
        """
        if not self._validate_identifier(factor_name):
            raise ValueError(f"非法因子名: {factor_name}")
        table = self._quote_identifier(table_name)
        col = self._quote_identifier(factor_name)
        self.con.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
        LOGGER.debug(f"已删除列 {table_name}.{factor_name}")
    

    def rename_factor_column(self, table_name: str, old_name: str, new_name: str) -> None:
        """
        重命名因子列
        """
        for name in (old_name, new_name):
            if not self._validate_identifier(name):
                raise ValueError(f"非法因子名: {name}")
        table = self._quote_identifier(table_name)
        old = self._quote_identifier(old_name)
        new = self._quote_identifier(new_name)
        self.con.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
        LOGGER.debug(f"列 {table_name}.{old_name} 重命名为 {new_name}")
    

    def clear_factor(self, table_name: str, factor_name: str) -> None:
        """
        将因子列的所有值设为 NULL
        """
        if not self._validate_identifier(factor_name):
            raise ValueError(f"非法因子名: {factor_name}")
        table = self._quote_identifier(table_name)
        col = self._quote_identifier(factor_name)
        self.con.execute(f'UPDATE {table} SET {col} = NULL')
        LOGGER.debug(f"已清空列 {table_name}.{factor_name}")
    

    def list_tables(self) -> List[str]:
        """
        返回所有用户表名（不包括系统表）
        """
        rows = self.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        tables = [r[0] for r in rows]
        self._known_tables.update(tables)
        return tables
    

    def list_factors(self, table_name: str) -> List[str]:
        """获取因子表中除 date, code 外的所有列名"""
        if not self._table_exists(table_name):
            return []
        rows = self.con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position
            """,
            [table_name],
        ).fetchall()
        return [r[0] for r in rows if r[0] not in ("date", "code")]


    def write_factor_frame(self, table_name, factor_name, frame, sql_type=None, replace=False):
        """
        将因子数据写入数据库，支持宽表和长表两种格式。

        宽表格式: index 为日期, columns 为股票代码，值为因子值
        长表格式：必须包含 'date', 'code', factor_name 三列

        Parameters
        ----------
        table_name : str
            目标表名
        factor_name : str
            因子列名
        frame : pd.DataFrame
            因子数据（宽表或长表）
        """
        if frame.empty:
            return
        
        # 确保目标表存在且包含所有 universe 主键行
        self.create_table(table_name)

        # 检测输入格式：若存在 'date' 和 'code' 列则视为长表，否则视为宽表
        if {"date", "code"}.issubset(frame.columns):
            df_long = frame[["date", "code", factor_name]].copy()
        else:
            # 宽表格式：index 为日期，columns 为股票代码
            df_long = frame.stack().reset_index()
            df_long.columns = ["date", "code", factor_name]

        df_long["date"] = pd.to_datetime(df_long["date"]).dt.date
        df_long = df_long[df_long[factor_name].notna()]
        if df_long.empty:
            return
        
        # 自动推断 SQL 类型（如果未提供）
        if sql_type is None:
            sample = df_long[factor_name].dropna()
            if not sample.empty:
                val = sample.iloc[0]
                if isinstance(val, str) or pd.api.types.is_string_dtype(sample):
                    sql_type = "VARCHAR"
                else:
                    sql_type = "DOUBLE"
            else:
                sql_type = "DOUBLE"

        # 若需要替换列，先删除已存在的列（如果存在）
        if replace and self._table_exists(table_name):
            current_type = self.con.execute(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_name = ? AND column_name = ?
                """,
                [table_name, factor_name],
            ).fetchone()
            if current_type:
                self.drop_factor_column(table_name, factor_name)

        # 确保因子列存在（若类型不匹配会重建列）
        self.ensure_factor_column(table_name, factor_name, sql_type)

        # 注册临时表并写入
        self.con.register("_tmp_factor", df_long)
        table = self._quote_identifier(table_name)
        col = self._quote_identifier(factor_name)
        try:
            self.con.execute(
                f"""
                INSERT INTO {table} (date, code, {col})
                SELECT date, code, {col}
                FROM _tmp_factor
                ON CONFLICT (date, code) DO UPDATE SET {col} = EXCLUDED.{col}
                """
            )
            LOGGER.debug(f"因子 {factor_name} 写入完成，共 {len(df_long)} 条有效记录")
        except Exception as e:
            LOGGER.error(f"写入因子 {factor_name} 失败: {e}")
            raise
        finally:
            self.con.unregister("_tmp_factor")


    def query(
        self,
        table_name: str,
        factors: list[str] | str | None = None,
        codes: list[str] | str | None = None,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """查询指定表的数据，自动过滤不在市的股票"""
        factor_cols = [factors] if isinstance(factors, str) else factors
        columns = ["date", "code"] + (factor_cols or self.factor_columns(table_name))
        clauses = []
        params = []
        if codes:
            code_list = [codes] if isinstance(codes, str) else codes
            clauses.append(f"code IN ({', '.join(['?'] * len(code_list))})")
            params.extend(code_list)
        if start_date:
            clauses.append("date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("date <= ?")
            params.append(end_date)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = f"LIMIT {int(limit)}" if limit else ""
        return self.con.execute(
            f"""
            SELECT {", ".join(f'"{col}"' for col in columns)}
            FROM "{table_name}"
            {where_sql}
            ORDER BY date, code
            {limit_sql}
            """,
            params,
        ).df()


    def begin(self) -> None:
        """开始事务"""
        self.con.execute("BEGIN")


    def commit(self) -> None:
        """提交事务"""
        self.con.execute("COMMIT")


    def rollback(self) -> None:
        """回滚事务"""
        self.con.execute("ROLLBACK")


    def close(self) -> None:
        self.con.close()


    def __enter__(self) -> None:
        return self


    def __exit__(self, *_) -> None:
        self.close()


if __name__ == "__main__":
    with FactorDuckDB() as db:
        print(db._quote_identifier("code"))
        print(db._quote_identifier('code'))
        print(db._quote_identifier(""))
        print(db._quote_identifier(''))
        