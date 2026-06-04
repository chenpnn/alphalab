from __future__ import annotations

import datetime as dt
import pickle
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd


def fetch_h5file(h5_path, ids=None, dts=None):
    with h5py.File(h5_path, "r") as DataFile:
        DataType = DataFile.attrs["DataType"]
        DateTimes = DataFile["DateTime"][...]
        if h5py.version.version >= "3.0.0":
            IDs = DataFile["ID"].asstr(encoding="utf-8")[...]
        else:
            IDs = DataFile["ID"][...]
        
        if dts is None:
            if ids is None:
                if (h5py.version.version >= "3.0.0") and (DataType == "string"):
                    Rslt = pd.DataFrame(
                        DataFile["Data"].asstr(encoding="utf-8")[...], 
                        index=DateTimes, 
                        columns=IDs
                    ).sort_index(axis=1)
                else:
                    Rslt = pd.DataFrame(
                        DataFile["Data"][...], 
                        index=DateTimes, 
                        columns=IDs
                    ).sort_index(axis=1)
            
            elif set(ids).isdisjoint(IDs):
                Rslt = pd.DataFrame(index=DateTimes, columns=ids)
            
            else:
                if (h5py.version.version >= "3.0.0") and (DataType == "string"):
                    Rslt = pd.DataFrame(
                        DataFile["Data"].asstr(encoding="utf-8")[...], 
                        index=DateTimes, 
                        columns=IDs
                    ).reindex(columns=ids)
                else:
                    Rslt = pd.DataFrame(
                        DataFile["Data"][...], 
                        index=DateTimes, 
                        columns=IDs
                    ).reindex(columns=ids)
            
            Rslt.index = [dt.datetime.fromtimestamp(itms) for itms in Rslt.index]
        
        elif (ids is not None) and set(ids).isdisjoint(IDs):
            Rslt = pd.DataFrame(index=dts, columns=ids)
        
        else:
            dts = [x.to_pydatetime().timestamp() if isinstance(x, pd.Timestamp) else x.timestamp() for x in dts]
            DateTimes = pd.Series(np.arange(0, DateTimes.shape[0]), index=DateTimes, dtype=int)
            DateTimes = DateTimes[DateTimes.index.intersection(dts)]
            nDT = DateTimes.shape[0]

            if nDT == 0:
                if ids is None: 
                    Rslt = pd.DataFrame(index=dts, columns=IDs).sort_index(axis=1)
                else: 
                    Rslt = pd.DataFrame(index=dts, columns=ids)
            elif nDT < 1000:
                DateTimes = DateTimes.sort_values()
                Mask = DateTimes.tolist()
                DateTimes = DateTimes.index.values
                if ids is None:
                    if (h5py.version.version >= "3.0.0") and (DataType == "string"):
                        Rslt = pd.DataFrame(
                            DataFile["Data"].asstr(encoding="utf-8")[Mask, :], 
                            index=DateTimes, 
                            columns=IDs
                        ).reindex(index=dts).sort_index(axis=1)
                    else:
                        Rslt = pd.DataFrame(
                            DataFile["Data"][Mask, :], 
                            index=DateTimes, 
                            columns=IDs
                        ).reindex(index=dts).sort_index(axis=1)
                else:
                    IDRuler = pd.Series(np.arange(0,IDs.shape[0]), index=IDs)
                    IDRuler = IDRuler.reindex(index=ids)
                    StartInd, EndInd = int(IDRuler.min()), int(IDRuler.max())
                    if (h5py.version.version >= "3.0.0") and (DataType == "string"):
                        Rslt = pd.DataFrame(
                            DataFile["Data"].asstr(encoding="utf-8")[Mask, StartInd:EndInd+1], 
                            index=DateTimes, 
                            columns=IDs[StartInd: EndInd+1]
                        ).reindex(index=dts, columns=ids)
                    else:
                        Rslt = pd.DataFrame(
                            DataFile["Data"][Mask, StartInd:EndInd+1], 
                            index=DateTimes, 
                            columns=IDs[StartInd:EndInd+1]
                        ).reindex(index=dts, columns=ids)
            else:
                if (h5py.version.version >= "3.0.0") and (DataType == "string"):
                    Rslt = pd.DataFrame(
                        DataFile["Data"].asstr(encoding="utf-8")[...], 
                        index=DataFile["DateTime"][...], 
                        columns=IDs
                    ).reindex(index=dts)
                else:
                    Rslt = pd.DataFrame(
                        DataFile["Data"][...], 
                        index=DataFile["DateTime"][...], 
                        columns=IDs
                    ).reindex(index=dts)
                if ids is not None: 
                    Rslt = Rslt.reindex(columns=ids)
                else: 
                    Rslt.sort_index(axis=1, inplace=True)
            Rslt.index = [dt.datetime.fromtimestamp(itms) for itms in Rslt.index]
        
    if DataType == "string":
        Rslt = Rslt.where(pd.notnull(Rslt), None)
        Rslt = Rslt.where(Rslt != "", None)
    elif DataType == "object":
        Rslt = Rslt.map(lambda x: pickle.loads(bytes(x)) if isinstance(x, np.ndarray) and (x.shape[0] > 0) else None)
    return Rslt.sort_index(axis=0)


@lru_cache(maxsize=1000)
def decode_pickle(data_bytes: bytes):
    return pickle.loads(data_bytes)


def map_pickle(x):
    if isinstance(x, np.ndarray) and x.size > 0:
        return decode_pickle(x.tobytes())
    return None


class HDFData:
    def __init__(self, h5_path):
        h5_path = str(h5_path)
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.data_type = f.attrs["DataType"]
            self.timestamps = f["DateTime"][...]  # 数值时间戳（秒）
            if h5py.version.version >= "3.0.0":
                self.all_ids = f["ID"].asstr(encoding="utf-8")[...]
            else:
                self.all_ids = f["ID"][...]

        self.dates = [dt.datetime.fromtimestamp(ts) for ts in self.timestamps]
        self.codes = sorted(self.all_ids)

        # 预建立快速索引映射
        self.ts_to_idx = {ts: i for i, ts in enumerate(self.timestamps)}
        self.code_to_idx = {code: i for i, code in enumerate(self.all_ids)}
        self.h5py_ge_3 = h5py.version.version >= "3.0.0"

    def fetch_df(self, ids=None, dts=None):
        """从 HDF5 文件中读取数据"""
        if dts is not None:
            # 将输入的时间对象转换为数值时间戳（秒）
            dts_vals = []
            for x in dts:
                if isinstance(x, pd.Timestamp):
                    dts_vals.append(x.to_pydatetime().timestamp())
                else:
                    dts_vals.append(x.timestamp())
        else:
            dts_vals = None

        if ids is not None:
            # 快速判断是否存在交集
            common = set(ids) & set(self.all_ids)
            if not common:
                if dts_vals is None:
                    idx = self.dates
                else:
                    idx = [dt.datetime.fromtimestamp(v) for v in dts_vals]
                empty_df = pd.DataFrame(index=idx, columns=ids)
                # 后处理（字符串/对象）对空表无影响，但保持流程一致性
                return self._postprocess(empty_df)

        if dts_vals is None:
            row_indices = slice(None)
            row_timestamps = self.timestamps
            need_row_reindex = False
        else:
            # 查找匹配的行位置
            matched_positions = []
            matched_timestamps = []
            for v in dts_vals:
                pos = self.ts_to_idx.get(v)
                if pos is not None:
                    matched_positions.append(pos)
                    matched_timestamps.append(v)
            if not matched_positions:
                # 无匹配行 → 返回空 DataFrame
                idx = [dt.datetime.fromtimestamp(v) for v in dts_vals]
                empty_df = pd.DataFrame(index=idx, columns=ids if ids is not None else self.codes)
                if ids is None:
                    empty_df.sort_index(axis=1, inplace=True)
                return self._postprocess(empty_df)

            # 保持与原代码一致：按时间戳排序并转换为列表（用于 HDF5 切片）
            order = np.argsort(matched_timestamps)
            row_indices = np.array(matched_positions)[order]
            row_timestamps = np.array(matched_timestamps)[order]
            need_row_reindex = True

        if ids is None:
            col_slice = slice(None)
            col_labels = self.all_ids
            need_col_reindex = False
            need_col_sort = True
        else:
            # 获取所需列的位置，并找出最小最大索引（连续读取）
            needed_positions = [self.code_to_idx[id_] for id_ in ids if id_ in self.code_to_idx]
            min_pos = min(needed_positions)
            max_pos = max(needed_positions)
            col_slice = slice(min_pos, max_pos + 1)
            col_labels = self.all_ids[min_pos:max_pos+1]
            need_col_reindex = True
            need_col_sort = False

        with h5py.File(self.h5_path, "r") as f:
            data_ds = f["Data"]
            if self.h5py_ge_3 and self.data_type == "string":
                data_subset = data_ds.asstr(encoding="utf-8")[row_indices, col_slice]
            else:
                data_subset = data_ds[row_indices, col_slice]

        df = pd.DataFrame(data_subset, index=row_timestamps, columns=col_labels)

        # 必要时重排行/列索引
        if need_row_reindex:
            df = df.reindex(index=dts_vals)
        if need_col_reindex:
            df = df.reindex(columns=ids)
        if need_col_sort:
            df.sort_index(axis=1, inplace=True)

        df.index = [dt.datetime.fromtimestamp(ts) for ts in df.index]
        df = self._postprocess(df)
        df.sort_index(axis=0, inplace=True)
        return df

    def _postprocess(self, df):
        """根据 DataType 进行后处理（字符串替换 / pickle 反序列化）"""
        if self.data_type == "string":
            df = df.where(pd.notnull(df), None)
            df = df.where(df != "", None)
        elif self.data_type == "object":
            df = df.map(map_pickle)
        return df

if __name__ == "__main__":
    root = "D:/CPResearch/market_data/XYQuantData/HDF5Data/ElementaryFactor"
    factor = "中信行业"
    f = HDFData(f"{root}/{factor}.hdf5")

    dts = [x for x in f.dates if x >= dt.datetime(2026,1,1)]

    df = f.fetch_df(dts=dts)
    display(df)

    # df_long= df.reset_index().melt(
    #     id_vars="index", var_name="code", value_name="是否在市"
    # )
    # df_long = df_long.rename(columns={"index": "date"})
    # df_long = df_long[df_long["是否在市"] == 1]

    # df_long.set_index(["date", "code"])
    # df_long.groupby("date")["code"].count()



