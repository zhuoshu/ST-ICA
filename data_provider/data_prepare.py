
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import os
import numpy as np
import pandas as pd


import torch
from torch.utils.data import Dataset, DataLoader


# data_root_path = 'none'
data_root_path = './raw_data'
if data_root_path == 'none':
    raise ValueError('Please replace `root_path` into the ABSOLUTE path of `raw_data` folder!')

file_path_dict = {
                'EPeMS':f'{data_root_path}/EPeMS/gla_pemsd7.csv',
                'UrbanEV_s':f'{data_root_path}/UrbanEV_s/volume-11kw.csv',
                'GBA':f'{data_root_path}/GBA/gba_his_2019.h5',
                'AQI':f'{data_root_path}/AQI/AQI_stages.csv',
                'METR-LA':f'{data_root_path}/METR-LA/metr-la.h5'
                }
meta_info_dict= {        
        'EPeMS':['2017-01-01 00:00:00','5m'],
        'UrbanEV_s':['2022-09-01 00:00:00','1h'],
        'GBA':['2019-01-01 00:00:00','15m'],
        'AQI':['2014-05-01 00:00:00','1h'],
        'METR-LA':['2012-03-01 00:00:00', '5m']
    }


def compute_mean(x, index=None):
    """Compute the mean values for each datetime. The mean is first computed hourly over the week of the year.
    Further NaN values are computed using hourly mean over the same month through the years. If other NaN are present,
    they are removed using the mean of the sole hours. Hoping reasonably that there is at least a non-NaN entry of the
    same hour of the NaN datetime in all the dataset."""
    if isinstance(x, np.ndarray) and index is not None:
        shape = x.shape
        x = x.reshape((shape[0], -1))
        df_mean = pd.DataFrame(x, index=index)
    else:
        df_mean = x.copy()
    cond0 = [df_mean.index.year, df_mean.index.isocalendar().week, df_mean.index.hour]
    cond1 = [df_mean.index.year, df_mean.index.month, df_mean.index.hour]
    conditions = [cond0, cond1, cond1[1:], cond1[2:]]
    while df_mean.isna().values.sum() and len(conditions):
        nan_mean = df_mean.groupby(conditions[0]).transform(np.nanmean)
        df_mean = df_mean.fillna(nan_mean)
        conditions = conditions[1:]
    if df_mean.isna().values.sum():
        df_mean = df_mean.fillna(method='ffill')
        df_mean = df_mean.fillna(method='bfill')
    if isinstance(x, np.ndarray):
        df_mean = df_mean.values.reshape(shape)
    return df_mean

class StandardScaler():
    def __init__(self, mean=None, std=None, channel_wise=False, stage1_node_idx = None, stage2_node_idx = None):
        self.mean = mean
        self.std = std
        self.channel_wise = channel_wise
        print('scale channel_wise :', channel_wise)
    
    def fit(self, data): # data: [T, N]
        if self.channel_wise:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)            
        else:
            self.mean = data.mean()
            self.std = data.std()
        return self

    def transform(self, data, stage=None):
        if self.channel_wise == True:
            node_idx = np.arange(self.mean.shape[0])
            if len(data.shape) == 2:
                assert node_idx is not None
                mean = self.mean[np.newaxis, node_idx]
                std = self.std[np.newaxis, node_idx]
            elif len(data.shape) == 3:
                assert node_idx is not None
                mean = self.mean[np.newaxis, node_idx, np.newaxis]
                std = self.std[np.newaxis, node_idx, np.newaxis]
            elif len(data.shape) == 4:
                assert node_idx is not None
                mean = self.mean[np.newaxis, np.newaxis, node_idx, np.newaxis]
                std = self.std[np.newaxis, np.newaxis, node_idx, np.newaxis]
            else:
                raise ValueError("input dimension error!!!")
        else:
            mean = self.mean
            std = self.std
        if type(data) == np.ndarray:
            return (data - mean) / std
        elif type(data) == torch.Tensor:
            return (data -torch.tensor(mean).to(data.device))/ torch.tensor(std).to(data.device)


    def inverse_transform(self, data, stage=None):
        if self.channel_wise == True:
            node_idx = np.arange(self.mean.shape[0])
            if len(data.shape) == 2:
                assert node_idx is not None
                mean = self.mean[np.newaxis, node_idx]
                std = self.std[np.newaxis, node_idx]
            elif len(data.shape) == 3:
                assert node_idx is not None
                mean = self.mean[np.newaxis, node_idx, np.newaxis]
                std = self.std[np.newaxis, node_idx, np.newaxis]
            elif len(data.shape) == 4:
                assert node_idx is not None
                mean = self.mean[np.newaxis, np.newaxis, node_idx, np.newaxis]
                std = self.std[np.newaxis, np.newaxis, node_idx, np.newaxis]
            else:
                raise ValueError("input dimension error!!!")
        else:
            mean = self.mean
            std = self.std
        if type(data) == np.ndarray:
            return (data * std) + mean
        elif type(data) == torch.Tensor:
            return (data * torch.tensor(std).to(data.device)) + torch.tensor(mean).to(data.device)

import numpy as np
from typing import Dict, Sequence, Any, Union, Tuple

def _accumulate_stats_from_stage(
    full_data: np.ndarray,                 # (L, N, D) or (L, N)
    stage_spec: Dict[str, Any],
    stage_id: int,
    ignore_zeros: bool = False,
) -> Tuple[float, float, int]:
    rec = stage_spec[str(int(stage_id))]
    t0, t1 = int(rec["time_start_idx"]), int(rec["time_end_idx"])
    cols = np.asarray(rec["node_cols_in_full"], dtype=np.int64)

    x = full_data[t0:t1, cols]
    if x.ndim == 3:
        arr = x.reshape(-1).astype(np.float64, copy=False)
    elif x.ndim == 2:
        arr = x.reshape(-1).astype(np.float64, copy=False)
    else:
        raise ValueError(f"full_data must be 2D or 3D after slicing, got {x.shape}")

    if ignore_zeros:
        arr = arr[arr != 0.0]

    cnt = int(arr.size)
    if cnt == 0:
        return 0.0, 0.0, 0

    s = float(arr.sum())
    ss = float((arr * arr).sum())
    return s, ss, cnt


def build_scaler_by_stages(
    full_data: np.ndarray,
    stage_spec: Dict[str, Any],
    train_stages: Sequence[int],
    mode: str = "concat",          # "concat" | "per_stage"
    eps: float = 1e-6,
    ignore_zeros: bool = False,
) -> Union["StandardScaler", Dict[int, "StandardScaler"]]:
    """
    Build scaler(s) from selected training stages.

    mode:
      - "concat": one global scaler, conceptually concatenating stages along node dim
      - "per_stage": one scaler per stage, returns dict {stage_id: scaler}

    ignore_zeros:
      - if True, values == 0.0 are excluded from stats (useful when missing is filled with 0)
    """
    mode = str(mode).lower()
    if mode not in ["concat", "per_stage"]:
        raise ValueError("mode must be 'concat' or 'per_stage'")
    if len(train_stages) == 0:
        raise ValueError("train_stages must be non-empty")

    def _finalize(sum_v: float, sumsq_v: float, cnt_v: int) -> "StandardScaler":
        if cnt_v <= 0:
            raise ValueError("no valid samples found for scaler stats")
        mean = sum_v / cnt_v
        var = sumsq_v / cnt_v - mean * mean
        std = float(np.sqrt(max(var, 0.0)))
        std = max(std, float(eps))
        return StandardScaler(mean=mean, std=std, channel_wise=False)

    if mode == "concat":
        total_s, total_ss, total_cnt = 0.0, 0.0, 0
        for s in train_stages:
            s_sum, s_sumsq, s_cnt = _accumulate_stats_from_stage(
                full_data, stage_spec, int(s), ignore_zeros=ignore_zeros
            )
            total_s += s_sum
            total_ss += s_sumsq
            total_cnt += s_cnt
        return [_finalize(total_s, total_ss, total_cnt)]

    # mode == "per_stage"
    out: Dict[int, StandardScaler] = {}
    for s in train_stages:
        s = int(s)
        s_sum, s_sumsq, s_cnt = _accumulate_stats_from_stage(
            full_data, stage_spec, s, ignore_zeros=ignore_zeros
        )
        out[s] = _finalize(s_sum, s_sumsq, s_cnt)
    return out


def generate_datetime_series(start_datetime_str, interval_str, n, col_name='date'):
    start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M:%S')
    interval = pd.to_timedelta(interval_str)
    date_time_series = [start_datetime + i * interval for i in range(n)]
    df = pd.DataFrame({col_name: date_time_series})
    return df

def generate_time_features(df_time, freq):
    # tod (time of day)
    time_ind = (df_time['date'].values - df_time['date'].values.astype('datetime64[D]')) / pd.to_timedelta(freq)
    t_of_d = time_ind.reshape([time_ind.shape[0],1])
    
    # dow (day of week)
    dow = df_time.date.apply(lambda row:row.dayofweek, 1)
    d_of_w = dow.values.reshape([dow.shape[0],1])
    
    return np.concatenate([t_of_d, d_of_w], axis=-1).astype(np.int64)

def load_raw_data(raw_data_path):
    df_time = None
    node_ids_full = None

    if raw_data_path.endswith('npz'):
        data = np.load(raw_data_path)['data']

    elif raw_data_path.endswith('csv'):
        df_raw = pd.read_csv(raw_data_path)
        
        if 'AQI' in raw_data_path:
            # df_raw = df_raw.fillna(compute_mean(df_raw))
            df_raw = df_raw.fillna(0.0)
        else:
            df_raw = df_raw.fillna(0.0)
        val_cols = df_raw.columns[1:]
        df_raw.rename(columns={df_raw.columns[0]: 'date'}, inplace=True)

        node_ids_full = [str(c) for c in val_cols]  # full_data 的列顺序
        data = df_raw[val_cols].values
        data = np.expand_dims(data, axis=-1).astype(np.float32, copy=False)

        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_time = df_raw[['date']]

    elif raw_data_path.endswith('h5'):
        ori = pd.read_hdf(raw_data_path)
        data = ori.values
        data = np.expand_dims(data, axis=-1).astype(np.float32, copy=False)
        df_time = pd.DataFrame({'date': ori.index})
        node_ids_full = [str(c) for c in ori.columns] if hasattr(ori, "columns") else None

    return data, df_time, node_ids_full


def load_full_data_time(dataset_name, in_dim = 1):
    if not os.path.exists(f'datasets/{dataset_name}/'):
        os.makedirs(f'datasets/{dataset_name}')
    

    print(f'loading raw data of {dataset_name}...')
    raw_data_path = file_path_dict[dataset_name]
    data, df_time, node_ids_full = load_raw_data(raw_data_path)
    full_data = data[..., :in_dim]
    filled_data = data
    full_valid_mask = None
    start_time_str, freq = meta_info_dict[dataset_name]

    if node_ids_full is None:
        node_ids_full = [str(it) for it in range(full_data.shape[1])]

    full_data = filled_data[...,:in_dim]
    
    print(f'Full Data shape:{full_data.shape}')
    

    if df_time is None:
        df_time = generate_datetime_series(start_time_str, freq, full_data.shape[0], 'date')
    
    
    full_time = generate_time_features(df_time, freq)
    # print(f"Full Time shape: {full_time.shape}")
    # np.savez_compressed(data_cache_path, data = full_data)
    # np.savez_compressed(time_cache_path, time_data = full_time)
    # df_time.to_csv(timestamp_cache_path, index=False)
    # print('full data and time_data saved!')

    return full_data, full_time, df_time, full_valid_mask, node_ids_full


@dataclass
class StageView:
    stage: int
    t0: int
    t1: int
    node_cols: np.ndarray
    node_station_ids: List[str]


class StageWindowDataset(Dataset):
    def __init__(
        self,
        full_data: np.ndarray,   # (L,N,D)
        full_time: np.ndarray,   # (L,F)
        stage_spec: Dict[str, Any],
        stages: Sequence[int],
        history_len: int,
        pred_len: int,
        stride: int = 1,
        return_node_info: bool = False,
        return_stage_id: bool = True,
        scaler = None
    ):
        if torch is None:
            raise ImportError("PyTorch is required for StageWindowDataset")

        self.full_data = full_data
        self.full_time = full_time
        self.H = int(history_len)
        self.P = int(pred_len)
        self.stride = int(stride)
        self.return_node_info = bool(return_node_info)
        self.return_stage_id = bool(return_stage_id)
        self.scaler= scaler


        if full_data.ndim != 3:
            raise ValueError(f"full_data must be (L,N,D), got {full_data.shape}")
        if full_time.ndim != 2:
            raise ValueError(f"full_time must be (L,F), got {full_time.shape}")
        if full_data.shape[0] != full_time.shape[0]:
            raise ValueError("full_data and full_time must share the same L")

        self.views: List[StageView] = []
        for s in stages:
            key = str(int(s))
            rec = stage_spec[key]
            node_cols = np.asarray(rec["node_cols_in_full"], dtype=np.int64)
            node_station_ids = [str(x) for x in rec["node_order"]]
            self.views.append(StageView(
                stage=int(s),
                t0=int(rec["time_start_idx"]),
                t1=int(rec["time_end_idx"]),
                node_cols=node_cols,
                node_station_ids=node_station_ids,
            ))

        self.index: List[Tuple[int, int]] = []
        span = self.H + self.P
        for v_idx, v in enumerate(self.views):
            if (v.t1 - v.t0) < span:
                continue
            max_start = v.t1 - span
            for st in range(v.t0, max_start + 1, self.stride):
                self.index.append((v_idx, st))

        if len(self.index) == 0:
            raise ValueError("no available windows, check time split and (history_len + pred_len)")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        v_idx, st = self.index[idx]
        v = self.views[v_idx]

        x = self.full_data[st : st + self.H, v.node_cols, :].astype(np.float32, copy=False)
        x = self.scaler.transform(x)
        y = self.full_data[st + self.H : st + self.H + self.P, v.node_cols, :].astype(np.float32, copy=False)
        tx = self.full_time[st : st + self.H, :].astype(np.int64, copy=False)
        ty = self.full_time[st + self.H : st + self.H + self.P, :].astype(np.int64, copy=False)
        out = {
                "x": torch.from_numpy(x),
                "y": torch.from_numpy(y),
                "tx": torch.from_numpy(tx),
                "ty": torch.from_numpy(ty),
                "time_start_in_full": int(st),
            }
        if self.return_stage_id:
            out["stage"] = str(v.stage)
        if self.return_node_info:
            out["node_cols_in_full"] = torch.from_numpy(v.node_cols)
            out["node_station_ids"] = v.node_station_ids
        return out
    
    
def build_stagewise_dataloaders(
    full_data: np.ndarray,
    full_time: np.ndarray,
    stage_spec: Dict[str, Any],
    stages: Sequence[str],
    history_len: int,
    pred_len: int,
    batch_size: int = 32,
    stride: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    scaler_list=None
) -> Dict[int, DataLoader]:
    if torch is None:
        raise ImportError("PyTorch is required for build_stagewise_dataloaders")

    loaders: Dict[str, DataLoader] = {}
    stage_datasets: Dict[str, DataLoader] = {}
    scaler = scaler_list[-1]



    for s in stages:
        # s = int(s)
        ds = StageWindowDataset(
            full_data=full_data,
            full_time=full_time,
            stage_spec=stage_spec,
            stages=[s],
            history_len=history_len,
            pred_len=pred_len,
            stride=stride,
            return_node_info=False,
            return_stage_id=False,
            scaler=scaler
        )
        stage_datasets[s] = ds
        loaders[s] = DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            pin_memory=True,
            drop_last=False,
        )

    return stage_datasets, loaders



def build_stagewise_splits(
    full_data: np.ndarray,
    full_time: np.ndarray,
    stage_spec: Dict[str, Any],
    train_stages: Sequence[int],
    val_stages: Sequence[int],
    test_stages: Sequence[int],
    history_len: int,
    pred_len: int,
    batch_size: int = 32,
    stride: int = 1,
    num_workers: int = 0,
    full_valid_mask: Optional[np.ndarray] = None,
    scaler_mode = 'concat'
) -> Dict[str, Dict[int, DataLoader]]:
    test_batch_size = 1

    train_scaler_list = build_scaler_by_stages(full_data, stage_spec, train_stages, mode=scaler_mode)

    train_ds, train_loaders = build_stagewise_dataloaders(
            full_data, full_time, stage_spec, train_stages,
            history_len, pred_len,
            batch_size=batch_size, stride=stride, num_workers=num_workers,
            shuffle=True, scaler_list=train_scaler_list
        )
    val_ds, val_loaders = build_stagewise_dataloaders(
            full_data, full_time, stage_spec, val_stages,
            history_len, pred_len,
            batch_size=batch_size, stride=stride, num_workers=num_workers,
            shuffle=False, scaler_list=train_scaler_list
        )
    test_ds, test_loaders = build_stagewise_dataloaders(
            full_data, full_time, stage_spec, test_stages,
            history_len, pred_len,
            batch_size=test_batch_size, stride=stride, num_workers=num_workers,
            shuffle=False, scaler_list=train_scaler_list
        )

    dataloader = {
        "train": train_loaders,
        "val": val_loaders,
        "test": test_loaders,
    }
    return dataloader, train_scaler_list



def get_stage1_survivor_index_dict(stage_spec, stage_ids=None, stage1_id=1):
    """
    Return a dict:
      {stage_id: idx_list}
    where idx_list are indices in stage_spec[stage_id]["node_order"]
    corresponding to nodes that belong to stage1 AND still appear in this stage.
    """
    stage1_nodes = set(str(x) for x in stage_spec[str(int(stage1_id))]["node_order"])

    if stage_ids is None:
        stage_ids_sorted = sorted(int(k) for k in stage_spec.keys())
    else:
        stage_ids_sorted = [int(s) for s in stage_ids]

    out = {}
    for s in stage_ids_sorted:
        node_order = [str(x) for x in stage_spec[str(s)]["node_order"]]
        idx_list = [i for i, nid in enumerate(node_order) if nid in stage1_nodes]
        out[str(s)] = idx_list
    return out
