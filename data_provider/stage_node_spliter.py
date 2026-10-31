import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import os

def assign_expansion_stages(
    df_meta: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    k: int,
    method: str = "equal_radius",
    first_stage_count: Optional[int] = None,
    first_radius_ratio: Optional[float] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[int, Dict[str, Any]]]:
    if k <= 0:
        raise ValueError("k must be positive")
    df = df_meta.copy()
    df["station_id"] = df["station_id"].astype(str)

    R = 6371.0
    lat_rad = np.radians(df["latitude"].to_numpy(dtype=float))
    lon_rad = np.radians(df["longitude"].to_numpy(dtype=float))
    clat = math.radians(float(center_lat))
    clon = math.radians(float(center_lon))

    dlat = lat_rad - clat
    dlon = lon_rad - clon
    a = np.sin(dlat / 2) ** 2 + np.cos(clat) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    dist_km = R * c
    df["dist_km"] = dist_km

    n = len(df)
    method = str(method).lower()
    if method == "quantile":
        
        if first_stage_count is not None and k > 1:
            first_stage_count = max(1, min(int(first_stage_count), n))
            df_sorted = df.sort_values("dist_km")
            first_threshold = float(df_sorted["dist_km"].iloc[first_stage_count - 1])

            mask_remain = df["dist_km"] > first_threshold
            dist_remain = df.loc[mask_remain, "dist_km"].to_numpy(dtype=float)

            thresholds: List[float] = [0.0, first_threshold]
            if len(dist_remain) > 0:
                qs_remain = np.linspace(0, 1, (k - 1) + 1)
                t_rem = np.quantile(dist_remain, qs_remain)
                thresholds.extend([float(x) for x in t_rem[1:]])
            else:
                max_r = float(df["dist_km"].max())
                thresholds = [float(x) for x in np.linspace(0.0, max_r, k + 1)]
        else:
            qs = np.linspace(0, 1, k + 1)
            thresholds = [float(x) for x in np.quantile(df["dist_km"].to_numpy(dtype=float), qs)]
    elif method == "equal_radius":
        r_min = 0.0
        r_max = float(df["dist_km"].max())
        if first_radius_ratio is not None and k > 1:
            first_radius_ratio = max(0.0, min(1.0, float(first_radius_ratio)))
            if first_radius_ratio == 0.0:
                thresholds = [float(x) for x in np.linspace(r_min, r_max, k + 1)]
            else:
                r1 = r_max * first_radius_ratio
                thresholds = [0.0, float(r1)]
                remaining = r_max - r1
                for i in range(2, k + 1):
                    ri = r1 + remaining * (i - 1) / (k - 1)
                    thresholds.append(float(ri))
        else:
            thresholds = [float(x) for x in np.linspace(r_min, r_max, k + 1)]
    else:
        raise ValueError(f"unknown method: {method!r}")

    thresholds = np.asarray(thresholds, dtype=float)
    bins = thresholds[1:-1]
    ring_id = np.digitize(df["dist_km"].to_numpy(dtype=float), bins) + 1
    df["ring_id"] = ring_id.astype(int)

    stages: Dict[int, Dict[str, Any]] = {}
    time_ratio_list = [] 
    for stage in range(1, k + 1):
        new_nodes = df.loc[df["ring_id"] == stage, "station_id"].astype(str).tolist()
        current_nodes = df.loc[df["ring_id"] <= stage, "station_id"].astype(str).tolist()
        stages[stage] = {
            "new_nodes": new_nodes,
            "current_nodes": current_nodes,
            "num_all_nodes": len(current_nodes),
            "num_new_nodes": len(new_nodes),
        }
        time_ratio_list.append(len(new_nodes)*1.0/n)
        # print(len(new_nodes)*1.0/n)

    return df, thresholds, stages, time_ratio_list



def _stage_colors(k: int):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    cmap = plt.get_cmap("tab20")
    colors = []
    for i in range(k):
        c = cmap(i % 20)
        colors.append(mcolors.to_hex(c))
    return colors

import folium

def save_stage_map_html(
    df_stage,
    out_html,
    center_lat=None,
    center_lon=None,
    thresholds_km=None,
    zoom_start=10,
    point_radius=4,
    point_opacity=0.85,
):
    df = df_stage.copy()
    df["station_id"] = df.get("station_id", df.get("sensor_id")).astype(str)
    df["ring_id"] = df["ring_id"].astype(int)

    if center_lat is None:
        center_lat = float(df["latitude"].mean())
    if center_lon is None:
        center_lon = float(df["longitude"].mean())

    k = int(df["ring_id"].max())
    colors = _stage_colors(k)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=None, control_scale=True)

    folium.TileLayer("CartoDB Voyager", name="Carto Voyager", control=True).add_to(m)
    if thresholds_km is not None:
        thresholds_km = np.asarray(thresholds_km, dtype=float)
        for i, r_km in enumerate(thresholds_km[1:], start=1):
            folium.Circle(
                location=[center_lat, center_lon],
                radius=float(r_km) * 1000.0,
                color=colors[(i - 1) % k],
                weight=2,
                fill=False,
                opacity=0.6,
                tooltip=f"threshold r{i} = {r_km:.2f} km",
            ).add_to(m)

    for stage in range(1, k + 1):
        fg = folium.FeatureGroup(name=f"Stage {stage}", show=True)
        sub = df[df["ring_id"] == stage]
        color = colors[stage - 1]
        for _, row in sub.iterrows():
            tip = f"id={row['station_id']}, {row['latitude']},{row['longitude']},stage={stage}"
            if "dist_km" in row and pd.notna(row["dist_km"]):
                tip += f", dist={float(row['dist_km']):.2f}km"
            folium.CircleMarker(
                location=[float(row["latitude"]), float(row["longitude"])],
                radius=point_radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=point_opacity,
                opacity=point_opacity,
                tooltip=tip,
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    m.save(out_html)
    return out_html, m





def split_time_by_ratios(L: int, ratios: Sequence[float]) -> List[Tuple[int, int]]:
    ratios = [float(r) for r in ratios]
    if any(r < 0 for r in ratios):
        raise ValueError("ratios must be non-negative")
    s = sum(ratios)
    if s <= 0:
        raise ValueError("sum(ratios) must be positive")
    ratios = [r / s for r in ratios]

    lens = [int(math.floor(L * r)) for r in ratios]
    if L >= len(lens):
        for i in range(len(lens)):
            if lens[i] == 0:
                lens[i] = 1

    used = sum(lens)
    if used > L:
        excess = used - L
        order = np.argsort(lens)[::-1]
        for idx in order:
            if excess <= 0:
                break
            dec = min(excess, max(0, lens[idx] - 1))
            lens[idx] -= dec
            excess -= dec
    else:
        lens[-1] += (L - used)

    bounds: List[Tuple[int, int]] = []
    cur = 0
    for ln in lens:
        bounds.append((cur, cur + ln))
        cur += ln
    bounds[-1] = (bounds[-1][0], L)
    return bounds

def apply_stagewise_abandon(
    stages: Dict[int, Dict[str, Any]],
    abandon_rate: float,
    seed: int = 42,
    abandon_scope: str = "old",      # "old" or "all"
    stage1_abandon_rate: float = 0.0 # 如果你仍然希望第一阶段不丢，就保持 0.0
) -> Dict[int, Dict[str, List[str]]]:
    """
    Stage-wise random node abandon, independent across stages.

    abandon_scope:
      - "old": only abandon old nodes at stage s, old = current_nodes(s) - new_nodes(s)
      - "all": abandon from all current nodes at stage s

    Returns:
      abandon_info[s] = {"abandoned_this_stage": [...]}  # no cumulative effect
    """
    rng = np.random.default_rng(seed)
    abandon_rate = max(0.0, min(1.0, float(abandon_rate)))
    stage1_abandon_rate = max(0.0, min(1.0, float(stage1_abandon_rate)))
    abandon_scope = str(abandon_scope).lower()
    if abandon_scope not in ["old", "all"]:
        raise ValueError("abandon_scope must be 'old' or 'all'")

    info: Dict[int, Dict[str, List[str]]] = {}

    for s in sorted(stages):
        cur_nodes = set(map(str, stages[s]["current_nodes"]))
        new_nodes = set(map(str, stages[s]["new_nodes"]))
        old_nodes = cur_nodes - new_nodes

        rate_s = stage1_abandon_rate if s == 1 else abandon_rate

        if rate_s == 0.0:
            info[s] = {"abandoned_this_stage": []}
            continue

        if abandon_scope == "old":
            candidates = sorted(old_nodes)
        else:  # "all"
            candidates = sorted(cur_nodes)

        n = len(candidates)
        k = int(round(n * rate_s))
        k = max(0, min(k, n))
        chosen = rng.choice(candidates, size=k, replace=False).tolist() if k > 0 else []
        info[s] = {"abandoned_this_stage": sorted(chosen)}

    return info




def build_stage_node_order(meta_station_ids: Sequence[str], old_nodes: Sequence[str], new_nodes: Sequence[str]) -> List[str]:
    old_set = set(map(str, old_nodes))
    new_set = set(map(str, new_nodes))
    meta_station_ids = [str(x) for x in meta_station_ids]
    old_order = [sid for sid in meta_station_ids if sid in old_set]
    new_order = [sid for sid in meta_station_ids if sid in new_set]
    return old_order + new_order


def _build_stage_spec(
    df_meta: pd.DataFrame,
    stages: Dict[int, Dict[str, Any]],
    time_bounds: Sequence[Tuple[int, int]],
    abandon_info: Dict[int, Dict[str, List[str]]],
    node_ids_full: Sequence[str], 
) -> Dict[str, Any]:
    
    node_ids_full = [str(x) for x in node_ids_full]
    id_to_col = {sid: i for i, sid in enumerate(node_ids_full)}

    df_meta = df_meta.copy()
    df_meta["station_id"] = df_meta["station_id"].astype(str)
    df_meta = df_meta[df_meta["station_id"].isin(id_to_col)].reset_index(drop=True)

    spec: Dict[str, Any] = {}
    for idx, s in enumerate(sorted(stages)):
        t0, t1 = time_bounds[idx]

        current_nodes = set(map(str, stages[s]["current_nodes"]))
        new_nodes = set(map(str, stages[s]["new_nodes"]))

        abandoned = set(map(str, abandon_info.get(s, {}).get("abandoned_this_stage", [])))

        visible_nodes = current_nodes - abandoned

        old_visible = visible_nodes - new_nodes
        new_visible = visible_nodes & new_nodes
        
        old_order = [sid for sid in node_ids_full if sid in old_visible]
        new_order = [sid for sid in node_ids_full if sid in new_visible]
        node_order = old_order + new_order
        node_cols = [int(id_to_col[sid]) for sid in node_order]

        

        spec[str(s)] = {
            "stage": int(s),
            "time_start_idx": int(t0),
            "time_end_idx": int(t1),
            "node_order": node_order,
            "node_cols_in_full": node_cols,
            "num_visible_nodes": int(len(node_order)),
            "abandoned_this_stage": sorted(abandoned),
            "num_new_nodes":len(new_order),
            "num_ori_nodes":len(current_nodes),
            "num_abandon_nodes":len(abandoned)
        }

    return spec



def save_stage_description(
    out_dir: Union[str, Path],
    df_meta_with_stage: pd.DataFrame,
    thresholds_km: np.ndarray,
    stage_spec: Dict[str, Any],
    summary: Dict[str, Any],
    overwrite: bool = True,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta_with_stage.csv"
    if overwrite or (not meta_path.exists()):
        df_meta_with_stage.to_csv(meta_path, index=False)

    with open(out_dir / "stage_spec.json", "w", encoding="utf-8") as f:
        json.dump(stage_spec, f, ensure_ascii=False, indent=2)

    summary = dict(summary)
    summary["thresholds_km"] = [float(x) for x in thresholds_km.tolist()]
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)




def build_space_time_division(
    dataset_name: str,
    meta_csv_path: Union[str, Path],
    L,
    N,
    node_ids_full,
    out_dir: Union[str, Path],
    k: int,
    method: str = "equal_radius",
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    first_stage_count: Optional[int] = None,
    first_radius_ratio: Optional[float] = None,
    time_ratios: Optional[Sequence[float]] = None,
    time_ratio_mode: str = "all",
    abandon_rate: float = 0.02,
    seed: int = 42,
    overwrite: bool = True,
    
) -> Dict[str, Any]:
    
    df_meta = pd.read_csv(meta_csv_path)
    required = {"station_id", "latitude", "longitude"}
    if not required.issubset(df_meta.columns):
        raise ValueError(f"meta csv must contain columns: {sorted(required)}")
    df_meta = df_meta.copy()
    df_meta["station_id"] = df_meta["station_id"].astype(str)


    if len(df_meta) != N:
        raise ValueError(f"meta rows N={len(df_meta)} must equal full_data N={N}, align ordering first")

    if center_lat is None:
        center_lat = float(df_meta["latitude"].mean())
    if center_lon is None:
        center_lon = float(df_meta["longitude"].mean())

    df_meta_with_stage, thresholds, stages, default_time_ratios = assign_expansion_stages(
        df_meta,
        center_lat=center_lat,
        center_lon=center_lon,
        k=int(k),
        method=method,
        first_stage_count=first_stage_count,
        first_radius_ratio=first_radius_ratio,
    )


    out_html, m = save_stage_map_html(
        df_stage=df_meta_with_stage,
        out_html=f'{out_dir}/stage_map.html',
        center_lat=center_lat,
        center_lon=center_lon,
        thresholds_km=thresholds,
        zoom_start=11,
    )
    # print(out_html)

    ratios = default_time_ratios if time_ratios is None else [float(x) for x in time_ratios]
    if len(ratios) != k:
        raise ValueError(f"time ratios length must be k={k}, got {len(ratios)}")

    time_bounds = split_time_by_ratios(L, ratios)
    abandon_info = apply_stagewise_abandon(
        stages,
        abandon_rate=abandon_rate,   
        seed=seed,
        abandon_scope="old",     
        stage1_abandon_rate=0.0
    )
    stage_spec = _build_stage_spec(df_meta, stages, time_bounds, abandon_info, node_ids_full)

    summary = {
        "dataset_name": dataset_name,
        "k": int(k),
        "method": method,
        "center": {"lat": float(center_lat), "lon": float(center_lon)},
        "time_ratio_mode": str(time_ratio_mode),
        "time_ratios": [float(x) for x in ratios],
        "num_ori_nodes":'->'.join([str(stage_spec[str(s+1)]['num_ori_nodes']) for s in range(k)]),
        "num_nodes":'->'.join([str(stage_spec[str(s+1)]['num_visible_nodes']) for s in range(k)]),
        "num_new_nodes":'->'.join([str(stage_spec[str(s+1)]['num_new_nodes']) for s in range(k)]),
        "num_aban_nodes":'->'.join([str(stage_spec[str(s+1)]['num_abandon_nodes']) for s in range(k)]),
        "time_bounds": [{"stage": i + 1, "start": int(s), "end": int(e)} for i, (s, e) in enumerate(time_bounds)],
        "abandon_rate": float(abandon_rate),
        "seed": int(seed),
        "meta_csv_path": str(meta_csv_path),
    }

    # save_stage_description(out_dir, df_meta_with_stage, thresholds, stage_spec, summary, overwrite=overwrite)
    return {"summary": summary, "stage_spec": stage_spec}, m



def main():

    dataset_name = 'UrbanEV'
    config = {
        'urbanev':{
            'k':6,
            'center_lat':22.5463133, 
            'center_lon':114.0534009,
            # 'method':'equal_radius',
            'method':'quantile',
            'first_stage_count':125,
            'first_radius_ratio':None,
            'abandon_rate':0.02,
            'L':52128,
            'N':275,
            "meta_csv_path":'./raw_data/UrbanEV_s/UrbanEV_station_info.csv',
            'out_dir':'.',
            'seed':42,
        }
    }
    from types import SimpleNamespace
    args = SimpleNamespace(**config[dataset_name.lower()])

    ret = build_space_time_division(
        dataset_name=dataset_name,
        meta_csv_path=args.meta_csv_path,
        out_dir=args.out_dir,
        k=args.k,
        method=args.method,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        first_stage_count=args.first_stage_count,
        first_radius_ratio=args.first_radius_ratio,
        time_ratios=None,
        time_ratio_mode='all',
        abandon_rate=args.abandon_rate,
        seed=args.seed,
        overwrite=True,
        L=args.L,
        N=args.N
    )
    print(json.dumps(ret["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
