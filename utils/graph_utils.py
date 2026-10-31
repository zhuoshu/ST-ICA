
import numpy as np
import copy
import scipy.sparse as sp
from scipy.sparse import linalg

import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import torch
import os

# data_root_path = 'none'
data_root_path = './raw_data'
if data_root_path == 'none':
    raise ValueError('Please replace `root_path` into the absolute path of `raw_data` folder!')


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    # print(f"Number of isolated points: {isolated_point_num}")
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num

def cal_lape(adj_mx, lape_dim=32):
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    laplacian_pe = EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]
    return laplacian_pe



def compute_clustering_coefficient(A):
    N = A.shape[0]
    C = np.zeros((N,1))
    for i in range(N):
        neighbors = np.where(A[i] > 0)[0]
        d = len(neighbors)
        if d < 2:
            C[i] = 0.0
            continue
        sub = A[np.ix_(neighbors, neighbors)]
        actual = sub.sum()  
        possible = d * (d - 1)
        C[i] = actual / possible
    return C

def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights



def cal_adj(adj, adjtype='scalap'):
    if adjtype == 'auto':
        if (adj==adj.T).all():
            adjtype='transition'
        else:
            adjtype='doubletransition'

    # d = np.sum(A, axis=1)
    # sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # A_mx = np.mat(np.identity(A.shape[0]) + sinvD * A * sinvD)
    # adj_mx = copy.deepcopy(A_mx)

    if adjtype == "scalap":
        adj_mx = [calculate_scaled_laplacian(adj)]
    elif adjtype == "normlap":
        adj_mx,_ = [calculate_normalized_laplacian(adj).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj_mx = [sym_adj(adj)]
    elif adjtype == "transition":
        adj_mx = [asym_adj(adj)]
    elif adjtype == "doubletransition":
        adj_mx = [asym_adj(adj), asym_adj(np.transpose(adj))]
    elif adjtype == "identity":
        adj_mx = [np.diag(np.ones(adj.shape[0])).astype(np.float32)]
    else:
        assert 0, "adj type not defined"
    return adj_mx



def geographical_distance(x=None, to_rad=True):
    _AVG_EARTH_RADIUS_KM = 6371.0088
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)
    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM
    return pd.DataFrame(distances, x.index, x.index) if isinstance(x, pd.DataFrame) else distances


def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    w = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = (x > threshold) if threshold_on_input else (w < threshold)
        w[mask] = 0.0
    return w

def load_or_build_adjacency(
    node_ids_full,
    meta_df: pd.DataFrame,                  # station_id, latitude, longitude
    adj_csv_path: str = None,
    csv_matrix_type: str = "distance",        # "binary" | "distance"
    build_mode: str = "gaussian",           # when no csv: "distance" | "gaussian"
    threshold: float = None,
    threshold_on_input: bool = False,        # if gaussian: True -> threshold on distance(km)
    self_loop: bool = True,
    dtype=np.float32,
):
    """
    Return A (N,N) whose node order is consistent with node_ids_full.

    If adj_csv_path exists:
      - read csv numeric matrix
      - reorder rows/cols to node_ids_full
      - if csv_matrix_type == "distance": convert to weights via gaussian kernel
    Else:
      - build from lat/lon (distance or gaussian weights)
    """
    node_ids_full = [str(x) for x in node_ids_full]

    if adj_csv_path is not None and os.path.exists(adj_csv_path):
        df = pd.read_csv(adj_csv_path)
        col_ids = [str(c) for c in df.columns]
        M = df.values.astype(np.float64, copy=False)  # distance or weight

        order = [col_ids.index(sid) for sid in node_ids_full]
        M = M[np.ix_(order, order)]  # reorder to full order
        
        csv_matrix_type = str(csv_matrix_type).lower()
        if csv_matrix_type == "distance":
            theta = M.std()
            A = thresholded_gaussian_kernel(
                M, theta=theta, threshold=threshold, threshold_on_input=bool(threshold_on_input)
            ).astype(dtype, copy=False)
        elif csv_matrix_type == "binary":
            A = M.astype(dtype, copy=False)
        else:
            raise ValueError("csv_matrix_type must be 'weight' or 'distance'")

    else:
        meta_df = meta_df.copy()
        meta_df["station_id"] = meta_df["station_id"].astype(str)
        meta_df = meta_df.set_index("station_id")
        latlon = meta_df.loc[node_ids_full, ["latitude", "longitude"]].to_numpy(dtype=np.float64)

        dist_km = geographical_distance(latlon, to_rad=True)
        theta = np.std(dist_km)
        build_mode = str(build_mode).lower()
        if build_mode == "distance":
            A = dist_km.astype(dtype, copy=False)
        elif build_mode == "gaussian":
            A = thresholded_gaussian_kernel(
                dist_km, theta=theta, threshold=threshold, threshold_on_input=bool(threshold_on_input)
            ).astype(dtype, copy=False)
        else:
            raise ValueError("build_mode must be 'distance' or 'gaussian'")

    if not self_loop:
        np.fill_diagonal(A, 0.0)
    return A


def load_full_adj(dataset_name, node_ids_full, raw_adj_type ='distance'):
    if dataset_name in ['UrbanEV_s']:
        if raw_adj_type =='distance':
            adj_csv_path = f'{data_root_path}/UrbanEV_s/distance.csv'
        elif raw_adj_type == 'binary':
            adj_csv_path = f'{data_root_path}/UrbanEV_s/adj.csv'
    else:
        adj_csv_path = None

    adj_info_dict= {
        'UrbanEV_s':{
            'meta_info_path':f'{data_root_path}/UrbanEV_s/UrbanEV_station_info.csv',
            'threshold':0.2},
        'GBA':{
            'meta_info_path':f'{data_root_path}/GBA/GBA_station_info.csv',
            'threshold':0.2,},
        'EPeMS':{
            'meta_info_path':f'{data_root_path}/EPeMS/EPeMS_station_info.csv',
            'threshold':0.2,},
        'AQI':{
            'meta_info_path':f'{data_root_path}/AQI/AQI_station_info.csv',
            'threshold':0.1,},
        'METR-LA':{
            'meta_info_path':f'{data_root_path}/METR-LA/METR-LA_station_info.csv',
            'threshold':0.1,},
            
        }    
    meta_df = pd.read_csv(adj_info_dict[dataset_name]['meta_info_path'])
    full_adj = load_or_build_adjacency(node_ids_full, meta_df, adj_csv_path,
                                       raw_adj_type, threshold = adj_info_dict[dataset_name]['threshold'])
    return full_adj

def build_stage_adj_dict(full_adj, stage_spec, stage_ids=None, add_self_loop=True, norm_type = 'auto', device=None, lpls=False):
    """
    full_adj: (N, N) numpy array, order matches node_ids_full
    stage_spec: dict, each stage has "node_cols_in_full"
    Returns:
      stage_adjs: list of (N_s, N_s) numpy arrays, ordered by stage_ids_sorted
      stage_ids_sorted: list of stage ids corresponding to stage_adjs
    """
    if stage_ids is None:
        stage_ids_sorted = sorted(int(k) for k in stage_spec.keys())
    else:
        stage_ids_sorted = [int(s) for s in stage_ids]

    full_adj = np.asarray(full_adj)
    stage_adjs = {}
    stage_ori_adj = {}
    stage_lpls_dict = {} if lpls else None
    stage_nadj_dict = {} if lpls else None

    for s in stage_ids_sorted:
        cols = np.asarray(stage_spec[str(s)]["node_cols_in_full"], dtype=np.int64)
        A = full_adj[np.ix_(cols, cols)].copy()

        if add_self_loop:
            np.fill_diagonal(A, 1.0)

        if norm_type:
            A_list = cal_adj(A, norm_type)
        stage_adjs[str(s)]= [torch.tensor(it).to(device) for it in A_list]
        stage_ori_adj[str(s)]= A
        if lpls:
            stage_lpls_dict[str(s)] = torch.tensor(cal_lape(A), dtype=torch.float32).to(device)
            stage_nadj_dict[str(s)] = torch.tensor(get_normalized_adj(A), dtype=torch.float32).to(device)

    return stage_adjs, stage_ori_adj, stage_lpls_dict, stage_nadj_dict


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave