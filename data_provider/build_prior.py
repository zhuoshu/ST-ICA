import numpy as np
from einops import repeat
import torch

def build_stage_prior_dict_calibrated(stage_spec, d, sigma=1.0, seed=42, eps=1e-12, stage_ids=None):
    """
    Stage 1: sample pri ~ N(0, sigma^2 I)
    Stage s>1:
      - carry over pri for nodes that existed before
      - for new nodes: sample z ~ N(0, I), then pri_new = mu_ref + std_ref * z
        where mu_ref/std_ref are computed from the carried-over nodes in CURRENT stage.
    """
    if stage_ids is None:
        stage_ids_sorted = sorted(int(k) for k in stage_spec.keys())
    else:
        stage_ids_sorted = [int(s) for s in stage_ids]

    rng = np.random.default_rng(seed)

    prior_dict = {}
    prev_map = None  # node_id -> (d,)
    stage1_mu = np.zeros((d,), dtype=np.float32)
    stage1_std = np.full((d,), sigma, dtype=np.float32)

    for s in stage_ids_sorted:
        rec = stage_spec[str(s)]
        node_order = [str(x) for x in rec["node_order"]]
        n = len(node_order)

        pri = np.zeros((n, d), dtype=np.float32)

        if prev_map is None:
            # stage 1
            pri = rng.normal(0.0, sigma, size=(n, d)).astype(np.float32)
            # cache global reference (optional)
            stage1_mu = pri.mean(axis=0)
            stage1_std = pri.std(axis=0) + eps
        else:
            # 1) fill carried nodes
            carried_rows = []
            carried_idx = []
            new_idx = []

            for i, nid in enumerate(node_order):
                if nid in prev_map:
                    pri[i] = prev_map[nid]
                    carried_rows.append(pri[i])
                    carried_idx.append(i)
                else:
                    new_idx.append(i)

            # 2) compute reference stats from carried nodes in this stage
            if len(carried_rows) > 0:
                ref = np.stack(carried_rows, axis=0)  # (n_remain, d)
                mu_ref = ref.mean(axis=0)
                std_ref = ref.std(axis=0) + eps
            else:
                # fallback: use stage1 stats, or use (0, sigma)
                mu_ref, std_ref = stage1_mu, stage1_std
                # mu_ref = np.zeros((d,), dtype=np.float32)
                # std_ref = np.full((d,), sigma, dtype=np.float32)

            # 3) sample and calibrate new nodes
            n_new = len(new_idx)
            if n_new > 0:
                z = rng.normal(0.0, 1.0, size=(n_new, d)).astype(np.float32)
                pri_new = mu_ref[None, :] + z * std_ref[None, :]
                pri[new_idx, :] = pri_new

        prior_dict[str(s)] = torch.tensor(pri)

        prev_map = {nid: pri[i].copy() for i, nid in enumerate(node_order)}

    return prior_dict



def build_stage_prior_dict_all_same(stage_spec, d, sigma=1.0, seed=42, eps=1e-12, stage_ids=None):
    if stage_ids is None:
        stage_ids_sorted = sorted(int(k) for k in stage_spec.keys())
    else:
        stage_ids_sorted = [int(s) for s in stage_ids]

    rng = np.random.default_rng(seed)
    one_prior = pri = rng.normal(0.0, sigma, size=(1, d)).astype(np.float32)

    prior_dict = {}

    for s in stage_ids_sorted:
        rec = stage_spec[str(s)]
        node_order = [str(x) for x in rec["node_order"]]
        n = len(node_order)

        pri = repeat(one_prior, '1 d -> n d', n = n)
           
        prior_dict[str(s)] = torch.tensor(pri)


    return prior_dict



def build_stage_prior_dict_stage_new(stage_spec, d, sigma=1.0, seed=42, eps=1e-12, stage_ids=None):
    """
    Stage 1: sample pri ~ N(0, sigma^2 I)
    Stage s>1:
      - carry over pri for nodes that existed before
      - for new nodes: sample z ~ N(0, I), then pri_new = mu_ref + std_ref * z
        where mu_ref/std_ref are computed from the carried-over nodes in CURRENT stage.
    """
    if stage_ids is None:
        stage_ids_sorted = sorted(int(k) for k in stage_spec.keys())
    else:
        stage_ids_sorted = [int(s) for s in stage_ids]

    rng = np.random.default_rng(seed)

    prior_dict = {}
    
    for s in stage_ids_sorted:
        rec = stage_spec[str(s)]
        node_order = [str(x) for x in rec["node_order"]]
        n = len(node_order)

        pri = rng.normal(0.0, sigma, size=(n, d)).astype(np.float32)

        prior_dict[str(s)] = torch.tensor(pri)

    return prior_dict




def build_stage_prior_dict_calibrated_from_first(stage_spec, d, sigma=1.0, seed=42, eps=1e-12, stage_ids=None):
    """
    Stage 1: sample pri ~ N(0, sigma^2 I)
    Stage s>1:
      - carry over pri for nodes that existed before
      - for new nodes: sample z ~ N(0, I), then pri_new = mu_ref + std_ref * z
        where mu_ref/std_ref are computed from the carried-over nodes in CURRENT stage.
    """
    if stage_ids is None:
        stage_ids_sorted = sorted(int(k) for k in stage_spec.keys())
    else:
        stage_ids_sorted = [int(s) for s in stage_ids]

    rng = np.random.default_rng(seed)

    prior_dict = {}
    first_map = None  # node_id -> (d,)
    stage1_mu = np.zeros((d,), dtype=np.float32)
    stage1_std = np.full((d,), sigma, dtype=np.float32)

    for s in stage_ids_sorted:
        rec = stage_spec[str(s)]
        node_order = [str(x) for x in rec["node_order"]]
        n = len(node_order)

        pri = np.zeros((n, d), dtype=np.float32)

        if first_map is None:
            # stage 1
            pri = rng.normal(0.0, sigma, size=(n, d)).astype(np.float32)
            # cache global reference (optional)
            stage1_mu = pri.mean(axis=0)
            stage1_std = pri.std(axis=0) + eps
            first_map = {nid: pri[i].copy() for i, nid in enumerate(node_order)}
        else:
            # 1) fill carried nodes
            carried_rows = []
            carried_idx = []
            new_idx = []

            for i, nid in enumerate(node_order):
                if nid in first_map:
                    pri[i] = first_map[nid]
                    carried_rows.append(pri[i])
                    carried_idx.append(i)
                else:
                    new_idx.append(i)

            # 2) compute reference stats from carried nodes in this stage
            if len(carried_rows) > 0:
                ref = np.stack(carried_rows, axis=0)  # (n_remain, d)
                mu_ref = ref.mean(axis=0)
                std_ref = ref.std(axis=0) + eps
            else:
                # fallback: use stage1 stats, or use (0, sigma)
                mu_ref, std_ref = stage1_mu, stage1_std
                # mu_ref = np.zeros((d,), dtype=np.float32)
                # std_ref = np.full((d,), sigma, dtype=np.float32)

            # 3) sample and calibrate new nodes
            n_new = len(new_idx)
            if n_new > 0:
                z = rng.normal(0.0, 1.0, size=(n_new, d)).astype(np.float32)
                pri_new = mu_ref[None, :] + z * std_ref[None, :]
                pri[new_idx, :] = pri_new

        prior_dict[str(s)] = torch.tensor(pri)

        # prev_map = {nid: pri[i].copy() for i, nid in enumerate(node_order)}

    return prior_dict

