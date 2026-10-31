import torch

def load_partial_state_dict(model, ckpt_path, map_location="cpu", verbose=True):
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    model_state = model.state_dict()

    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if verbose:
        print(f"[load_partial] loaded: {len(filtered)} params")
        if skipped:
            print(f"[load_partial] skipped (not found or shape mismatch): {len(skipped)}")
        if missing:
            print(f"[load_partial] missing in ckpt (kept init): {len(missing)}")
        if unexpected:
            print(f"[load_partial] unexpected (should be none after filter): {len(unexpected)}")

    return {"loaded": list(filtered.keys()), "skipped": skipped, "missing": missing, "unexpected": unexpected}
