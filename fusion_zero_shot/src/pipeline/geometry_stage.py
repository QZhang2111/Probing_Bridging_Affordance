from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Optional, Sequence


def smooth_map(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        return arr
    return cv2.GaussianBlur(arr, (0, 0), sigma)


def largest_component(base_mask: np.ndarray, priority_mask: Optional[np.ndarray] = None) -> np.ndarray:
    base_u8 = (base_mask > 0).astype(np.uint8)
    if base_u8.sum() == 0:
        return base_u8.astype(np.float32)
    num, labels = cv2.connectedComponents(base_u8)
    if num <= 1:
        return base_u8.astype(np.float32)

    best_label = 1
    if priority_mask is not None and priority_mask.sum() > 0:
        scores = []
        for lab in range(1, num):
            overlap = int(((labels == lab) & (priority_mask > 0)).sum())
            scores.append((overlap, np.count_nonzero(labels == lab), lab))
        scores.sort(reverse=True)
        best_label = scores[0][2] if scores else 1
    else:
        areas = [(np.count_nonzero(labels == lab), lab) for lab in range(1, num)]
        areas.sort(reverse=True)
        best_label = areas[0][1]

    return (labels == best_label).astype(np.float32)


def _normalize_for_cosine(arr: np.ndarray) -> Optional[np.ndarray]:
    vec = np.asarray(arr, dtype=np.float32).reshape(-1)
    if vec.size == 0:
        return None
    vmin = float(vec.min())
    vmax = float(vec.max())
    if abs(vmax - vmin) < 1e-6:
        return None
    vec = (vec - vmin) / (vmax - vmin)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def _softmax_prob(
    arr: np.ndarray,
    *,
    temperature: float,
    dirichlet: float,
    use_log1p: bool,
    eps: float,
) -> np.ndarray:
    """Convert a heatmap into a probability map via temperature-scaled softmax."""

    clamped = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    if use_log1p:
        clamped = np.log1p(clamped)

    flat = clamped.reshape(-1)
    if flat.size == 0:
        return clamped

    temp = float(max(temperature, eps))
    scaled = flat / temp
    scaled -= float(np.max(scaled))
    exp_vals = np.exp(scaled)
    denom = float(exp_vals.sum()) + eps
    prob = exp_vals / denom

    smooth = float(max(dirichlet, 0.0))
    if smooth > 0.0:
        smooth = min(smooth, 1.0)
        uniform = np.full_like(prob, 1.0 / prob.size, dtype=np.float32)
        prob = prob * (1.0 - smooth) + uniform * smooth
        prob = prob / (float(prob.sum()) + eps)

    return prob.reshape(arr.shape)


def _soft_fuse_heatmaps(
    verb_map: np.ndarray,
    geom_energy: np.ndarray,
    *,
    lam: float,
    gamma: float,
    temperature: float,
    dirichlet: float,
    use_log1p: bool,
    eps: float = 1e-6,
) -> np.ndarray:
    lam = float(np.clip(lam, 0.0, 1.0))
    gamma = float(max(gamma, eps))

    verb_prob = _softmax_prob(
        verb_map,
        temperature=temperature,
        dirichlet=dirichlet,
        use_log1p=use_log1p,
        eps=eps,
    )
    geom_pre = np.power(np.clip(geom_energy, 0.0, None), gamma)
    geom_prob = _softmax_prob(
        geom_pre,
        temperature=temperature,
        dirichlet=dirichlet,
        use_log1p=use_log1p,
        eps=eps,
    )

    mix_log = lam * np.log(verb_prob + eps) + (1.0 - lam) * np.log(geom_prob + eps)
    temp = float(max(temperature, eps))
    mix_log = mix_log / temp
    fused = np.exp(mix_log)
    fused = np.clip(fused, 0.0, None)
    total = float(fused.sum())
    if total <= eps:
        return verb_prob.astype(np.float32)
    return (fused / total).astype(np.float32)


def _build_topk_mask(
    verb_map: np.ndarray,
    percent: float,
) -> Optional[np.ndarray]:
    arr = np.clip(np.asarray(verb_map, dtype=np.float32), 0.0, None)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return None
    pct = float(np.clip(percent, 0.1, 100.0))
    cutoff = np.percentile(flat, max(0.0, 100.0 - pct))
    mask = (arr >= cutoff).astype(np.uint8)
    if mask.sum() == 0:
        idx = int(np.argmax(flat))
        mask.reshape(-1)[idx] = 1
    return mask


def _nss_score(channel: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> Optional[float]:
    arr = np.asarray(channel, dtype=np.float32)
    valid = mask > 0
    count = int(valid.sum())
    if count == 0:
        return None
    mean = float(arr.mean())
    std = float(arr.std())
    if std < eps:
        return None
    z = (arr - mean) / std
    return float(z[valid].mean())


def _topk_energy_diff(channel: np.ndarray, mask: np.ndarray) -> Optional[float]:
    arr = np.asarray(channel, dtype=np.float32)
    valid = mask > 0
    count = int(valid.sum())
    if count == 0:
        return None
    fg = arr[valid]
    bg = arr[~valid]
    if fg.size == 0:
        return None
    fg_mean = float(fg.mean())
    bg_mean = float(bg.mean()) if bg.size > 0 else 0.0
    return fg_mean - bg_mean


def _attention_priority_choice(
    stack: np.ndarray,
    verb_map: np.ndarray,
    topk_percent: float,
    nss_weight: float,
    topk_weight: float,
) -> Optional[Dict[str, object]]:
    if stack.ndim != 3 or stack.shape[2] == 0:
        return None
    if nss_weight == 0.0 and topk_weight == 0.0:
        return None

    mask = _build_topk_mask(verb_map, topk_percent)
    if mask is None:
        return None

    scores: list[Optional[float]] = []
    for idx in range(stack.shape[2]):
        ch = stack[..., idx]
        score_val = 0.0
        valid = False
        if nss_weight != 0.0:
            nss = _nss_score(ch, mask)
            if nss is not None:
                score_val += nss_weight * nss
                valid = True
        if topk_weight != 0.0:
            diff = _topk_energy_diff(ch, mask)
            if diff is not None:
                score_val += topk_weight * diff
                valid = True
        scores.append(score_val if valid else None)

    valid_pairs = [(idx, sc) for idx, sc in enumerate(scores) if sc is not None]
    if not valid_pairs:
        return None
    best_idx = max(valid_pairs, key=lambda x: x[1])[0]
    return {
        "index": best_idx,
        "scores": scores,
        "topk_pixels": int(mask.sum()),
    }


def generate_geometry_mask(
    pcs_orig: np.ndarray,
    *,
    smooth_sigma: float = 1.2,
    binary_threshold: float = 0.5,
    verb_map: Optional[np.ndarray] = None,
    enable_soft_fusion: bool = False,
    soft_lambda: float = 0.65,
    soft_gamma: float = 0.7,
    soft_temperature: float | Sequence[float] = 1.15,
    soft_dirichlet: float = 0.008,
    soft_use_log1p: bool = False,
    max_channels: Optional[int] = None,
    forced_pc_index: Optional[int] = None,
    use_attention_fallback: bool = False,
    attention_topk_percent: float = 10.0,
    attention_nss_weight: float = 1.0,
    attention_topk_weight: float = 1.0,
) -> Dict[str, np.ndarray]:
    if pcs_orig.ndim != 3 or pcs_orig.shape[2] == 0:
        raise ValueError("pcs_orig must be an HxWxC array with at least one channel")

    total_channels = pcs_orig.shape[2]
    if max_channels is None:
        num_channels = min(3, total_channels)
    else:
        num_channels = max(1, min(int(max_channels), total_channels))

    channels = []
    for ch in range(num_channels):
        channels.append(smooth_map(np.clip(pcs_orig[..., ch], 0.0, 1.0), smooth_sigma))
    stack = np.stack(channels, axis=2)

    weights = np.ones(num_channels, dtype=np.float32) / max(1, num_channels)
    pc_labels: Sequence[str] = [f"pc{idx + 1}" for idx in range(num_channels)]
    similarity_scores: Optional[list[float]] = None
    similarity_method: Optional[str] = None
    attention_meta: Optional[Dict[str, object]] = None
    selected_idx = 0

    forced_idx = None
    if forced_pc_index is not None:
        forced_idx = int(forced_pc_index) - 1
        if forced_idx < 0 or forced_idx >= num_channels:
            raise ValueError(
                f"forced_pc_index {forced_pc_index} out of range (1~{num_channels})"
            )

    if forced_idx is not None:
        selected_idx = forced_idx
        energy = np.clip(stack[..., selected_idx], 0.0, None)
        weights = np.zeros_like(weights)
        weights[selected_idx] = 1.0
    else:
        attention_choice = None
        if use_attention_fallback and verb_map is not None:
            attention_choice = _attention_priority_choice(
                stack,
                verb_map,
                attention_topk_percent,
                attention_nss_weight,
                attention_topk_weight,
            )
        if attention_choice is not None:
            selected_idx = int(attention_choice["index"])
            similarity_scores = attention_choice["scores"]
            similarity_method = "attention"
            energy = np.clip(stack[..., selected_idx], 0.0, None)
            weights = np.zeros_like(weights)
            weights[selected_idx] = 1.0
            attention_meta = {
                "topk_percent": float(np.clip(attention_topk_percent, 0.1, 100.0)),
                "topk_pixels": int(attention_choice.get("topk_pixels", 0)),
                "nss_weight": float(attention_nss_weight),
                "topk_weight": float(attention_topk_weight),
            }
        else:
            if verb_map is not None:
                verb_vec = _normalize_for_cosine(verb_map)
                if verb_vec is not None:
                    similarity_scores = []
                    valid_pairs = []
                    for idx in range(num_channels):
                        channel_vec = _normalize_for_cosine(stack[..., idx])
                        if channel_vec is None:
                            similarity_scores.append(None)
                            continue
                        cosine = float(np.dot(channel_vec, verb_vec))
                        similarity_scores.append(cosine)
                        valid_pairs.append((idx, cosine))
                    if valid_pairs:
                        selected_idx = max(valid_pairs, key=lambda x: x[1])[0]
                        similarity_method = "cosine"
                    else:
                        similarity_scores = None

            if similarity_scores is not None:
                energy = np.clip(stack[..., selected_idx], 0.0, None)
                weights = np.zeros_like(weights)
                weights[selected_idx] = 1.0
            else:
                energy = np.clip(stack[..., 0], 0.0, None)
                selected_idx = 0
                similarity_scores = similarity_scores or []
                similarity_method = None

    denom = energy.max()
    if denom < 1e-6:
        energy_norm = np.zeros_like(energy, dtype=np.float32)
    else:
        energy_norm = energy / denom
    energy_norm = energy_norm.astype(np.float32)

    geom_mask = (energy_norm >= binary_threshold).astype(np.float32)

    if isinstance(soft_temperature, (list, tuple)):
        temp_list = [float(t) for t in soft_temperature if t is not None]
    else:
        temp_list = [float(soft_temperature)]
    temp_list = [t if t > 1e-6 else 1e-6 for t in temp_list]
    if not temp_list:
        temp_list = [1.15]

    soft_fusion = None
    soft_fusion_multi: list[dict[str, object]] = []
    if enable_soft_fusion and verb_map is not None:
        for temp in temp_list:
            fused = _soft_fuse_heatmaps(
                verb_map,
                energy_norm,
                lam=soft_lambda,
                gamma=soft_gamma,
                temperature=temp,
                dirichlet=soft_dirichlet,
                use_log1p=soft_use_log1p,
            )
            entry = {
                "map": fused,
                "temperature": float(temp),
                "params": {
                    "lambda": soft_lambda,
                    "gamma": soft_gamma,
                    "temperature": float(temp),
                    "dirichlet": soft_dirichlet,
                    "log1p": bool(soft_use_log1p),
                },
            }
            soft_fusion_multi.append(entry)
        if soft_fusion_multi:
            soft_fusion = soft_fusion_multi[0]

    return {
        "energy": energy_norm,
        "mask": geom_mask,
        "weights": weights,
        "thresholds": {"binary": binary_threshold},
        "selected_pc": pc_labels[selected_idx],
        "pc_index": int(selected_idx),
        "similarity_scores": similarity_scores if similarity_scores is not None else None,
        "similarity_method": similarity_method,
        "attention_meta": attention_meta,
        "soft_fusion": soft_fusion,
        "soft_fusion_multi": soft_fusion_multi if soft_fusion_multi else None,
        "soft_temperatures": temp_list,
        "channels_used": num_channels,
        "pc_labels": list(pc_labels),
    }
