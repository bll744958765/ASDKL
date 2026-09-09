"""Dataset preparation for ASDKL.

The split is performed before neighbour construction. Validation/test targets are
never used as neighbour features; their neighbourhoods are queried only against
the training set.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class SpatialNeighborhoodDataset(Dataset):
    def __init__(self, coordinates, features, targets, reference_coordinates,
                 reference_features, reference_targets, k=16, exclude_self=False,
                 query_reference_indices=None):
        self.s = torch.as_tensor(coordinates, dtype=torch.float32)
        self.x = torch.as_tensor(features, dtype=torch.float32)
        self.y = torch.as_tensor(targets, dtype=torch.float32).reshape(-1)
        ref_s = np.asarray(reference_coordinates, dtype=np.float32)
        query_k = min(k + int(exclude_self), len(ref_s))
        distances, indices = KDTree(ref_s).query(np.asarray(coordinates), k=query_k)
        if exclude_self:
            if query_reference_indices is None:
                raise ValueError(
                    "query_reference_indices is required when exclude_self=True; "
                    "dropping the first KDTree result is unsafe for tied distances."
                )
            self_indices = np.asarray(query_reference_indices, dtype=int).reshape(-1)
            if len(self_indices) != len(indices):
                raise ValueError("query_reference_indices must match the number of queries")
            keep_indices = np.empty((len(indices), min(k, max(len(ref_s) - 1, 0))), dtype=int)
            keep_distances = np.empty_like(keep_indices, dtype=distances.dtype)
            for row, self_index in enumerate(self_indices):
                mask = indices[row] != self_index
                selected_indices = indices[row, mask][:k]
                selected_distances = distances[row, mask][:k]
                if len(selected_indices) != keep_indices.shape[1]:
                    raise RuntimeError("Failed to exclude the query sample from its reference neighbourhood")
                keep_indices[row] = selected_indices
                keep_distances[row] = selected_distances
            indices, distances = keep_indices, keep_distances
        self.neigh_s = torch.as_tensor(ref_s[indices], dtype=torch.float32)
        self.neigh_index = torch.as_tensor(indices, dtype=torch.long)
        self.neigh_x = torch.as_tensor(np.asarray(reference_features)[indices], dtype=torch.float32)
        self.neigh_y = torch.as_tensor(np.asarray(reference_targets).reshape(-1)[indices], dtype=torch.float32)
        self.neigh_dist = torch.as_tensor(distances, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"coord": self.s[idx], "aux": self.x[idx], "target": self.y[idx],
                "neigh_coord": self.neigh_s[idx], "neigh_aux": self.neigh_x[idx],
                "neigh_target": self.neigh_y[idx], "neigh_distance": self.neigh_dist[idx]}


def make_spatial_splits(S, X, y, test_size=0.2, val_size=0.1, seed=42):
    """Random held-out split with scalers fitted exclusively on training data."""
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size / (1-test_size), random_state=seed)
    coord_scaler, feature_scaler, target_scaler = StandardScaler(), StandardScaler(), StandardScaler()
    X = np.asarray(X, dtype=float).copy()
    # Fit auxiliary-variable imputation on the training partition only.
    train_medians = np.nanmedian(X[train_idx], axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
    missing_rows, missing_cols = np.where(np.isnan(X))
    X[missing_rows, missing_cols] = train_medians[missing_cols]
    coord_scaler.fit(np.asarray(S)[train_idx]); feature_scaler.fit(X[train_idx])
    target_scaler.fit(np.asarray(y)[train_idx].reshape(-1, 1))
    transformed = (coord_scaler.transform(S), feature_scaler.transform(X),
                   target_scaler.transform(np.asarray(y).reshape(-1, 1)).reshape(-1))
    return transformed, (train_idx, val_idx, test_idx), (coord_scaler, feature_scaler, target_scaler)


def preprocess_spatial_split(S, X, y, train_idx, val_idx, test_idx):
    """Apply training-only preprocessing to an externally specified split."""
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    joined = np.concatenate([train_idx, val_idx, test_idx])
    if len(np.unique(joined)) != len(joined):
        raise ValueError("External split partitions overlap")
    if np.any(joined < 0) or np.any(joined >= len(y)):
        raise ValueError("External split contains out-of-range indices")
    coord_scaler, feature_scaler, target_scaler = StandardScaler(), StandardScaler(), StandardScaler()
    X = np.asarray(X, dtype=float).copy()
    train_medians = np.nanmedian(X[train_idx], axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
    missing_rows, missing_cols = np.where(np.isnan(X))
    X[missing_rows, missing_cols] = train_medians[missing_cols]
    coord_scaler.fit(np.asarray(S)[train_idx]); feature_scaler.fit(X[train_idx])
    target_scaler.fit(np.asarray(y)[train_idx].reshape(-1, 1))
    transformed = (coord_scaler.transform(S), feature_scaler.transform(X),
                   target_scaler.transform(np.asarray(y).reshape(-1, 1)).reshape(-1))
    return transformed, (train_idx, val_idx, test_idx), (coord_scaler, feature_scaler, target_scaler)

def build_neighbor_graph(coordinates, k=8):
    distances, indices = KDTree(coordinates).query(coordinates, k=min(k + 1, len(coordinates)))
    return indices[:, 1:], distances[:, 1:]


