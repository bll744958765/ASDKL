"""Fast structural tests that do not require external datasets or a GPU."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "methods" / "ASDKL"))

from synthetic_dataset import SyntheticSpatialDataset
from data_loader import SpatialNeighborhoodDataset, make_spatial_splits
from asdkl import ASDKL


class SmokeTests(unittest.TestCase):
    def setUp(self):
        raw = SyntheticSpatialDataset(n_samples=80, random_state=1042).generate()
        self.transformed, self.indices, _ = make_spatial_splits(*raw, seed=1042)

    def test_split_sizes_and_disjoint_indices(self):
        train, validation, test = (set(part.tolist()) for part in self.indices)
        self.assertEqual(len(train), 56)
        self.assertEqual(len(validation), 8)
        self.assertEqual(len(test), 16)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)

    def test_validation_neighbours_are_training_references(self):
        coordinates, features, targets = self.transformed
        train, validation, _ = self.indices
        dataset = SpatialNeighborhoodDataset(
            coordinates[validation], features[validation], targets[validation],
            coordinates[train], features[train], targets[train], k=6,
        )
        training_targets = targets[train]
        for value in dataset.neigh_y.numpy().ravel():
            self.assertTrue(np.isclose(value, training_targets).any())

    def test_model_forward_shapes(self):
        coordinates, features, targets = self.transformed
        train, _, _ = self.indices
        dataset = SpatialNeighborhoodDataset(
            coordinates[train], features[train], targets[train],
            coordinates[train], features[train], targets[train], k=6,
            exclude_self=True, query_reference_indices=np.arange(len(train)),
        )
        batch = [dataset[index] for index in range(4)]
        stack = lambda key: torch.stack([item[key] for item in batch])
        model = ASDKL(2, features.shape[1], freq_dim=8, hidden_dim=8,
                      trend_width=16, trend_depth=2)
        output = model(stack("coord"), stack("aux"), stack("neigh_coord"),
                       stack("neigh_aux"), stack("neigh_target"))
        self.assertEqual(tuple(output["mu"].shape), (4, 1))
        self.assertEqual(output["kernel_feature"].shape[0], 4)


if __name__ == "__main__":
    unittest.main()



