"""Controlled nonstationary local-spectral benchmark.

The latent field is a fixed mixture of spatially localized Fourier atoms.  It
therefore has a location-dependent spectrum by construction, while the random
seed controls sampling locations, covariate measurement noise and observation
noise only.  The generator is fixed before model evaluation and is shared by
every method.
"""
from __future__ import annotations

import numpy as np


class SyntheticSpatialDataset:
    def __init__(self, n_samples=3000, noise_std=0.08, random_state=42):
        self.n_samples = int(n_samples)
        self.noise_std = float(noise_std)
        self.random_state = int(random_state)

    @staticmethod
    def _local_spectral_field(coordinates):
        """Evaluate a deterministic mixture of localized oriented harmonics."""
        design_rng = np.random.default_rng(2026)
        n_atoms = 14
        centers = design_rng.uniform(0.08, 0.92, size=(n_atoms, 2))
        directions = design_rng.normal(size=(n_atoms, 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        frequencies = np.linspace(1.5, 8.0, n_atoms)
        widths = design_rng.uniform(0.12, 0.28, size=n_atoms)
        phases = design_rng.uniform(0.0, 2.0 * np.pi, size=n_atoms)
        amplitudes = design_rng.uniform(0.7, 1.3, size=n_atoms)

        field = np.zeros(len(coordinates), dtype=float)
        effective_mass = np.zeros(len(coordinates), dtype=float)
        for center, direction, frequency, width, phase, amplitude in zip(
                centers, directions, frequencies, widths, phases, amplitudes):
            offset = coordinates - center
            envelope = np.exp(-np.sum(offset * offset, axis=1) / (2.0 * width * width))
            # Explicit two-dimensional projection avoids vendor-BLAS startup
            # overhead and makes the generator stable on Windows worker jobs.
            projection = offset[:, 0] * direction[0] + offset[:, 1] * direction[1]
            carrier = np.sin(2.0 * np.pi * frequency * projection + phase)
            field += amplitude * envelope * carrier
            effective_mass += envelope
        field /= np.sqrt(np.maximum(effective_mass, 0.25))
        return (field - field.mean()) / field.std()

    def generate(self):
        rng = np.random.default_rng(self.random_state)
        coordinates = rng.uniform(0.0, 1.0, size=(self.n_samples, 2))
        s1, s2 = coordinates[:, 0], coordinates[:, 1]

        # Imperfect, low-frequency auxiliary variables.  None is a copy of the
        # response; their measurement noise is independent of observation noise.
        x1 = 0.8 * s1 - 0.35 * s2 + 0.28 * rng.normal(size=self.n_samples)
        x2 = (np.exp(-10.0 * ((s1 - 0.35) ** 2 + (s2 - 0.68) ** 2))
              + 0.18 * rng.normal(size=self.n_samples))
        x3 = (np.sin(2.0 * np.pi * s1) * np.cos(2.0 * np.pi * s2)
              + 0.18 * rng.normal(size=self.n_samples))
        features = np.column_stack([x1, x2, x3])

        # A smooth nonlinear attribute trend prevents any comparator from
        # receiving the true trend as a hand-specified linear formula.
        trend = (1.20 * np.sin(np.pi * x1 * x3)
                 + 0.85 * np.tanh(2.0 * x2 - 0.4)
                 + 0.55 * x1 ** 2 - 0.45 * x2 * x3)
        local_spectrum = self._local_spectral_field(coordinates)
        noise_scale = self.noise_std * (0.65 + 0.7 * s1)
        response = trend + 1.15 * local_spectrum + noise_scale * rng.normal(size=self.n_samples)
        return coordinates, features, response
