"""ASDKL dataset registry used by the command-line experiment runner."""
from pathlib import Path
import numpy as np
import pandas as pd
from synthetic_dataset import SyntheticSpatialDataset


def load_dataset(name, data_root, seed=42, max_samples=None):
    name = name.lower()
    if name == "synthetic":
        n = max_samples or 3000
        return SyntheticSpatialDataset(n_samples=n, random_state=seed).generate()
    if name == "california":
        path = Path(data_root) / "California_Housing" / "fetch_california_housing.csv"
        frame = pd.read_csv(path)
        required = ["Latitude", "Longitude", "MedInc", "HouseAge", "AveRooms",
                    "AveBedrms", "Population", "AveOccup", "target"]
        frame[required] = frame[required].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=required)
        if max_samples and len(frame) > max_samples:
            frame = frame.sample(max_samples, random_state=seed)
        coordinates = frame[["Longitude", "Latitude"]].to_numpy()
        features = frame[["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                          "Population", "AveOccup"]].to_numpy()
        return coordinates, features, frame["target"].to_numpy()
    if name == "lucas":
        path = Path(data_root) / "LUCAS 2018 TopSoil Dataset" / "OriginalSoilDataset.csv"
        frame = pd.read_csv(path)
        coords = ["TH_LONG", "TH_LAT"]
        features = ["pH_CaCl2", "pH_H2O", "EC", "CaCO3", "N", "P", "K", "Ox_Al", "Ox_Fe", "Elev"]
        required = coords + features + ["OC"]
        frame = frame[required].apply(pd.to_numeric, errors="coerce").dropna()
        if max_samples and len(frame) > max_samples:
            frame = frame.sample(max_samples, random_state=seed)
        return frame[coords].to_numpy(), frame[features].to_numpy(), np.log1p(frame["OC"].to_numpy())
    if name == "lucas_large":
        path = Path(data_root) / "LUCAS 2018 TopSoil Dataset" / "OriginalSoilDataset.csv"
        frame = pd.read_csv(path, low_memory=False)
        coords = ["TH_LONG", "TH_LAT"]
        # Ox-Al/Ox-Fe and CaCO3 are omitted because their sparse coverage would
        # reduce the 18,984-record survey to fewer than 1,500 complete rows.
        features = ["pH_CaCl2", "pH_H2O", "EC", "N", "P", "K", "Elev"]
        required = coords + features + ["OC"]
        frame = frame[required].apply(pd.to_numeric, errors="coerce").dropna()
        if max_samples and len(frame) > max_samples:
            frame = frame.sample(max_samples, random_state=seed)
        return frame[coords].to_numpy(), frame[features].to_numpy(), np.log1p(frame["OC"].to_numpy())
    if name == "pm25":
        path = Path(data_root) / "EPA_PM25_2023" / "raw" / "annual_conc_by_monitor_2023.csv"
        frame = pd.read_csv(path, low_memory=False)
        # One consistent regulatory annual summary per monitor. The coverage
        # threshold removes monitors whose annual means are weakly supported.
        frame = frame[
            frame["Parameter Code"].eq(88101)
            & frame["Pollutant Standard"].eq("PM25 Annual 2012")
            & frame["Metric Used"].eq("Quarterly Means of Daily Means")
            & frame["Observation Percent"].ge(75)
        ].copy()
        required = ["Latitude", "Longitude", "Arithmetic Mean", "Observation Percent", "Valid Day Count"]
        frame[required] = frame[required].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=required).drop_duplicates(
            subset=["State Code", "County Code", "Site Num", "POC"], keep="first"
        )
        if max_samples and len(frame) > max_samples:
            frame = frame.sample(max_samples, random_state=seed)
        S = frame[["Longitude", "Latitude"]].to_numpy()
        X = frame[["Observation Percent", "Valid Day Count"]].to_numpy()
        return S, X, frame["Arithmetic Mean"].to_numpy()
    if name == "meuse":
        path = Path(data_root) / "Meuse_Heavy_Metals" / "meuse.csv"
        frame = pd.read_csv(path)
        numeric = ["x", "y", "cadmium", "copper", "lead", "zinc", "elev", "dist", "om", "ffreq", "soil", "lime", "dist.m"]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=["x", "y", "zinc"])
        if max_samples and len(frame) > max_samples:
            frame = frame.sample(max_samples, random_state=seed)
        S = frame[["x", "y"]].to_numpy()
        # Co-located metal assays and terrain/floodplain descriptors are valid
        # auxiliary variables; zinc itself is used only as the response.
        features = ["cadmium", "copper", "lead", "elev", "dist", "om", "ffreq", "soil", "lime", "dist.m"]
        return S, frame[features].to_numpy(), np.log1p(frame["zinc"].to_numpy())
    raise ValueError(f"Unknown dataset '{name}'. Choose from: synthetic, california, lucas, lucas_large, pm25, meuse")
