"""Fetch the public scikit-learn California housing data into the expected CSV.

Run from the repository root: python scripts/prepare_california.py
The downloaded data live under data/ and are intentionally excluded from Git.
"""
from pathlib import Path

from sklearn.datasets import fetch_california_housing


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "California_Housing" / "fetch_california_housing.csv"


def main() -> None:
    frame = fetch_california_housing(as_frame=True).frame
    frame = frame.rename(columns={"MedHouseVal": "target"})
    required = {"Latitude", "Longitude", "MedInc", "HouseAge", "AveRooms",
                "AveBedrms", "Population", "AveOccup", "target"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Downloaded California data miss required fields: {sorted(missing)}")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DEST, index=False)
    print(f"Saved {len(frame)} rows to {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
