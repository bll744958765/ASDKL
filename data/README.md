# External data

The synthetic data are generated locally by `common/synthetic_dataset.py`. Download the three third-party datasets from their providers and place them at these paths (the loader lists the required columns):

- `data/EPA_PM25_2023/raw/annual_conc_by_monitor_2023.csv` — U.S. EPA AirData 2023 annual concentration archive
- `data/LUCAS 2018 TopSoil Dataset/OriginalSoilDataset.csv` — ESDAC/JRC LUCAS 2018 topsoil
- `data/California_Housing/fetch_california_housing.csv` — California housing input table

Data are not redistributed in this code repository. Training, validation and test preprocessing is calculated from training records only.
