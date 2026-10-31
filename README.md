This the implementation of  **Identity-to-Context Adaptation for Streaming Spatio-Temporal Forecasting on Evolving Networks**.

---

## 1. Environment

The experiments were run with the following setup:

- Python: 3.10.19
- CUDA: 13.0
- PyTorch: 2.9.0+cu130
- scipy: 1.13.1
- pandas: 2.1.4
- numpy: 1.26.1
- scikit-learn: 1.6.1
- einops: 0.8.0
- tables: 3.9.2*

---

## 2. Data Preparation

Please place the **raw data** of each dataset into the corresponding folder under `raw_data/`.

Expected structure (illustrative):

```text
raw_data/
  AQI/
    AQI_stages.csv
    AQI_station_info.csv
  METR-LA/
    METR-LA_station_info.csv
    metr-la.h5
  EPeMS/
    EPeMS_station_info.csv
    gla_pemsd7.csv
  GBA/
    gba_his_2019.h5
    GBA_station_info.csv
  UrbanEV_s/
    adj.csv
    distance.csv
    UrbanEV_station_info.csv
    volume-11kw.csv
```

### Included in this repo

* The **AQI dataset** used in our experiments is already included. It derived from [https://github.com/Sam1224/KITS](https://github.com/Sam1224/KITS)
* **Sensor metadata** (sensor information) for all datasets is already included.

### Datasets to download externally

For the remaining datasets, please download from the following public repositories and then place the raw files into `raw_data/<DATASET_NAME>/` accordingly:

* METR-LA: [https://github.com/liyaguang/DCRNN](https://github.com/liyaguang/DCRNN), `metr-la.h5`
* EPeMS: [https://github.com/mb-Ma/STEV](https://github.com/mb-Ma/STEV), `gla_pemsd7.csv`
* GBA: [https://github.com/liuxu77/LargeST](https://github.com/liuxu77/LargeST), `gba_his_2019.h5`
* UrbanEV: [https://github.com/IntelligentSystemsLab/UrbanEV](https://github.com/IntelligentSystemsLab/UrbanEV), `UrbanEVDataset/20220901-20230228_zone-cleaned-aggregated/charge_1hour/volume-11kw.csv`

> Note: Each upstream repository may use different filenames and formats. After downloading, keep the original filenames if possible, and place them under the matching `raw_data` subfolder.

---

## 3. Reproducibility

We provide default configurations and arguments for reproducing the main results.

To run the full pipeline:

```bash
bash run.sh
```

`run.sh` is configured to use the default settings defined in this repo, including dataset configurations and model hyperparameters.

---
