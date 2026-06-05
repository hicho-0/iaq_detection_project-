# iaq_detection_project-
# Indoor Air Pollution Detection with Ontology-driven Graph Neural Networks

Detecting indoor-air-pollution **anomalies** from sensor data using a **knowledge graph + GNN**, with a
secondary, leakage-clean **prediction comparison** against RNN/LSTM/GRU baselines.

- **Primary objective (the headline): detection.** An ontology-driven graph autoencoder learns what
  "normal" indoor air looks like; windows it cannot reconstruct are flagged as anomalies. Reported as
  the **taux** (event-window detection rate), ROC/PR-AUC, per-event recall, and a per-variable
  localization of *what* drove each detection.
- **Secondary objective (controlled comparison): prediction.** A leakage-clean pollutant regression,
  GNN (GCN / GIN / SAT-Graph) vs sequential baselines (RNN / LSTM / GRU), scored with R² / MAE / RMSE
  and positioned against the earlier "Xia" work under the **same train/val/test protocol**.

The two result types live on **separate axes** and are never merged into one score.

---

## Dataset

[2023 Indoor Air Quality Dataset — Germany](https://www.kaggle.com/datasets/welfposer/2023-indoor-air-quality-dataset-germany)
(FH-Aachen). Two non-overlapping deployments, 31 columns each, ~2-minute cadence:

| File | Period | Site | Known event |
|---|---|---|---|
| `laboratory.csv` (51,186 rows) | 2023-03-22 → 06-06 | Lab | Apr-17 power outage (data gap) · May-21 fire ~9 km away (subtle NO₂↑) |
| `one_room_apartement.csv` (12,446 rows) | 2023-06-24 → 07-11 | Apartment | Jul-9 humidity-induced dust measurement error (strong PM2.5 spike) |

Each anomaly appears in exactly one file. Download both CSVs and drop them in `./data/`.

---

## Pipeline (run in order)

| # | Notebook | Role | Key output |
|---|---|---|---|
| 0 | `Notebook_0_Data_Preparation_EDA.ipynb` | Clean, 2-min grid, label the 3 events, 5-class states, EDA | `processed/config.json`, cleaned CSVs |
| 1 | `Notebook_1_Ontology_Construction.ipynb` | OWL/RDF ontology (adopts SOSA/SSN + QUDT + OWL-Time) | `processed/ontology/ontology_edges.csv` |
| 2 | `Notebook_2_Graph_Construction.ipynb` | Shared 30-node graph, windowing, leakage-clean splits | `processed/graph/graph_windows.npz` |
| 3 | `Notebook_3_GNN_Detection_Models.ipynb` | GCN/GIN/SAT-Graph **autoencoders** | `*_ae.pt`, `recon_errors.npz` |
| 4 | `Notebook_4_Detection_Evaluation.ipynb` | ROC/PR, localization, **the taux** | `nb04_detection_*` |
| 5 | `Notebook_5_Prediction_Comparison.ipynb` | GNN vs RNN/LSTM/GRU regression | `nb05_pred_*` |
| 6 | `Notebook_6_Final_Comparison.ipynb` | Both stories side by side | `nb06_*` |

---

## How to run

```bash
pip install -r requirements.txt          # see the PyTorch Geometric note below
# put laboratory.csv and one_room_apartement.csv in ./data/
python run_all.py                        # executes Notebook 0 -> 6 in order
```

`run_all.py` writes executed notebooks to `./executed/` and all artifacts to `./processed/`.
You can also open the notebooks and run them top-to-bottom yourself, in numeric order.

### Environment notes
- **Python** 3.10+.
- **Java (JRE 8+)** lets Notebook 1 run the HermiT reasoner for the consistency check. Without Java the
  notebook still completes — `owlrl` materialises the RDFS closure in pure Python.
- **PyTorch Geometric** (Notebooks 3–5) sometimes needs wheels matched to your torch/CUDA build. If
  `pip install torch-geometric` fails, install `torch` first, then follow the official guide:
  <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>.
- Notebooks 0, 1, 2, 6 are torch-free; only 3, 4, 5 require torch.

---

## Ontologies adopted

The graph is built on published standards (adopted **by reference** — real IRIs, no file download), per
the project brief:

- **SOSA/SSN** (W3C/OGC) — each variable is a `sosa:ObservableProperty`, each reading a
  `sosa:Observation`, each row a `sosa:ObservationCollection`, grounded to a `sosa:FeatureOfInterest`.
- **QUDT** — units + physical meaning; 26/30 variables carry a `qudt:hasQuantityKind` and `qudt:unit`
  (e.g. `pm2_5 → MassConcentration → unit:MicroGM-PER-M3`).
- **OWL-Time** — temporal grounding.
- A thin **IAQ domain extension** (`AirQualityState`, `AnomalyEvent`, `Condition`, the relation
  backbone) specialising the standard classes.

The ontology **is** the graph topology: `ontology_edges.csv` is the only source of the GNN's edges.

---

## Integrity guardrails (why the numbers are trustworthy)

- The ontology drives the graph — edges come only from `ontology_edges.csv`, not correlation thresholds.
- No leakage — the prediction inputs exclude same-sensor proxies (particle counts, `TypPS`,
  `dCO2dt`, `health`, `performance`); event windows appear only in the detection test set.
- Same protocol — GNN and every RNN/LSTM/GRU baseline use identical 70/15/15 splits and the same scaler.
- Detection ≠ prediction — taux (reconstruction) and R²/MAE/RMSE (regression) are reported separately.
- Honest PM regression — leakage-clean PM R² is lower than a leaky pipeline would show; that lower
  number is the correct one.

---

## Verification

See **`VERIFICATION_GUIDE.md`** for a per-step checklist: the expected values on the real data, the red
flags, the baseline-protocol guarantee, and a defense Q&A cheat-sheet.

## Baseline positioning (to complete)

The comparison section is built to take the **Xia/Xuanzhe report's reported numbers** (RNN/LSTM/GRU
RMSE / AQI accuracy) as labelled rows beside ours, under the same split. Transcribe those figures from
the report into Notebooks 5/6 — do not invent them. If the report used a different split than 70/15/15,
re-run Notebook 2 with that split so the comparison stays apples-to-apples.
