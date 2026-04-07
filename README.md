# Azure PdM Dataset Augmentation

Synthetic data augmentation for the [Azure AI Predictive Maintenance dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) using a TimeGAN-style generative model. Scales the dataset to an arbitrary target size while preserving temporal patterns, machine correlations, and failure distributions.

---

## How it works

The script trains a TimeGAN architecture — embedding, recovery, generator, and discriminator networks — on 24-hour sliding windows of sensor data from the real telemetry. Once trained, the generator synthesizes new sensor sequences which are decoded back into the original feature space, assigned to new synthetic machine IDs, and streamed to disk in chunks to stay memory-efficient.

Relational tables (`machines`, `failures`, `errors`, `maint`) are replicated for each synthetic machine ID by sampling from real records and jittering timestamps.

The trained model is saved to disk so subsequent runs skip retraining.

---

## Requirements

- Python ≥ 3.10
- [`uv`](https://github.com/astral-sh/uv) (recommended) — dependencies are declared inline and installed automatically

Or install manually:

```bash
pip install pandas numpy scikit-learn torch tqdm pyarrow joblib
```

---

## Usage

```bash
uv run python main.py \
  --input  ./data/kaggle \
  --output ./iot_20GB \
  --target_gb 20.0
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `./archive` | Directory containing the original Kaggle CSVs |
| `--output` | `./augmented` | Output directory for augmented files |
| `--target_gb` | `1.0` | Target dataset size in GB |
| `--format` | `csv` | Output format: `csv` or `parquet` |

### Examples

```bash
# Generate a 20 GB CSV dataset
uv run python main.py --input ./data/kaggle --output ./iot_20GB --target_gb 20.0

# Generate a 20 GB Parquet dataset
uv run python main.py --input ./data/kaggle --output ./iot_20GB_parquet --target_gb 20.0 --format parquet

# Generate a 100 GB CSV dataset
uv run python main.py --input ./data/kaggle --output ./iot_100GB/csv --target_gb 100.0

# Generate a 100 GB Parquet dataset
uv run python main.py --input ./data/kaggle --output ./iot_100GB/parquet --target_gb 100.0 --format parquet

```

---

## Input files

Expected inside the `--input` directory (standard Kaggle archive layout):

```
PdM_telemetry.csv
PdM_machines.csv
PdM_failures.csv
PdM_errors.csv
PdM_maint.csv
```

---

## Output files

```
<output>/
├── PdM_telemetry.csv   # (or .parquet) — original + synthetic sensor readings
├── PdM_machines.csv    # original + synthetic machine metadata
├── PdM_failures.csv    # original + jittered failure records
├── PdM_errors.csv      # original + jittered error records
└── PdM_maint.csv       # original + jittered maintenance records
```

A `timegan_model.pt` and `timegan_model_scaler.pkl` are saved to the `--input` directory after the first training run and reused automatically on subsequent runs.

---

## Model architecture

| Component | Type | Role |
|---|---|---|
| Embedder | GRU | Maps real sequences → latent space |
| Recovery | GRU | Reconstructs sequences from latent space |
| Generator | GRU | Produces fake latent sequences from noise |
| Discriminator | Bidirectional GRU | Distinguishes real from fake latent sequences |

Training runs in two phases: embedding pre-training (MSE reconstruction loss), then adversarial training (BCE loss). Default: 150 embedding epochs + 300 adversarial epochs.

---

## Hardware

The script auto-selects the best available device:

```
MPS (Apple Silicon) > CUDA (NVIDIA GPU) > CPU
```

Training on CPU is functional but slow for large epoch counts. For 20 GB+ targets, a GPU is recommended.

---

## Benchmarks

Runs on Apple Silicon (MPS) with the saved model (training skipped). Source dataset: 876,100 telemetry rows.

| Target | Format | Output rows | Actual size on disk | Generation time |
|---|---|---|---|---|
| 2 GB | CSV | 23,475,868 | 2.12 GB | ~1 min 11 s |
| 2 GB | Parquet | 23,475,868 | 0.49 GB | ~25 s |
| 20 GB | CSV | 234,758,644 | 21.30 GB | ~12 min 47 s |
| 20 GB | Parquet | 234,758,644 | 5.04 GB | ~4 min 27 s |
| 100 GB | CSV | 1,173,793,228 | 106.87 GB | ~1 h 44 min |
| 100 GB | Parquet | 1,173,793,228 | 25.53 GB | ~22 min 19 s |

**Throughput:** Parquet generation runs at ~71 it/s vs ~25 it/s for CSV, and compresses to roughly 25% of the equivalent CSV size — strongly preferred for large targets.

> Actual disk size will slightly exceed the `--target_gb` value because the original telemetry is always written first and the target is calculated from row estimates, not byte-exact generation.

---

## SQL Query Examples

Sample queries against the augmented dataset, ordered from simple to complex. Assumes tables are loaded into a SQL-capable engine (e.g. DuckDB, PostgreSQL, SQLite).

---

### 🟢 Light — Basic Lookups

**List all machines and their model/age:**
```sql
SELECT machineID, model, age
FROM PdM_machines
ORDER BY machineID;
```

**Get the latest telemetry reading per machine:**
```sql
SELECT machineID, MAX(datetime) AS last_reading
FROM PdM_telemetry
GROUP BY machineID
ORDER BY machineID;
```

**Count total failures by failure type:**
```sql
SELECT failure, COUNT(*) AS total
FROM PdM_failures
GROUP BY failure
ORDER BY total DESC;
```

---

### 🟡 Medium — Joins & Aggregations

**Average sensor readings per machine model:**
```sql
SELECT m.model,
       ROUND(AVG(t.volt), 2)     AS avg_volt,
       ROUND(AVG(t.rotate), 2)   AS avg_rotate,
       ROUND(AVG(t.pressure), 2) AS avg_pressure,
       ROUND(AVG(t.vibration), 2) AS avg_vibration
FROM PdM_telemetry t
JOIN PdM_machines m ON t.machineID = m.machineID
GROUP BY m.model
ORDER BY m.model;
```

**Machines with the most errors in the last 30 days:**
```sql
SELECT e.machineID, e.errorID, COUNT(*) AS error_count
FROM PdM_errors e
WHERE e.datetime >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY e.machineID, e.errorID
ORDER BY error_count DESC
LIMIT 20;
```

**Failure rate by machine age bucket:**
```sql
SELECT
    CASE
        WHEN m.age < 5  THEN '0–4 yrs'
        WHEN m.age < 10 THEN '5–9 yrs'
        ELSE                  '10+ yrs'
    END AS age_group,
    COUNT(DISTINCT f.machineID) AS machines_with_failures,
    COUNT(f.failure)            AS total_failures
FROM PdM_machines m
LEFT JOIN PdM_failures f ON m.machineID = f.machineID
GROUP BY age_group
ORDER BY age_group;
```

---

### 🔴 Heavy — Window Functions & CTEs

**Rolling 24-hour average vibration per machine (anomaly baseline):**
```sql
SELECT
    machineID,
    datetime,
    vibration,
    AVG(vibration) OVER (
        PARTITION BY machineID
        ORDER BY datetime
        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
    ) AS rolling_24h_avg_vibration
FROM PdM_telemetry
ORDER BY machineID, datetime;
```

**Time between last maintenance and next failure per machine:**
```sql
WITH last_maint AS (
    SELECT machineID, MAX(datetime) AS last_maint_date
    FROM PdM_maint
    GROUP BY machineID
),
next_failure AS (
    SELECT machineID, MIN(datetime) AS first_failure_date
    FROM PdM_failures
    GROUP BY machineID
)
SELECT
    m.machineID,
    lm.last_maint_date,
    nf.first_failure_date,
    DATEDIFF('day', lm.last_maint_date, nf.first_failure_date) AS days_to_failure
FROM PdM_machines m
LEFT JOIN last_maint  lm ON m.machineID = lm.machineID
LEFT JOIN next_failure nf ON m.machineID = nf.machineID
WHERE lm.last_maint_date IS NOT NULL
  AND nf.first_failure_date > lm.last_maint_date
ORDER BY days_to_failure ASC;
```

**Top 10% highest-vibration readings with machine context (percentile ranking):**
```sql
WITH ranked AS (
    SELECT
        t.machineID,
        t.datetime,
        t.vibration,
        m.model,
        m.age,
        PERCENT_RANK() OVER (ORDER BY t.vibration) AS pct_rank
    FROM PdM_telemetry t
    JOIN PdM_machines m ON t.machineID = m.machineID
)
SELECT machineID, datetime, vibration, model, age
FROM ranked
WHERE pct_rank >= 0.90
ORDER BY vibration DESC
LIMIT 100;
```

---

## Configuration

Key hyperparameters are defined at the top of `main.py`:

```python
SEQ_LEN    = 24      # Sliding window length (hours)
HIDDEN_DIM = 24      # GRU hidden size
NUM_LAYERS = 3       # GRU depth
BATCH_SIZE = 128
EPOCHS     = 300
NOISE_STD  = 0.01    # Gaussian jitter added to generated values
```
