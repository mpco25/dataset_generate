# 🔴 Heavy & Super-Heavy Queries (CPU + Memory + Join Pressure)

## Assumptions
- All datasets are stored as **Parquet** files in **ADLS Gen2**
- The SQL logic is broadly compatible with **DuckDB**, **Databricks SQL**, and **Spark SQL**
- Files available:

  - `PdM_telemetry.parquet`
  - `PdM_errors.parquet`
  - `PdM_maint.parquet`
  - `PdM_failures.parquet`

- All datasets share `machineID` and compatible date/time fields

Example path format:
```sql
abfss://<container>@<account>.dfs.core.windows.net/pdm/
````

---

These queries represent **real PdM / ML feature-generation workloads**.
They combine **wide aggregations**, **window functions**, **CTEs**, and **multi-way joins**.

---

### 7. Feature table per machine per day

**Complexity:** 🟡 Heavy

```sql
SELECT
  DATE(datetime) AS day,
  machineID,

  AVG(volt)         AS avg_volt,
  AVG(rotate)       AS avg_rotate,
  AVG(pressure)     AS avg_pressure,
  AVG(vibration)    AS avg_vibration,

  STDDEV(volt)      AS std_volt,
  STDDEV(rotate)    AS std_rotate,
  STDDEV(vibration) AS std_vibration,

  MAX(vibration)    AS max_vibration
FROM read_parquet(
  'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_telemetry.parquet'
)
GROUP BY day, machineID;
```

**What this tests**

* Wide aggregations
* Grouping on computed keys
* Increased CPU usage
* Moderate memory pressure
* Classic PdM feature table generation

---

### 8. Z-score–based anomaly detection

**Complexity:** 🔴 Super-Heavy

```sql
WITH stats AS (
  SELECT
    machineID,
    AVG(vibration)    AS mean_vib,
    STDDEV(vibration) AS std_vib
  FROM read_parquet(
    'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_telemetry.parquet'
  )
  GROUP BY machineID
)
SELECT
  t.datetime,
  t.machineID,
  t.vibration,
  (t.vibration - s.mean_vib) / NULLIF(s.std_vib, 0) AS z_score
FROM read_parquet(
  'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_telemetry.parquet'
) t
JOIN stats s
  ON t.machineID = s.machineID
WHERE ABS((t.vibration - s.mean_vib) / NULLIF(s.std_vib, 0)) > 3;
```

**What this tests**

* Derived-statistics joins
* Row-level statistical computation
* Join + aggregation interplay
* Anomaly detection pattern

---

### 9. Long rolling window trend & deviation (7-day window)

**Complexity:** 🔴 Super-Heavy

```sql
WITH base AS (
  SELECT
    datetime,
    machineID,
    vibration,
    AVG(vibration) OVER (
      PARTITION BY machineID
      ORDER BY datetime
      ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
    ) AS vib_7d_avg
  FROM read_parquet(
    'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_telemetry.parquet'
  )
)
SELECT
  datetime,
  machineID,
  vibration,
  vib_7d_avg,
  vibration - vib_7d_avg AS deviation
FROM base
ORDER BY machineID, datetime;
```

**What this tests**

* Large window state per partition
* Ordered processing within partitions
* Sustained memory usage
* Trend and deviation analysis

> Note: This is a **168-row rolling window** (7×24 samples), assuming hourly data.

---

## 🔵 Join-Heavy Super-Heavy Queries

### 10. Telemetry × failures (label enrichment)

**Complexity:** 🔴 Super-Heavy

```sql
WITH failure_daily AS (
  SELECT
    machineID,
    date,
    MAX(failure) AS failure
  FROM read_parquet(
    'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_failures.parquet'
  )
  GROUP BY machineID, date
)
SELECT
  t.datetime,
  t.machineID,
  t.vibration,
  COALESCE(f.failure, 'None') AS failure_type
FROM read_parquet(
  'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_telemetry.parquet'
) t
LEFT JOIN failure_daily f
  ON t.machineID = f.machineID
 AND DATE(t.datetime) = f.date;
```

**What this tests**

* Large outer joins
* Join correctness with pre-aggregation
* Label alignment across datasets
* Join selectivity effects

---

### 11. Full PdM training dataset (multi-way join + aggregation)

**Complexity:** 🔴 Super-Heavy

```sql
WITH errors_daily AS (
  SELECT machineID, date, 1 AS had_error
  FROM read_parquet(
    'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_errors.parquet'
  )
  GROUP BY machineID, date
),
maint_daily AS (
  SELECT machineID, date, 1 AS had_maintenance
  FROM read_parquet(
    'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_maint.parquet'
  )
  GROUP BY machineID, date
),
failures_daily AS (
  SELECT machineID, date, 1 AS had_failure
  FROM read_parquet(
    'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_failures.parquet'
  )
  GROUP BY machineID, date
)
SELECT
  DATE(t.datetime) AS day,
  t.machineID,

  AVG(t.volt)      AS avg_volt,
  AVG(t.rotate)    AS avg_rotate,
  AVG(t.pressure)  AS avg_pressure,
  AVG(t.vibration) AS avg_vibration,

  MAX(COALESCE(e.had_error, 0))        AS had_error,
  MAX(COALESCE(m.had_maintenance, 0))  AS had_maintenance,
  MAX(COALESCE(f.had_failure, 0))      AS had_failure
FROM read_parquet(
  'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_telemetry.parquet'
) t
LEFT JOIN errors_daily   e ON t.machineID = e.machineID AND DATE(t.datetime) = e.date
LEFT JOIN maint_daily    m ON t.machineID = m.machineID AND DATE(t.datetime) = m.date
LEFT JOIN failures_daily f ON t.machineID = f.machineID AND DATE(t.datetime) = f.date
GROUP BY day, t.machineID;
```

**What this tests**

* Multi-way joins with pre-aggregation
* Wide aggregation outputs
* Join correctness under duplication risk
* End-to-end ML training dataset preparation

---

## Notes

* These queries place **substantial pressure on CPU, memory, and join execution**
* They closely resemble:

  * feature engineering pipelines
  * anomaly detection workflows
  * ML dataset preparation
* They are useful for stress-testing:

  * join strategies
  * window execution
  * memory management
  * single-node vs distributed engine behavior
