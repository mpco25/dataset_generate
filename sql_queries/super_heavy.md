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

#### DuckDB

```sql
.mode table
.maxrows 1000000
.maxwidth 0

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
GROUP BY day, machineID
ORDER BY machineID, day;
```
If the result is still too large for the terminal to be practical, export it instead:
```sql
COPY (
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
  GROUP BY day, machineID
  ORDER BY machineID, day
) TO 'output.csv' (HEADER, DELIMITER ',');
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


### 12. Monthly fleet risk summary (multi-stage aggregation + joins + windowing)

**Complexity:** 🔴 Super-Heavy

```sql
WITH telemetry_daily AS (
  SELECT
    DATE(datetime) AS day,
    machineID,
    AVG(volt)         AS avg_volt,
    AVG(rotate)       AS avg_rotate,
    AVG(pressure)     AS avg_pressure,
    AVG(vibration)    AS avg_vibration,
    STDDEV(vibration) AS std_vibration,
    MAX(vibration)    AS max_vibration,
    COUNT(*)          AS telemetry_rows
  FROM "iot-587_mil_rows-parquet".main.pdm_telemetry
  WHERE datetime >= TIMESTAMP '2016-01-01 00:00:00'
    AND datetime <  TIMESTAMP '2017-01-01 00:00:00'
  GROUP BY DATE(datetime), machineID
),

errors_daily AS (
  SELECT
    DATE(datetime) AS day,
    machineID,
    COUNT(*) AS error_count
  FROM "iot-587_mil_rows-parquet".main.pdm_errors
  WHERE datetime >= TIMESTAMP '2016-01-01 00:00:00'
    AND datetime <  TIMESTAMP '2017-01-01 00:00:00'
  GROUP BY DATE(datetime), machineID
),

maint_daily AS (
  SELECT
    DATE(datetime) AS day,
    machineID,
    COUNT(*) AS maint_count
  FROM "iot-587_mil_rows-parquet".main.pdm_maint
  WHERE datetime >= TIMESTAMP '2016-01-01 00:00:00'
    AND datetime <  TIMESTAMP '2017-01-01 00:00:00'
  GROUP BY DATE(datetime), machineID
),

failures_daily AS (
  SELECT
    DATE(datetime) AS day,
    machineID,
    COUNT(*) AS failure_count
  FROM "iot-587_mil_rows-parquet".main.pdm_failures
  WHERE datetime >= TIMESTAMP '2016-01-01 00:00:00'
    AND datetime <  TIMESTAMP '2017-01-01 00:00:00'
  GROUP BY DATE(datetime), machineID
),

machine_month AS (
  SELECT
    DATE_TRUNC('month', t.day) AS month,
    t.machineID,
    m.model,
    m.age,

    AVG(t.avg_volt)         AS avg_volt,
    AVG(t.avg_rotate)       AS avg_rotate,
    AVG(t.avg_pressure)     AS avg_pressure,
    AVG(t.avg_vibration)    AS avg_vibration,
    AVG(t.std_vibration)    AS avg_daily_std_vibration,
    MAX(t.max_vibration)    AS month_max_vibration,
    SUM(t.telemetry_rows)   AS telemetry_rows,

    SUM(COALESCE(e.error_count, 0))   AS total_errors,
    SUM(COALESCE(md.maint_count, 0))  AS total_maintenance,
    SUM(COALESCE(f.failure_count, 0)) AS total_failures,

    MAX(CASE WHEN COALESCE(f.failure_count, 0) > 0 THEN 1 ELSE 0 END) AS had_failure
  FROM telemetry_daily t
  LEFT JOIN errors_daily e
    ON t.machineID = e.machineID
   AND t.day = e.day
  LEFT JOIN maint_daily md
    ON t.machineID = md.machineID
   AND t.day = md.day
  LEFT JOIN failures_daily f
    ON t.machineID = f.machineID
   AND t.day = f.day
  LEFT JOIN "iot-587_mil_rows-parquet".main.pdm_machines m
    ON t.machineID = m.machineID
  GROUP BY
    DATE_TRUNC('month', t.day),
    t.machineID,
    m.model,
    m.age
),

WITH telemetry_daily AS (
  SELECT
    DATE(datetime) AS day,
    machineID,
    AVG(volt)         AS avg_volt,
    AVG(rotate)       AS avg_rotate,
    AVG(pressure)     AS avg_pressure,
    AVG(vibration)    AS avg_vibration,
    STDDEV(vibration) AS std_vibration,
    MAX(vibration)    AS max_vibration,
    COUNT(*)          AS telemetry_rows
  FROM "iot-587_mil_rows-parquet".main.pdm_telemetry
  WHERE datetime >= TIMESTAMP '2016-01-01 00:00:00'
    AND datetime <  TIMESTAMP '2017-01-01 00:00:00'
  GROUP BY DATE(datetime), machineID
),

errors_daily AS (
  SELECT
    DATE(datetime) AS day,
    machineID,
    COUNT(*) AS error_count
  FROM "iot-587_mil_rows-parquet".main.pdm_errors
  WHERE datetime >= TIMESTAMP '2016-01-01 00:00:00'
    AND datetime <  TIMESTAMP '2017-01-01 00:00:00'
  GROUP BY DATE(datetime), machineID
),

maint_daily AS (
  SELECT
    DATE(datetime) AS day,
    machineID,
    COUNT(*) AS maint_count
  FROM "iot-587_mil_rows-parquet".main.pdm_maint
  WHERE datetime >= TIMESTAMP '2016-01-01 00:00:00'
    AND datetime <  TIMESTAMP '2017-01-01 00:00:00'
  GROUP BY DATE(datetime), machineID
),

failures_daily AS (
  SELECT
    DATE(datetime) AS day,
    machineID,
    COUNT(*) AS failure_count
  FROM "iot-587_mil_rows-parquet".main.pdm_failures
  WHERE datetime >= TIMESTAMP '2016-01-01 00:00:00'
    AND datetime <  TIMESTAMP '2017-01-01 00:00:00'
  GROUP BY DATE(datetime), machineID
),

machine_month AS (
  SELECT
    DATE_TRUNC('month', t.day) AS month,
    t.machineID,
    m.model,
    m.age,

    AVG(t.avg_volt)         AS avg_volt,
    AVG(t.avg_rotate)       AS avg_rotate,
    AVG(t.avg_pressure)     AS avg_pressure,
    AVG(t.avg_vibration)    AS avg_vibration,
    AVG(t.std_vibration)    AS avg_daily_std_vibration,
    MAX(t.max_vibration)    AS month_max_vibration,
    SUM(t.telemetry_rows)   AS telemetry_rows,

    SUM(COALESCE(e.error_count, 0))   AS total_errors,
    SUM(COALESCE(md.maint_count, 0))  AS total_maintenance,
    SUM(COALESCE(f.failure_count, 0)) AS total_failures,

    MAX(CASE WHEN COALESCE(f.failure_count, 0) > 0 THEN 1 ELSE 0 END) AS had_failure
  FROM telemetry_daily t
  LEFT JOIN errors_daily e
    ON t.machineID = e.machineID
   AND t.day = e.day
  LEFT JOIN maint_daily md
    ON t.machineID = md.machineID
   AND t.day = md.day
  LEFT JOIN failures_daily f
    ON t.machineID = f.machineID
   AND t.day = f.day
  LEFT JOIN "iot-587_mil_rows-parquet".main.pdm_machines m
    ON t.machineID = m.machineID
  GROUP BY
    DATE_TRUNC('month', t.day),
    t.machineID,
    m.model,
    m.age
),

scored AS (
  SELECT
    *,
    AVG(total_errors) OVER (
      PARTITION BY machineID
      ORDER BY month
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3m_avg_errors,

    AVG(total_failures) OVER (
      PARTITION BY machineID
      ORDER BY month
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3m_avg_failures,

    RANK() OVER (
      PARTITION BY month
      ORDER BY total_failures DESC, total_errors DESC, month_max_vibration DESC
    ) AS monthly_risk_rank
  FROM machine_month
)

SELECT
  month,
  COUNT(*) AS machines_in_month,
  SUM(telemetry_rows) AS telemetry_rows,
  SUM(total_errors) AS total_errors,
  SUM(total_maintenance) AS total_maintenance,
  SUM(total_failures) AS total_failures,
  AVG(avg_volt) AS fleet_avg_volt,
  AVG(avg_rotate) AS fleet_avg_rotate,
  AVG(avg_pressure) AS fleet_avg_pressure,
  AVG(avg_vibration) AS fleet_avg_vibration,
  MAX(month_max_vibration) AS fleet_max_vibration,
  AVG(age) AS avg_machine_age,
  SUM(CASE WHEN monthly_risk_rank <= 10 THEN 1 ELSE 0 END) AS top_10_risky_machines,
  AVG(rolling_3m_avg_errors) AS fleet_rolling_3m_avg_errors,
  AVG(rolling_3m_avg_failures) AS fleet_rolling_3m_avg_failures
FROM scored
GROUP BY month
ORDER BY month;
scored AS (
  SELECT
    *,
    AVG(total_errors) OVER (
      PARTITION BY machineID
      ORDER BY month
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3m_avg_errors,

    AVG(total_failures) OVER (
      PARTITION BY machineID
      ORDER BY month
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3m_avg_failures,

    RANK() OVER (
      PARTITION BY month
      ORDER BY total_failures DESC, total_errors DESC, month_max_vibration DESC
    ) AS monthly_risk_rank
  FROM machine_month
)

SELECT
  month,
  COUNT(*) AS machines_in_month,
  SUM(telemetry_rows) AS telemetry_rows,
  SUM(total_errors) AS total_errors,
  SUM(total_maintenance) AS total_maintenance,
  SUM(total_failures) AS total_failures,
  AVG(avg_volt) AS fleet_avg_volt,
  AVG(avg_rotate) AS fleet_avg_rotate,
  AVG(avg_pressure) AS fleet_avg_pressure,
  AVG(avg_vibration) AS fleet_avg_vibration,
  MAX(month_max_vibration) AS fleet_max_vibration,
  AVG(age) AS avg_machine_age,
  SUM(CASE WHEN monthly_risk_rank <= 10 THEN 1 ELSE 0 END) AS top_10_risky_machines,
  AVG(rolling_3m_avg_errors) AS fleet_rolling_3m_avg_errors,
  AVG(rolling_3m_avg_failures) AS fleet_rolling_3m_avg_failures
FROM scored
GROUP BY month
ORDER BY month;
```

**What this tests**

* Multi-stage aggregation over large telemetry volumes
* Multi-way joins with operational event tables and machine metadata
* Rolling-window analytics and per-month risk ranking
* Heavy intermediate computation with minimal final UI output


**Why it qualifies as super-heavy**

* it scans the large pdm_telemetry table
* it aggregates at daily machine level first
* it joins 4 other datasets
* it aggregates again at monthly machine level
* it applies window functions
* it applies ranking
* only then does it collapse to a tiny monthly summary

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
