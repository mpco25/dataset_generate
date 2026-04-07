````md
# 🟡 Analytical Queries (Higher CPU + Memory Pressure)

## Assumptions
- All datasets are stored as **Parquet** files in **ADLS Gen2**
- The SQL logic is compatible with **DuckDB**, **Databricks SQL**, and **Spark SQL**
- Example path format:

```sql
abfss://<container>@<account>.dfs.core.windows.net/pdm/
````

Replace `<container>` and `<account>` as needed.

---

These queries introduce **grouping on derived columns**, **sorting**, and **window functions**.
They are representative of **feature engineering and analytical workloads** in PdM pipelines.

---

### 4. Time-based aggregation (hour → day)

```sql
SELECT
  DATE(datetime) AS day,
  machineID,
  AVG(pressure) AS avg_pressure
FROM read_parquet(
  'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_telemetry.parquet'
)
GROUP BY day, machineID
ORDER BY day, machineID;
```

**What this tests**

* Timestamp projection
* Grouping on computed fields
* Sort cost
* Typical daily aggregation workload

---

### 5. Rolling window (24-row average per machine)

```sql
SELECT
  datetime,
  machineID,
  volt,
  AVG(volt) OVER (
    PARTITION BY machineID
    ORDER BY datetime
    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
  ) AS volt_rolling_avg
FROM read_parquet(
  'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_telemetry.parquet'
);
```

**What this tests**

* Window function execution
* State retention per partition
* Ordered processing within partitions
* Higher computational cost than simple aggregation

> Note: This represents a **24-row rolling average**. It corresponds to a true 24-hour window only if data is sampled at one row per hour per machine.

---

### 6. Multi-metric aggregation per machine

```sql
SELECT
  machineID,
  AVG(volt)        AS avg_volt,
  STDDEV(volt)     AS std_volt,
  MAX(vibration)   AS max_vibration,
  AVG(pressure)    AS avg_pressure
FROM read_parquet(
  'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_telemetry.parquet'
)
GROUP BY machineID;
```

**What this tests**

* Multiple aggregates in a single pass
* Increased CPU usage
* Intermediate state and memory pressure
* Realistic PdM feature extraction step

---

## Notes

* These queries place greater pressure on **CPU and memory** than simple scan-oriented baselines
* They are representative of:

  * feature engineering
  * trend calculation
  * rolling statistics
* They are useful for comparing how engines handle:

  * computed grouping keys
  * sorting
  * window functions
  * multi-metric aggregations

```
```
