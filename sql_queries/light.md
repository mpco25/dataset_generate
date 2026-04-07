# 🟢 Light Queries (Baseline / I/O Bound)

## Assumptions
- All datasets are stored as **Parquet** files in **ADLS Gen2**
- Queries are compatible with **DuckDB**, **Databricks SQL**, and **Spark SQL**
- Example path format:

```sql
abfss://<container>@<account>.dfs.core.windows.net/pdm/
````

Replace `<container>` and `<account>` as needed.

***

These queries are meant to measure **pure scan performance** and **basic aggregation**, with minimal CPU and memory pressure.

***

### 1. Full table scan (row count)

```sql
SELECT COUNT(*)
FROM read_parquet(
  'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_telemetry.parquet'
);
```

**What this tests**

*   Sequential I/O throughput
*   Metadata handling
*   Baseline performance for all engines

***

### 2. Simple predicate filter

```sql
SELECT COUNT(*)
FROM read_parquet(
  'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_telemetry.parquet'
)
WHERE machineID = 1;
```

**What this tests**

*   Predicate pushdown
*   Column pruning
*   Selectivity handling

***

### 3. Single-metric aggregation

```sql
SELECT
  machineID,
  AVG(volt) AS avg_volt
FROM read_parquet(
  'abfss://<container>@<account>.dfs.core.windows.net/pdm/PdM_telemetry.parquet'
)
GROUP BY machineID;
```

**What this tests**

*   Hash aggregation
*   Minimal CPU pressure
*   Typical KPI-style query in PdM dashboards

***

## Notes

These queries represent **scan-heavy baseline workloads with low computational complexity**, useful for evaluating:

- storage scan performance  
- predicate pushdown effectiveness  
- simple aggregation behavior  

They serve as **baseline benchmarks** for comparing:

- Parquet scan efficiency  
- cloud storage latency impact  
- engine behavior under light analytical workloads  
