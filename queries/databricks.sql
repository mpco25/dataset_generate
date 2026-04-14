spark.conf.set(
    "fs.azure.account.key.stdatasetsdb02.blob.core.windows.net",
    "<REDACTED>"
)

spark.read.parquet(
    "wasbs://iot@stdatasetsdb02.blob.core.windows.net/587_million_rows/parquet/PdM_telemetry.parquet"
).createOrReplaceTempView("pdm_telemetry")

spark.read.parquet(
    "wasbs://iot@stdatasetsdb02.blob.core.windows.net/587_million_rows/parquet/PdM_errors.parquet"
).createOrReplaceTempView("pdm_errors")

spark.read.parquet(
    "wasbs://iot@stdatasetsdb02.blob.core.windows.net/587_million_rows/parquet/PdM_maint.parquet"
).createOrReplaceTempView("pdm_maint")

spark.read.parquet(
    "wasbs://iot@stdatasetsdb02.blob.core.windows.net/587_million_rows/parquet/PdM_failures.parquet"
).createOrReplaceTempView("pdm_failures")

spark.read.parquet(
    "wasbs://iot@stdatasetsdb02.blob.core.windows.net/587_million_rows/parquet/PdM_machines.parquet"
).createOrReplaceTempView("pdm_machines")

%sql
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
  FROM pdm_telemetry
  WHERE datetime >= TIMESTAMP '2016-01-01 00:00:00'
    AND datetime <  TIMESTAMP '2017-01-01 00:00:00'
  GROUP BY DATE(datetime), machineID
),

errors_daily AS (
  SELECT
    DATE(datetime) AS day,
    machineID,
    COUNT(*) AS error_count
  FROM pdm_errors
  WHERE datetime >= TIMESTAMP '2016-01-01 00:00:00'
    AND datetime <  TIMESTAMP '2017-01-01 00:00:00'
  GROUP BY DATE(datetime), machineID
),

maint_daily AS (
  SELECT
    DATE(datetime) AS day,
    machineID,
    COUNT(*) AS maint_count
  FROM pdm_maint
  WHERE datetime >= TIMESTAMP '2016-01-01 00:00:00'
    AND datetime <  TIMESTAMP '2017-01-01 00:00:00'
  GROUP BY DATE(datetime), machineID
),

failures_daily AS (
  SELECT
    DATE(datetime) AS day,
    machineID,
    COUNT(*) AS failure_count
  FROM pdm_failures
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
  LEFT JOIN pdm_machines m
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
