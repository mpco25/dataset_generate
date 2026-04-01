uv run python main.py --input ./kaggle --output ./iot_100GB/csv --target_gb 100.0       
🖥  Device: mps
📂 Loading CSVs...
   Telemetry rows: 876,100
📊 Current rows: 876,100 → Target rows: 1,173,793,204
   Sequences to generate: 48,871,547
   Training sequences from machine 1: 8,737
⚡ Found saved model — skipping training...
🔗 Augmenting relational tables...
💾 Streaming telemetry to disk in chunks (CSV)...
  Generating: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 95453/95453 [1:44:55<00:00, 15.16it/s]

✅ Done!
   Output directory : ./iot_100GB/csv
   Telemetry rows   : 1,173,793,228
   Total size on disk: 106.87 GB


uv run python main.py --input ./kaggle --output ./iot_100GB/parquet --target_gb 100.0 --format parquet
🖥  Device: mps
📂 Loading CSVs...
   Telemetry rows: 876,100
📊 Current rows: 876,100 → Target rows: 1,173,793,204
   Sequences to generate: 48,871,547
   Training sequences from machine 1: 8,737
⚡ Found saved model — skipping training...
🔗 Augmenting relational tables...
💾 Streaming telemetry to disk in chunks (PARQUET)...
  Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 95453/95453 [22:19<00:00, 71.28it/s]

✅ Done!
   Output directory : ./iot_100GB/parquet
   Telemetry rows   : 1,173,793,228
   Total size on disk: 25.53 GB
   

uv run python main.py --input ./kaggle --output ./iot_50GB_parquet --target_gb 50.0 --format parquet
🖥  Device: mps
📂 Loading CSVs...
   Telemetry rows: 876,100
📊 Current rows: 876,100 → Target rows: 586,896,602
   Sequences to generate: 24,417,521
   Training sequences from machine 1: 8,737
⚡ Found saved model — skipping training...
🔗 Augmenting relational tables...
💾 Streaming telemetry to disk in chunks (PARQUET)...
  Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 47691/47691 [11:08<00:00, 71.30it/s]

✅ Done!
   Output directory : ./iot_50GB_parquet
   Telemetry rows   : 586,896,604
   Total size on disk: 12.17 GB