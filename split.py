#!/usr/bin/env python3
import os
import math
import pyarrow as pa
import pyarrow.parquet as pq

# =========================
# User settings (edit these)
# =========================
INPUT_FILE = "PdM_telemetry.parquet"
OUTPUT_DIR = "pdm_telemetry_split"
NUM_FILES = 20
# =========================

def main():
    if NUM_FILES < 1:
        raise ValueError("NUM_FILES must be >= 1")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pf = pq.ParquetFile(INPUT_FILE)
    total_rows = pf.metadata.num_rows
    if total_rows == 0:
        raise RuntimeError("Input Parquet has 0 rows")

    rows_per_file = math.ceil(total_rows / NUM_FILES)

    # Stream in manageable batches; tune if you want (lower uses less RAM).
    batch_size = 256_000

    print(f"Input: {INPUT_FILE}")
    print(f"Total rows: {total_rows:,}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Target files: {NUM_FILES}")
    print(f"Rows per file (approx): {rows_per_file:,}")
    print(f"Batch size: {batch_size:,}")

    file_idx = 0
    rows_in_current = 0
    batches = []
    schema = None

    def flush():
        nonlocal file_idx, rows_in_current, batches, schema
        if not batches:
            return
        table = pa.Table.from_batches(batches, schema=schema)
        out_path = os.path.join(OUTPUT_DIR, f"part_{file_idx:05d}.parquet")
        pq.write_table(table, out_path, compression="snappy")
        print(f"Wrote {out_path}  rows={rows_in_current:,}")
        file_idx += 1
        rows_in_current = 0
        batches = []

    for batch in pf.iter_batches(batch_size=batch_size):
        if schema is None:
            schema = batch.schema

        # If adding this batch would exceed the target rows, split the batch.
        batch_rows = batch.num_rows
        offset = 0
        while offset < batch_rows:
            remaining_in_file = rows_per_file - rows_in_current
            take = min(remaining_in_file, batch_rows - offset)

            sliced = batch.slice(offset, take)
            batches.append(sliced)
            rows_in_current += take
            offset += take

            if rows_in_current >= rows_per_file and file_idx < (NUM_FILES - 1):
                flush()

    # Flush whatever remains (last file)
    flush()

    print(f"Done. Created {file_idx} file(s) in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()