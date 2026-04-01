# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas",
#   "numpy",
#   "scikit-learn",
#   "torch",
#   "tqdm",
# ]
# ///

"""
Azure PdM Dataset Augmentation Script
Uses TimeGAN-style synthetic generation to expand dataset to ~1GB
Preserves temporal patterns, machine correlations, and failure distributions.

Usage:
    pip install pandas numpy scikit-learn torch tqdm
    python augment_pdm.py --input ./archive --output ./augmented --target_gb 1.0
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import pyarrow as pa
import pyarrow.parquet as pq

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
TARGET_GB       = 1.0
SEQ_LEN         = 24       # 24-hour windows
HIDDEN_DIM      = 24
NUM_LAYERS      = 3
BATCH_SIZE      = 128
EPOCHS          = 300
NOISE_STD       = 0.01     # small noise for diversity

SENSOR_COLS     = ["volt", "rotate", "pressure", "vibration"]
DATETIME_COL    = "datetime"
MACHINE_COL     = "machineID"

# ─────────────────────────────────────────────
# 2. TIMEGAN COMPONENTS
# ─────────────────────────────────────────────

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        h, _ = self.rnn(x)
        return torch.sigmoid(self.fc(h))


class RecoveryNetwork(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        r, _ = self.rnn(h)
        return torch.sigmoid(self.fc(r))


class GeneratorNetwork(nn.Module):
    def __init__(self, noise_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(noise_dim, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z):
        e, _ = self.rnn(z)
        return torch.sigmoid(self.fc(e))


class DiscriminatorNetwork(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers,
                          batch_first=True, bidirectional=True)
        self.fc  = nn.Linear(hidden_dim * 2, 1)

    def forward(self, h):
        d, _ = self.rnn(h)
        return self.fc(d)


# ─────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────

def load_data(input_dir):
    print("📂 Loading CSVs...")
    telemetry = pd.read_csv(os.path.join(input_dir, "PdM_telemetry.csv"),
                            parse_dates=[DATETIME_COL])
    machines  = pd.read_csv(os.path.join(input_dir, "PdM_machines.csv"))
    failures  = pd.read_csv(os.path.join(input_dir, "PdM_failures.csv"),
                            parse_dates=[DATETIME_COL])
    errors    = pd.read_csv(os.path.join(input_dir, "PdM_errors.csv"),
                            parse_dates=[DATETIME_COL])
    maint     = pd.read_csv(os.path.join(input_dir, "PdM_maint.csv"),
                            parse_dates=[DATETIME_COL])
    print(f"   Telemetry rows: {len(telemetry):,}")
    return telemetry, machines, failures, errors, maint


def make_sequences(df, seq_len):
    """Convert time-series DataFrame to overlapping windows."""
    data = df[SENSOR_COLS].values
    seqs = []
    for i in range(len(data) - seq_len):
        seqs.append(data[i: i + seq_len])
    return np.array(seqs, dtype=np.float32)


def train_timegan(sequences, device, epochs=EPOCHS):
    """Simplified TimeGAN training loop (embedding + adversarial phases)."""
    input_dim  = sequences.shape[2]
    scaler     = MinMaxScaler()
    flat       = sequences.reshape(-1, input_dim)
    scaler.fit(flat)
    seqs_scaled = scaler.transform(flat).reshape(sequences.shape)
    X = torch.tensor(seqs_scaled).to(device)

    embedder   = EmbeddingNetwork(input_dim,  HIDDEN_DIM, NUM_LAYERS).to(device)
    recovery   = RecoveryNetwork(HIDDEN_DIM,  input_dim,  NUM_LAYERS).to(device)
    generator  = GeneratorNetwork(input_dim,  HIDDEN_DIM, NUM_LAYERS).to(device)
    discriminator = DiscriminatorNetwork(HIDDEN_DIM, NUM_LAYERS).to(device)

    opt_er  = torch.optim.Adam(list(embedder.parameters()) +
                               list(recovery.parameters()), lr=1e-3)
    opt_g   = torch.optim.Adam(generator.parameters(),     lr=1e-3)
    opt_d   = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    print("🧠 Training TimeGAN...")

    # Phase 1 — Embedding pre-training
    for epoch in tqdm(range(epochs // 2), desc="  Embedding"):
        idx = np.random.randint(0, len(X), BATCH_SIZE)
        x_batch = X[idx]
        h = embedder(x_batch)
        x_hat = recovery(h)
        loss = mse(x_hat, x_batch)
        opt_er.zero_grad(); loss.backward(); opt_er.step()

    # Phase 2 — Adversarial training
    for epoch in tqdm(range(epochs), desc="  Adversarial"):
        idx  = np.random.randint(0, len(X), BATCH_SIZE)
        x_b  = X[idx]
        z    = torch.randn(BATCH_SIZE, SEQ_LEN, input_dim).to(device)

        # Generator step
        h_fake = generator(z)
        d_fake = discriminator(h_fake)
        g_loss = bce(d_fake, torch.ones_like(d_fake))
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()

        # Discriminator step
        h_real = embedder(x_b).detach()
        h_fake = generator(z).detach()
        d_real = discriminator(h_real)
        d_fake = discriminator(h_fake)
        d_loss = bce(d_real, torch.ones_like(d_real)) + \
                 bce(d_fake, torch.zeros_like(d_fake))
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()

    return generator, recovery, scaler, input_dim


def generate_synthetic_telemetry(generator, recovery, scaler,
                                 input_dim, n_sequences, device,
                                 chunk_size=50000):
    """Generate synthetic windows in chunks to avoid OOM on large targets."""
    generator.eval(); recovery.eval()
    batch = 512
    with torch.no_grad():
        chunk = []
        chunk_rows = 0
        for i in tqdm(range(0, n_sequences, batch), desc="  Generating"):
            b = min(batch, n_sequences - i)
            z = torch.randn(b, SEQ_LEN, input_dim).to(device)
            h = generator(z)
            x = recovery(h).cpu().numpy()
            flat = x.reshape(-1, input_dim)
            flat = scaler.inverse_transform(flat)
            flat = np.clip(flat, 0, None)
            flat += np.random.normal(0, NOISE_STD, flat.shape)
            chunk.append(flat)
            chunk_rows += len(flat)
            if chunk_rows >= chunk_size:
                yield np.vstack(chunk)
                chunk = []
                chunk_rows = 0
        if chunk:
            yield np.vstack(chunk)


def augment_relational_tables(machines, failures, errors, maint,
                               synthetic_machine_ids):
    """
    Replicate relational tables for synthetic machine IDs,
    jittering timestamps and preserving failure/error distributions.
    """
    def jitter_df(df, new_ids, id_col=MACHINE_COL):
        frames = []
        real_ids = df[id_col].unique()
        for mid in new_ids:
            src = df[df[id_col] == np.random.choice(real_ids)].copy()
            src[id_col] = mid
            if DATETIME_COL in src.columns:
                offset = pd.Timedelta(days=np.random.randint(1, 365))
                src[DATETIME_COL] = src[DATETIME_COL] + offset
            frames.append(src)
        return pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0]

    aug_machines = pd.DataFrame({
        "machineID": synthetic_machine_ids,
        "model":     np.random.choice(machines["model"].values,
                                      len(synthetic_machine_ids)),
        "age":       np.random.randint(machines["age"].min(),
                                       machines["age"].max() + 1,
                                       len(synthetic_machine_ids))
    })
    aug_failures = jitter_df(failures, synthetic_machine_ids)
    aug_errors   = jitter_df(errors,   synthetic_machine_ids)
    aug_maint    = jitter_df(maint,    synthetic_machine_ids)

    return aug_machines, aug_failures, aug_errors, aug_maint


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

def main(input_dir, output_dir, target_gb, fmt="csv"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"🖥  Device: {device}")

    # Load
    telemetry, machines, failures, errors, maint = load_data(input_dir)

    # Scale target to rows
    disk_size       = os.path.getsize(os.path.join(input_dir, "PdM_telemetry.csv"))
    bytes_per_row   = disk_size / len(telemetry)
    target_bytes    = target_gb * 1024 ** 3
    target_rows     = int(target_bytes / bytes_per_row)
    current_rows    = len(telemetry)
    rows_needed     = max(target_rows - current_rows, 0)
    n_seqs_needed   = rows_needed // SEQ_LEN + 1
    print(f"📊 Current rows: {current_rows:,} → Target rows: {target_rows:,}")
    print(f"   Sequences to generate: {n_seqs_needed:,}")

    # Prepare sequences per machine (use machine with most data)
    top_machine = telemetry[MACHINE_COL].value_counts().idxmax()
    machine_df  = telemetry[telemetry[MACHINE_COL] == top_machine].sort_values(DATETIME_COL)
    sequences   = make_sequences(machine_df, SEQ_LEN)
    print(f"   Training sequences from machine {top_machine}: {len(sequences):,}")

    # Train TimeGAN or load saved model
    model_path = os.path.join(input_dir, "timegan_model.pt")
    input_dim  = len(SENSOR_COLS)

    if os.path.exists(model_path):
        print("⚡ Found saved model — skipping training...")
        checkpoint  = torch.load(model_path, map_location=device, weights_only=True)
        scaler_path = model_path.replace(".pt", "_scaler.pkl")
        scaler      = joblib.load(scaler_path)
        generator   = GeneratorNetwork(input_dim, HIDDEN_DIM, NUM_LAYERS).to(device)
        recovery    = RecoveryNetwork(HIDDEN_DIM, input_dim, NUM_LAYERS).to(device)
        generator.load_state_dict(checkpoint["generator"])
        recovery.load_state_dict(checkpoint["recovery"])
    else:
        generator, recovery, scaler, input_dim = train_timegan(sequences, device)
        print("💾 Saving model for future runs...")
        torch.save({
            "generator": generator.state_dict(),
            "recovery":  recovery.state_dict(),
        }, model_path)
        scaler_path = model_path.replace(".pt", "_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"   Model saved to {model_path}")

    # Build synthetic machine IDs
    max_id = telemetry[MACHINE_COL].max()
    n_synth_machines = max(1, rows_needed // current_rows + 1)
    synth_ids = list(range(max_id + 1, max_id + 1 + n_synth_machines))

    # Augment relational tables
    print("🔗 Augmenting relational tables...")
    aug_mach, aug_fail, aug_err, aug_maint = augment_relational_tables(
        machines, failures, errors, maint, synth_ids)

    # Save relational tables
    machines_out  = pd.concat([machines, aug_mach], ignore_index=True)
    failures_out  = pd.concat([failures, aug_fail], ignore_index=True)
    errors_out    = pd.concat([errors,   aug_err],  ignore_index=True)
    maint_out     = pd.concat([maint,    aug_maint], ignore_index=True)
    def save_df(df, name):
        if fmt == "parquet":
            df.to_parquet(os.path.join(output_dir, f"{name}.parquet"), index=False)
        else:
            df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

    save_df(machines_out, "PdM_machines")
    save_df(failures_out, "PdM_failures")
    save_df(errors_out,   "PdM_errors")
    save_df(maint_out,    "PdM_maint")

    # Stream telemetry to disk in chunks — avoids OOM on large targets
    ext        = "parquet" if fmt == "parquet" else "csv"
    telem_path = os.path.join(output_dir, f"PdM_telemetry.{ext}")
    base_time  = telemetry[DATETIME_COL].max() + pd.Timedelta(hours=1)
    total_rows = len(telemetry)
    pq_writer  = None

    print(f"💾 Streaming telemetry to disk in chunks ({fmt.upper()})...")

    # Write original telemetry first
    if fmt == "parquet":
        table     = pa.Table.from_pandas(telemetry, preserve_index=False)
        pq_writer = pq.ParquetWriter(telem_path, table.schema)
        pq_writer.write_table(table)
    else:
        telemetry.to_csv(telem_path, index=False)

    chunk_base = base_time
    for chunk_values in generate_synthetic_telemetry(
            generator, recovery, scaler, len(SENSOR_COLS), n_seqs_needed, device):
        n = len(chunk_values)
        # Use a cycling datetime within a 1-year window to avoid overflow
        timestamps  = pd.date_range(chunk_base, periods=n, freq="1s")
        chunk_base  = timestamps[-1] + pd.Timedelta(seconds=1)
        # Reset base_time to avoid overflow past pandas max datetime
        if chunk_base > pd.Timestamp("2200-01-01"):
            chunk_base = base_time
        mids        = np.resize(np.array(synth_ids), n)
        chunk_df    = pd.DataFrame(chunk_values, columns=SENSOR_COLS).astype("float64")
        chunk_df.insert(0, DATETIME_COL, timestamps)
        chunk_df.insert(1, MACHINE_COL,  mids)
        if fmt == "parquet":
            table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(telem_path, table.schema)
            pq_writer.write_table(table)
        else:
            chunk_df.to_csv(telem_path, mode="a", header=False, index=False)
        total_rows  += n

    if pq_writer:
        pq_writer.close()

    # Summary
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir) if f.endswith(".csv") or f.endswith(".parquet")
    ) / 1024 ** 3

    print("\n✅ Done!")
    print(f"   Output directory : {output_dir}")
    print(f"   Telemetry rows   : {total_rows:,}")
    print(f"   Total size on disk: {total_size:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="./archive",   help="Path to original CSVs")
    parser.add_argument("--output",    default="./augmented", help="Path for augmented output")
    parser.add_argument("--target_gb", default=1.0, type=float, help="Target size in GB")
    parser.add_argument("--format", default="csv", choices=["csv", "parquet"], help="Output format: csv or parquet")
    args = parser.parse_args()
    main(args.input, args.output, args.target_gb, args.format)
