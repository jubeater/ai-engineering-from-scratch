import sys
import time

try:
    from datasets import Dataset, load_dataset
except ImportError:
    print("Install the datasets library: pip install datasets")
    sys.exit(1)
from data_utils import convert_format, load_and_inspect, make_splits

if __name__ == "__main__":
    print("=" * 60)
    print("Data Management Utility")
    print("=" * 60)

    print("\n--- 1. Load and inspect a dataset ---")
    ds = load_and_inspect("nyu-mll/glue", "mrpc", split="train")
    for row in range(5):
        print(f"  {ds[row]}")

    print("\n--- 2. Stream a dataset ---")
    rows = load_dataset("allenai/c4", name="en", split="train", streaming=True)
    start = time.perf_counter()
    batch = []
    try:
        for i, row in enumerate(rows):
            batch.append(row)
            if (time.perf_counter() - start) >= 10:
                print(f"Got {i + 1} records")
                break
    except Exception:
        raise KeyboardInterrupt
    finally:
        print(f"Streamed {len(batch)} records in 10 seconds")

    print("\n--- 3. Convert formats ---")
    small_ds = ds.select(range(500))
    paths = convert_format(small_ds, "/tmp/data_utils_demo", "glue_sample")

    print("\n--- 4. Create train/val/test splits ---")
    splits = make_splits(small_ds, train_ratio=0.7, val_ratio=0.15, seed=42)
