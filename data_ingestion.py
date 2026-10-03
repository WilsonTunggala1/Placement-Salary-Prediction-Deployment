from pathlib import Path
import pandas as pd

# Base directory
BASE_DIR = Path(__file__).parent

# Define folders
RAW_DIR = BASE_DIR
INGESTED_DIR = BASE_DIR / "ingested"

# Define input and output files
INPUT_DATA_FILE = RAW_DIR / "A.csv"
INPUT_TARGET_FILE = RAW_DIR / "A_targets.csv"
OUTPUT_FILE = INGESTED_DIR / "student_placement_merged.csv"

def ingest_data():
    # Ensure output folder exists
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading raw data files...")
    
    # Read raw data
    df_data = pd.read_csv(INPUT_DATA_FILE)
    df_target = pd.read_csv(INPUT_TARGET_FILE)

    print("Merging datasets on 'Student_ID'...")
    # Merge the datasets
    df = pd.merge(df_data, df_target, on='Student_ID')

    # Basic validation
    assert not df.empty, "The merged dataset is empty! Check if the Student_IDs match."
    
    # Save ingested data
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Data successfully merged and ingested: {OUTPUT_FILE}")
    print(f"Final Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

if __name__ == "__main__":
    ingest_data()