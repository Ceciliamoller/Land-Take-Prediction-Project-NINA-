from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
VHR_DIR = RAW_DIR / "vhr"
MASK_DIR = RAW_DIR / "masks"
SENTINEL_DIR = RAW_DIR / "Sentinel"
PLANETSCOPE_DIR = RAW_DIR / "PlanetScope"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Output / reports
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Training defaults
PATCH_SIZE = 256
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 10

# Create folders if missing 
for d in [INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
