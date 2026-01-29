from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data folders on IDUN
DATA_ROOT = ROOT / "data" / "raw"

SENTINEL_DIR = DATA_ROOT / "Sentinel"
MASK_DIR     = DATA_ROOT / "masks"
VHR_DIR      = DATA_ROOT / "vhr"
PLANETSCOPE_DIR = DATA_ROOT / "PlanetScope"
APLHAEARTH_DIR = DATA_ROOT / "AlphaEarth"

# Output / reports
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Training defaults
PATCH_SIZE = 256
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 10

# Create folders if missing 
for d in [REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
