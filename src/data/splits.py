"""
Shared train/val/test split utilities for fair model comparison.

This module ensures both U-Net and FCEF baselines use identical data splits.
"""

from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


def get_splits(
    ref_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[list[str], list[str], list[str]]:
    """
    Split reference IDs into train/val/test sets with fixed random seed.
    
    This function ensures reproducible splits across different notebooks and models,
    enabling fair baseline comparison under identical data conditions.
    
    Args:
        ref_ids: List of reference IDs (e.g., tile identifiers)
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_ref_ids, val_ref_ids, test_ref_ids)
    
    Raises:
        ValueError: If ratios don't sum to 1.0
    
    Example:
        >>> ref_ids = ["tile_001", "tile_002", ..., "tile_034"]
        >>> train_ids, val_ids, test_ids = get_splits(ref_ids)
        >>> len(train_ids), len(val_ids), len(test_ids)
        (23, 5, 5)  # For 34 total tiles with 70/15/15 split
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )
    
    # First split: separate train from (val + test)
    train_ref_ids, val_test_ref_ids = train_test_split(
        ref_ids,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
    )
    
    # Second split: separate val from test
    # Adjust the test_size to be relative to the remaining data
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_ref_ids, test_ref_ids = train_test_split(
        val_test_ref_ids,
        test_size=relative_test_size,
        random_state=random_state,
    )
    
    return train_ref_ids, val_ref_ids, test_ref_ids


def get_ref_ids_from_directory(
    directory: Path,
    pattern: str = "*_RGBNIRRSWIRQ_Mosaic.tif",
    exclude_suffix: str = "_RGBNIRRSWIRQ_Mosaic",
) -> list[str]:
    """
    Extract reference IDs from filenames in a directory.
    
    Args:
        directory: Directory containing data files
        pattern: Glob pattern to match files (default: Sentinel pattern)
        exclude_suffix: Suffix to remove from filenames to get ref_id
    
    Returns:
        Sorted list of reference IDs
    
    Example:
        >>> from src.config import SENTINEL_DIR
        >>> ref_ids = get_ref_ids_from_directory(SENTINEL_DIR)
        >>> ref_ids[:3]
        ['R101C117', 'R101C118', 'R101C119']
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    ref_ids = [f.stem.replace(exclude_suffix, "") for f in files]
    return ref_ids
