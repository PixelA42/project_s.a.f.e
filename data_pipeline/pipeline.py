"""DataPipeline module for splitting, pseudo-labeling, and manifest generation.

Implements Requirement 7: Data Pipeline — Dataset Preparation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

import numpy as np


class DatasetSplit(NamedTuple):
    """Represents a dataset split into train, validation, and test partitions."""
    
    train: list[str]
    validation: list[str]
    test: list[str]
    
    @property
    def full(self) -> list[str]:
        """Return concatenation of all splits."""
        return self.train + self.validation + self.test


@dataclass
class PseudoLabelResult:
    """Result of pseudo-labeling operation on unlabeled samples."""
    
    labeled: list[dict[str, str | float]]
    unlabeled: list[dict[str, str | float]]


@dataclass
class DatasetManifest:
    """Dataset manifest recording run metadata and split counts."""
    
    run_id: str
    timestamp: str
    labeled_count: int
    unlabeled_count: int
    pseudo_labeled_count: int
    pseudo_label_threshold: float
    splits: dict[str, int]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class DataPipeline:
    """Main data pipeline for dataset splitting, pseudo-labeling, and manifest generation."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the DataPipeline.
        
        Args:
            random_seed: Random seed for reproducible shuffling.
        """
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
    
    def split_dataset(
        self,
        file_list: list[str],
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> DatasetSplit:
        """
        Partition a dataset into train, validation, and test splits.
        
        Requirement 7.1: Partition the ASVspoof_Dataset into labeled training split (80%), 
        labeled validation split (10%), and held-out test split (10%) before model training 
        begins, with no sample appearing in more than one split.
        
        Args:
            file_list: List of file paths or identifiers to split.
            train_ratio: Proportion for training set (default 0.8).
            validation_ratio: Proportion for validation set (default 0.1).
            test_ratio: Proportion for test set (default 0.1).
        
        Returns:
            DatasetSplit containing train, validation, and test lists.
        
        Raises:
            ValueError: If ratios don't sum to 1.0 or if file_list is empty.
        """
        if not file_list:
            raise ValueError("file_list cannot be empty")
        
        total_ratio = train_ratio + validation_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Shuffle with deterministic seed for reproducibility
        shuffled = np.array(file_list)
        self.rng.shuffle(shuffled)
        shuffled = shuffled.tolist()
        
        n = len(shuffled)
        train_size = int(np.round(n * train_ratio))
        validation_size = int(np.round(n * validation_ratio))
        
        train = shuffled[:train_size]
        validation = shuffled[train_size : train_size + validation_size]
        test = shuffled[train_size + validation_size :]
        
        return DatasetSplit(
            train=train,
            validation=validation,
            test=test,
        )
    
    def apply_pseudo_labels(
        self,
        samples_with_confidence: list[dict[str, str | float]],
        threshold: float = 0.85,
    ) -> PseudoLabelResult:
        """
        Separate samples based on pseudo-label confidence threshold.
        
        Requirement 7.4: When the semi-supervised training phase runs, apply Pseudo_Labels 
        to unlabeled samples where the Spectral_Analyzer or Intent_Analyzer confidence 
        score exceeds 0.85.
        
        Requirement 7.5: If a Pseudo_Label confidence score is below 0.85, exclude that 
        sample from the labeled training set and retain it in the unlabeled batch.
        
        Args:
            samples_with_confidence: List of dicts with 'file' and 'confidence' keys.
            threshold: Confidence threshold for pseudo-labeling (default 0.85).
        
        Returns:
            PseudoLabelResult with labeled and unlabeled lists.
        
        Raises:
            ValueError: If threshold is not in [0, 1].
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        
        labeled = []
        unlabeled = []
        
        for sample in samples_with_confidence:
            confidence = float(sample.get("confidence", 0.0))
            if confidence >= threshold:
                labeled.append(sample)
            else:
                unlabeled.append(sample)
        
        return PseudoLabelResult(labeled=labeled, unlabeled=unlabeled)
    
    def generate_manifest(
        self,
        split: DatasetSplit,
        pseudo_result: PseudoLabelResult | None = None,
        threshold: float = 0.85,
    ) -> DatasetManifest:
        """
        Generate a dataset manifest with run metadata and split counts.
        
        Requirement 7.6: Produce a JSON dataset manifest file after each pipeline run, 
        documenting the run ID, timestamp, and counts of labeled, unlabeled, pseudo-labeled 
        samples, and per-split sizes.
        
        Args:
            split: DatasetSplit containing train, validation, test lists.
            pseudo_result: Optional PseudoLabelResult from pseudo-labeling.
            threshold: Pseudo-label threshold value used.
        
        Returns:
            DatasetManifest with all required metadata.
        """
        run_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        labeled_count = len(split.full)
        unlabeled_count = len(pseudo_result.unlabeled) if pseudo_result else 0
        pseudo_labeled_count = len(pseudo_result.labeled) if pseudo_result else 0
        
        splits_dict = {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        }
        
        return DatasetManifest(
            run_id=run_id,
            timestamp=timestamp,
            labeled_count=labeled_count,
            unlabeled_count=unlabeled_count,
            pseudo_labeled_count=pseudo_labeled_count,
            pseudo_label_threshold=threshold,
            splits=splits_dict,
        )
    
    def save_manifest(self, manifest: DatasetManifest, output_path: Path) -> None:
        """
        Save a manifest to a JSON file.
        
        Args:
            manifest: DatasetManifest to save.
            output_path: Path to write the JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
