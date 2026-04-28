"""Tests for data pipeline module.

Tests for Requirement 7: Data Pipeline — Dataset Preparation
- Property 14: Dataset Splits Are Disjoint and Complete
- Property 15: Pseudo-Label Threshold Invariant
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import pytest
from hypothesis import given, strategies as st

from data_pipeline import DataPipeline, DatasetSplit, PseudoLabelResult


class TestDatasetSplit:
    """Tests for DatasetSplit namedtuple."""
    
    def test_dataset_split_full_property(self) -> None:
        """Test that full property concatenates all splits."""
        split = DatasetSplit(
            train=["a", "b"],
            validation=["c"],
            test=["d", "e"],
        )
        assert split.full == ["a", "b", "c", "d", "e"]
    
    def test_dataset_split_empty(self) -> None:
        """Test split with empty lists."""
        split = DatasetSplit(train=[], validation=[], test=[])
        assert split.full == []


class TestDataPipelineInitialization:
    """Tests for DataPipeline initialization."""
    
    def test_pipeline_creation(self) -> None:
        """Test that DataPipeline can be instantiated."""
        pipeline = DataPipeline()
        assert pipeline.random_seed == 42
    
    def test_pipeline_with_custom_seed(self) -> None:
        """Test DataPipeline with custom random seed."""
        pipeline = DataPipeline(random_seed=999)
        assert pipeline.random_seed == 999


class TestSplitDataset:
    """Tests for split_dataset method."""
    
    # Property 14: Dataset Splits Are Disjoint and Complete
    @given(st.lists(st.integers(), min_size=1, max_size=1000, unique=True))
    def test_splits_are_disjoint_and_complete(self, file_ids: list[int]) -> None:
        """
        Property 14: Dataset Splits Are Disjoint and Complete
        
        For any list of file IDs, split_dataset should produce three lists that:
        1. Share no common elements (disjoint)
        2. Together contain exactly the same elements as the input (complete)
        """
        pipeline = DataPipeline()
        file_strings = [str(fid) for fid in file_ids]
        
        split = pipeline.split_dataset(file_strings)
        
        # Check disjointness
        train_set = set(split.train)
        val_set = set(split.validation)
        test_set = set(split.test)
        
        assert train_set.isdisjoint(val_set), "Train and validation share elements"
        assert train_set.isdisjoint(test_set), "Train and test share elements"
        assert val_set.isdisjoint(test_set), "Validation and test share elements"
        
        # Check completeness
        assert train_set | val_set | test_set == set(file_strings), \
            "Combined splits don't equal input"
    
    def test_split_ratio_adherence(self) -> None:
        """Test that splits approximately follow the 80/10/10 ratio."""
        pipeline = DataPipeline()
        file_list = [f"file_{i}" for i in range(1000)]
        
        split = pipeline.split_dataset(file_list)
        
        # Allow small rounding tolerance
        assert 750 <= len(split.train) <= 850, "Train split not ~80%"
        assert 50 <= len(split.validation) <= 150, "Validation split not ~10%"
        assert 50 <= len(split.test) <= 150, "Test split not ~10%"
    
    def test_split_reproducibility(self) -> None:
        """Test that same seed produces same split."""
        file_list = [f"file_{i}" for i in range(100)]
        
        split1 = DataPipeline(random_seed=42).split_dataset(file_list)
        split2 = DataPipeline(random_seed=42).split_dataset(file_list)
        
        assert split1.train == split2.train
        assert split1.validation == split2.validation
        assert split1.test == split2.test
    
    def test_split_different_seeds_produce_different_splits(self) -> None:
        """Test that different seeds produce different splits."""
        file_list = [f"file_{i}" for i in range(100)]
        
        split1 = DataPipeline(random_seed=42).split_dataset(file_list)
        split2 = DataPipeline(random_seed=999).split_dataset(file_list)
        
        # With high probability, different seeds should produce different splits
        assert (split1.train != split2.train or 
                split1.validation != split2.validation or
                split1.test != split2.test)
    
    def test_split_custom_ratios(self) -> None:
        """Test split with custom ratios."""
        pipeline = DataPipeline()
        file_list = [f"file_{i}" for i in range(100)]
        
        split = pipeline.split_dataset(
            file_list,
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
        )
        
        # Allow tolerance
        assert 50 <= len(split.train) <= 70
        assert 10 <= len(split.validation) <= 30
        assert 10 <= len(split.test) <= 30
    
    def test_split_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        pipeline = DataPipeline()
        with pytest.raises(ValueError, match="cannot be empty"):
            pipeline.split_dataset([])
    
    def test_split_invalid_ratios_raises_error(self) -> None:
        """Test that invalid ratios raise ValueError."""
        pipeline = DataPipeline()
        file_list = [f"file_{i}" for i in range(100)]
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            pipeline.split_dataset(file_list, train_ratio=0.5, validation_ratio=0.3, test_ratio=0.3)


class TestPseudoLabeling:
    """Tests for apply_pseudo_labels method."""
    
    # Property 15: Pseudo-Label Threshold Invariant
    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=0, max_size=1000))
    def test_pseudo_label_threshold_invariant(self, confidences: list[float]) -> None:
        """
        Property 15: Pseudo-Label Threshold Invariant
        
        For any list of confidence scores and threshold 0.85:
        1. Every sample with confidence >= 0.85 appears in labeled
        2. Every sample with confidence < 0.85 appears in unlabeled
        """
        pipeline = DataPipeline()
        threshold = 0.85
        
        samples = [
            {"file": f"sample_{i}", "confidence": conf}
            for i, conf in enumerate(confidences)
        ]
        
        result = pipeline.apply_pseudo_labels(samples, threshold=threshold)
        
        # Check invariant
        labeled_confidences = [s["confidence"] for s in result.labeled]
        unlabeled_confidences = [s["confidence"] for s in result.unlabeled]
        
        if labeled_confidences:
            assert min(labeled_confidences) >= threshold, \
                f"Labeled has confidence < {threshold}"
        if unlabeled_confidences:
            assert max(unlabeled_confidences) < threshold, \
                f"Unlabeled has confidence >= {threshold}"
    
    def test_pseudo_label_threshold_0_85(self) -> None:
        """Test pseudo-labeling with threshold 0.85."""
        pipeline = DataPipeline()
        
        samples = [
            {"file": "high1", "confidence": 0.95},
            {"file": "high2", "confidence": 0.90},
            {"file": "boundary", "confidence": 0.85},
            {"file": "low1", "confidence": 0.80},
            {"file": "low2", "confidence": 0.50},
        ]
        
        result = pipeline.apply_pseudo_labels(samples, threshold=0.85)
        
        assert len(result.labeled) == 3
        assert len(result.unlabeled) == 2
        assert all(s["confidence"] >= 0.85 for s in result.labeled)
        assert all(s["confidence"] < 0.85 for s in result.unlabeled)
    
    def test_pseudo_label_empty_list(self) -> None:
        """Test pseudo-labeling with empty list."""
        pipeline = DataPipeline()
        result = pipeline.apply_pseudo_labels([], threshold=0.85)
        
        assert len(result.labeled) == 0
        assert len(result.unlabeled) == 0
    
    def test_pseudo_label_all_above_threshold(self) -> None:
        """Test when all samples exceed threshold."""
        pipeline = DataPipeline()
        
        samples = [
            {"file": "s1", "confidence": 0.95},
            {"file": "s2", "confidence": 0.90},
        ]
        
        result = pipeline.apply_pseudo_labels(samples, threshold=0.85)
        assert len(result.labeled) == 2
        assert len(result.unlabeled) == 0
    
    def test_pseudo_label_all_below_threshold(self) -> None:
        """Test when all samples are below threshold."""
        pipeline = DataPipeline()
        
        samples = [
            {"file": "s1", "confidence": 0.50},
            {"file": "s2", "confidence": 0.70},
        ]
        
        result = pipeline.apply_pseudo_labels(samples, threshold=0.85)
        assert len(result.labeled) == 0
        assert len(result.unlabeled) == 2
    
    def test_pseudo_label_invalid_threshold_raises_error(self) -> None:
        """Test that invalid threshold raises ValueError."""
        pipeline = DataPipeline()
        
        with pytest.raises(ValueError, match="must be in"):
            pipeline.apply_pseudo_labels([], threshold=-0.1)
        
        with pytest.raises(ValueError, match="must be in"):
            pipeline.apply_pseudo_labels([], threshold=1.5)


class TestManifestGeneration:
    """Tests for manifest generation."""
    
    def test_generate_manifest_basic(self) -> None:
        """Test basic manifest generation."""
        pipeline = DataPipeline()
        
        split = DatasetSplit(
            train=[f"file_{i}" for i in range(80)],
            validation=[f"file_{i}" for i in range(80, 90)],
            test=[f"file_{i}" for i in range(90, 100)],
        )
        
        manifest = pipeline.generate_manifest(split)
        
        assert manifest.run_id  # UUID should be present
        assert manifest.timestamp  # Should have ISO timestamp
        assert manifest.labeled_count == 100
        assert manifest.unlabeled_count == 0
        assert manifest.pseudo_labeled_count == 0
        assert manifest.splits == {"train": 80, "validation": 10, "test": 10}
    
    def test_generate_manifest_with_pseudo_result(self) -> None:
        """Test manifest generation with pseudo-labeling result."""
        pipeline = DataPipeline()
        
        split = DatasetSplit(train=["a", "b"], validation=["c"], test=["d"])
        pseudo_result = PseudoLabelResult(
            labeled=[{"file": "u1", "confidence": 0.90}],
            unlabeled=[{"file": "u2", "confidence": 0.50}],
        )
        
        manifest = pipeline.generate_manifest(split, pseudo_result=pseudo_result)
        
        assert manifest.labeled_count == 4
        assert manifest.unlabeled_count == 1
        assert manifest.pseudo_labeled_count == 1
    
    def test_save_and_load_manifest(self) -> None:
        """Test saving and loading manifest from JSON."""
        pipeline = DataPipeline()
        
        split = DatasetSplit(
            train=["a", "b"],
            validation=["c"],
            test=["d"],
        )
        manifest = pipeline.generate_manifest(split)
        
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "manifest.json"
            pipeline.save_manifest(manifest, output_path)
            
            assert output_path.exists()
            
            with open(output_path) as f:
                loaded = json.load(f)
            
            assert loaded["labeled_count"] == 4
            assert loaded["splits"]["train"] == 2
            assert loaded["pseudo_label_threshold"] == 0.85


class TestIntegration:
    """Integration tests combining multiple pipeline operations."""
    
    def test_full_pipeline_workflow(self) -> None:
        """Test complete pipeline workflow: split, pseudo-label, manifest."""
        pipeline = DataPipeline(random_seed=42)
        
        # Create labeled dataset
        labeled_files = [f"labeled_{i}" for i in range(100)]
        split = pipeline.split_dataset(labeled_files)
        
        # Verify split
        assert len(split.full) == 100
        assert set(split.full) == set(labeled_files)
        
        # Create unlabeled with confidence scores
        unlabeled_with_confidence = [
            {"file": f"unlabeled_{i}", "confidence": 0.5 + (i % 50) / 100.0}
            for i in range(50)
        ]
        
        pseudo_result = pipeline.apply_pseudo_labels(
            unlabeled_with_confidence,
            threshold=0.85
        )
        
        # Generate manifest
        manifest = pipeline.generate_manifest(split, pseudo_result)
        
        assert manifest.labeled_count == 100
        assert manifest.unlabeled_count > 0
        assert manifest.pseudo_labeled_count > 0
        # Total samples: original labeled (100) + pseudo-labeled + remaining unlabeled
        total_unlabeled = manifest.pseudo_labeled_count + manifest.unlabeled_count
        assert total_unlabeled == 50
