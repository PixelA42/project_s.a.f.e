"""Uncertainty queue for handling ambiguous predictions.

Manages audio samples with predictions in the 0.35-0.65 probability range
(the 'uncertainty zone') for manual review and re-classification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


class UncertaintyStatus(StrEnum):
    """Status of an uncertainty queue item."""

    QUEUED = "queued"  # Pending review
    REVIEWED = "reviewed"  # Reviewed by human
    ESCALATED = "escalated"  # Escalated to security team
    RESOLVED = "resolved"  # Classification finalized
    DISMISSED = "dismissed"  # False alarm


class ReviewAction(StrEnum):
    """Action taken during review."""

    CONFIRM_AI = "confirm_ai"  # Confirmed as AI-generated
    CONFIRM_HUMAN = "confirm_human"  # Confirmed as human
    UNCLEAR = "unclear"  # Still uncertain
    ESCALATE = "escalate"  # Escalate to security team
    RETRAIN = "retrain"  # Flag for model retraining


@dataclass(slots=True)
class UncertaintyQueueItem:
    """Item in the uncertainty review queue."""

    # Identifiers
    item_id: str = field(default_factory=lambda: str(uuid4()))
    audio_file_path: str = ""

    # Prediction information
    predicted_probability: float = 0.0
    predicted_label: int | None = None  # 0=human, 1=ai
    confidence_score: float = 0.0

    # Metadata
    timestamp_created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    timestamp_reviewed: str | None = None

    # Review information
    status: UncertaintyStatus = UncertaintyStatus.QUEUED
    review_action: ReviewAction | None = None
    reviewer_comment: str = ""
    reviewer_id: str = ""

    # Additional context
    audio_duration_seconds: float = 0.0
    spectral_score: float | None = None
    intent_score: float | None = None
    model_name: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def mark_reviewed(
        self,
        action: ReviewAction,
        reviewer_id: str = "system",
        comment: str = "",
    ) -> None:
        """
        Mark this item as reviewed.

        Args:
            action: The review action taken.
            reviewer_id: ID of the reviewer (default 'system').
            comment: Optional comment from reviewer.
        """
        self.timestamp_reviewed = datetime.now(timezone.utc).isoformat()
        self.review_action = action
        self.reviewer_id = reviewer_id
        self.reviewer_comment = comment

        # Update status based on action
        if action in (ReviewAction.CONFIRM_AI, ReviewAction.CONFIRM_HUMAN):
            self.status = UncertaintyStatus.RESOLVED
        elif action == ReviewAction.ESCALATE:
            self.status = UncertaintyStatus.ESCALATED
        elif action == ReviewAction.UNCLEAR:
            self.status = UncertaintyStatus.REVIEWED
        elif action == ReviewAction.RETRAIN:
            self.status = UncertaintyStatus.REVIEWED
            self.tags.append("needs_retraining")

    def is_in_uncertainty_zone(
        self,
        lower_bound: float = 0.35,
        upper_bound: float = 0.65,
    ) -> bool:
        """
        Check if prediction is in uncertainty zone.

        Args:
            lower_bound: Lower probability bound (default 0.35).
            upper_bound: Upper probability bound (default 0.65).

        Returns:
            True if probability is within [lower_bound, upper_bound].
        """
        return lower_bound <= self.predicted_probability <= upper_bound


class UncertaintyQueue:
    """Manages a queue of uncertain predictions for manual review."""

    def __init__(
        self,
        queue_dir: str | Path = "outputs/uncertainty_queue",
        lower_bound: float = 0.35,
        upper_bound: float = 0.65,
    ):
        """
        Initialize the UncertaintyQueue.

        Args:
            queue_dir: Directory to store queue files (default 'outputs/uncertainty_queue').
            lower_bound: Lower probability threshold (default 0.35).
            upper_bound: Upper probability threshold (default 0.65).
        """
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Subdirectories for organization
        self.pending_dir = self.queue_dir / "pending"
        self.reviewed_dir = self.queue_dir / "reviewed"
        self.escalated_dir = self.queue_dir / "escalated"

        for subdir in [self.pending_dir, self.reviewed_dir, self.escalated_dir]:
            subdir.mkdir(parents=True, exist_ok=True)

    def should_queue(self, probability: float) -> bool:
        """
        Determine if a prediction should be queued for review.

        Args:
            probability: Predicted probability (0-1).

        Returns:
            True if probability is in uncertainty zone.
        """
        return self.lower_bound <= probability <= self.upper_bound

    def add_to_queue(
        self,
        audio_file_path: str,
        predicted_probability: float,
        predicted_label: int | None = None,
        confidence_score: float = 0.0,
        audio_duration_seconds: float = 0.0,
        spectral_score: float | None = None,
        intent_score: float | None = None,
        model_name: str = "",
        tags: list[str] | None = None,
    ) -> UncertaintyQueueItem:
        """
        Add an item to the uncertainty queue.

        Args:
            audio_file_path: Path to the audio file.
            predicted_probability: Model's predicted probability (0-1).
            predicted_label: Optional predicted label (0=human, 1=ai).
            confidence_score: Model's confidence score.
            audio_duration_seconds: Duration of audio in seconds.
            spectral_score: Optional spectral analysis score.
            intent_score: Optional intent analysis score.
            model_name: Name of the model that made the prediction.
            tags: Optional list of tags for categorization.

        Returns:
            Created UncertaintyQueueItem.
        """
        item = UncertaintyQueueItem(
            audio_file_path=audio_file_path,
            predicted_probability=predicted_probability,
            predicted_label=predicted_label,
            confidence_score=confidence_score,
            audio_duration_seconds=audio_duration_seconds,
            spectral_score=spectral_score,
            intent_score=intent_score,
            model_name=model_name,
            tags=tags or [],
        )

        # Save to pending directory
        self._save_item(item, self.pending_dir)

        return item

    def _save_item(self, item: UncertaintyQueueItem, target_dir: Path) -> Path:
        """
        Save item to JSON file.

        Args:
            item: Item to save.
            target_dir: Directory to save to.

        Returns:
            Path to saved file.
        """
        file_path = target_dir / f"{item.item_id}.json"
        with open(file_path, "w") as f:
            json.dump(item.to_dict(), f, indent=2, default=str)
        return file_path

    def get_pending_items(self) -> list[UncertaintyQueueItem]:
        """
        Get all pending items from queue.

        Returns:
            List of UncertaintyQueueItem objects.
        """
        items = []
        for json_file in self.pending_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                item = UncertaintyQueueItem(**data)
                items.append(item)
            except Exception as exc:
                print(f"Warning: Failed to load {json_file}: {exc}")
        return items

    def review_item(
        self,
        item_id: str,
        action: ReviewAction,
        reviewer_id: str = "system",
        comment: str = "",
    ) -> bool:
        """
        Mark an item as reviewed and move it to appropriate directory.

        Args:
            item_id: ID of the item to review.
            action: Review action taken.
            reviewer_id: ID of the reviewer.
            comment: Optional comment.

        Returns:
            True if successful, False if item not found.
        """
        # Find the item
        json_file = self.pending_dir / f"{item_id}.json"
        if not json_file.exists():
            return False

        # Load and update
        with open(json_file) as f:
            data = json.load(f)
        item = UncertaintyQueueItem(**data)

        item.mark_reviewed(action, reviewer_id, comment)

        # Move to appropriate directory based on final status
        if item.status == UncertaintyStatus.ESCALATED:
            target_dir = self.escalated_dir
        else:
            target_dir = self.reviewed_dir

        # Delete from pending
        json_file.unlink()

        # Save to target directory
        self._save_item(item, target_dir)

        return True

    def get_queue_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the uncertainty queue.

        Returns:
            Dictionary with queue statistics.
        """
        pending_items = list(self.pending_dir.glob("*.json"))
        reviewed_items = list(self.reviewed_dir.glob("*.json"))
        escalated_items = list(self.escalated_dir.glob("*.json"))

        stats = {
            "total_pending": len(pending_items),
            "total_reviewed": len(reviewed_items),
            "total_escalated": len(escalated_items),
            "uncertainty_bounds": {
                "lower": self.lower_bound,
                "upper": self.upper_bound,
            },
        }

        return stats

    def export_queue_summary(self, output_file: str | Path) -> None:
        """
        Export summary of queue to CSV for review.

        Args:
            output_file: Path to output CSV file.
        """
        import csv

        pending_items = self.get_pending_items()

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", newline="") as f:
            if not pending_items:
                f.write("No pending items in queue\n")
                return

            fieldnames = [
                "item_id",
                "audio_file_path",
                "predicted_probability",
                "predicted_label",
                "audio_duration_seconds",
                "spectral_score",
                "intent_score",
                "model_name",
                "timestamp_created",
                "tags",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for item in pending_items:
                row = {k: item.to_dict()[k] for k in fieldnames if k in item.to_dict()}
                writer.writerow(row)
