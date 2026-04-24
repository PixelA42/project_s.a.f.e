from hypothesis import given, settings, strategies as st

from config import SETTINGS
from coreML.risk_classifier import RiskClassifier, RiskLabel
from coreML.score_fuser import ScoreFuser


@settings(max_examples=500, deadline=None)
@given(
    spectral=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    intent=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_score_fusion_formula_invariant(spectral, intent):
    fuser = ScoreFuser()
    fused = fuser.fuse(spectral, intent)
    expected = (
        SETTINGS.fusion.spectral_weight * spectral
        + SETTINGS.fusion.intent_weight * intent
    )

    assert abs(fused - expected) < 1e-9
    assert 0.0 <= fused <= 100.0


@settings(max_examples=200, deadline=None)
@given(
    final_score=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    intent_score=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_risk_classification_completeness(final_score, intent_score):
    classifier = RiskClassifier()
    label = classifier.classify(final_score, intent_score)
    assert label in {RiskLabel.HIGH_RISK, RiskLabel.PRANK, RiskLabel.SAFE}
