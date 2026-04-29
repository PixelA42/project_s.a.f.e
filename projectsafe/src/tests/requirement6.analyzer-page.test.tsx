import { render, screen } from '@testing-library/react';
import { AnalyzerPage } from '@/pages/AnalyzerPage';
import type { CallState, ScoreResponse } from '@/types/score';

const analyzeCallMock = vi.fn();
const resetMock = vi.fn();

let currentCallState: CallState;

const makeScore = (riskLabel: ScoreResponse['risk_label']): ScoreResponse => ({
  risk_label: riskLabel,
  final_score: 82,
  spectral_score: 91,
  intent_score: 74,
  caller_id: 'John Doe',
  caller_number: '+1 (555) 123-4567',
  timestamp: new Date().toISOString(),
});

vi.mock('@/hooks/useScoreEngine', () => ({
  useScoreEngine: () => ({
    callState: currentCallState,
    analyzeCall: analyzeCallMock,
    reset: resetMock,
  }),
}));

vi.mock('@/components/DevControls', () => ({
  DevControls: () => <div data-testid="dev-controls" />,
}));

describe('Requirement 6 - AnalyzerPage UI states', () => {
  beforeEach(() => {
    analyzeCallMock.mockReset();
    resetMock.mockReset();
  });

  it('shows SCAM LIKELY state for HIGH_RISK', () => {
    currentCallState = {
      uiState: 'high_risk',
      data: makeScore('HIGH_RISK'),
      error: null,
    };

    render(<AnalyzerPage />);

    expect(screen.getByText('SCAM LIKELY')).toBeInTheDocument();
    expect(screen.getByText(/Synthetic Voice \+ Coercion Detected/i)).toBeInTheDocument();
  });

  it('shows Digital Avatar harmless state for PRANK', () => {
    currentCallState = {
      uiState: 'prank',
      data: makeScore('PRANK'),
      error: null,
    };

    render(<AnalyzerPage />);

    expect(screen.getByText(/^DIGITAL AVATAR$/i)).toBeInTheDocument();
    expect(screen.getByText(/Synthetic Voice Detected/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Harmless Content/i).length).toBeGreaterThan(0);
  });

  it('shows standard incoming-call screen for SAFE', () => {
    currentCallState = {
      uiState: 'safe',
      data: makeScore('SAFE'),
      error: null,
    };

    render(<AnalyzerPage />);

    expect(screen.getByText(/VERIFIED SAFE/i)).toBeInTheDocument();
    expect(screen.getByText(/INCOMING CALL/i)).toBeInTheDocument();
  });

  it('shows all three scores in non-loading states', () => {
    const nonLoadingStates: CallState[] = [
      { uiState: 'safe', data: makeScore('SAFE'), error: null },
      { uiState: 'prank', data: makeScore('PRANK'), error: null },
      { uiState: 'high_risk', data: makeScore('HIGH_RISK'), error: null },
    ];

    for (const state of nonLoadingStates) {
      currentCallState = state;
      const { unmount } = render(<AnalyzerPage />);

      expect(screen.getByText('FINAL SCORE')).toBeInTheDocument();
      expect(screen.getByText('SPECTRAL SCORE')).toBeInTheDocument();
      expect(screen.getByText('INTENT SCORE')).toBeInTheDocument();

      unmount();
    }
  });
});

