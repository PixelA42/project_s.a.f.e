import { useState, useCallback } from 'react';
import type { CallState, RiskLabel } from '@/types/score';
import { fetchScore } from '@/api/scoreApi';

const INITIAL_STATE: CallState = {
  uiState: 'loading',
  data: null,
  error: null,
};

export function useScoreEngine() {
  const [callState, setCallState] = useState<CallState>(INITIAL_STATE);

  const analyzeCall = useCallback(async (label: RiskLabel) => {
    // Always reset to loading first — never show stale data
    setCallState({ uiState: 'loading', data: null, error: null });

    try {
      const response = await fetchScore(label);

      const uiStateMap: Record<RiskLabel, CallState['uiState']> = {
        HIGH_RISK: 'high_risk',
        PRANK: 'prank',
        SAFE: 'safe',
      };

      setCallState({
        uiState: uiStateMap[response.risk_label],
        data: response,
        error: null,
      });
    } catch (err) {
      setCallState({
        uiState: 'safe',
        data: null,
        error: err instanceof Error ? err.message : 'Analysis failed',
      });
    }
  }, []);

  const reset = useCallback(() => {
    setCallState(INITIAL_STATE);
  }, []);

  return { callState, analyzeCall, reset };
}
