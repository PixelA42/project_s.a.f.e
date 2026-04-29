import { act, renderHook, waitFor } from '@testing-library/react';
import { useScoreEngine } from '@/hooks/useScoreEngine';
import type { ScoreResponse } from '@/types/score';

const fetchScoreMock = vi.fn();

vi.mock('@/api/scoreApi', () => ({
  fetchScore: (...args: unknown[]) => fetchScoreMock(...args),
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe('Requirement 6 - useScoreEngine loading behavior', () => {
  beforeEach(() => {
    fetchScoreMock.mockReset();
  });

  it('clears stale result while loading before next response', async () => {
    const safeResponse: ScoreResponse = {
      risk_label: 'SAFE',
      final_score: 12,
      spectral_score: 8,
      intent_score: 5,
      caller_id: 'Jane',
      caller_number: '+1 (555) 000-0000',
      timestamp: new Date().toISOString(),
    };
    const highRiskResponse: ScoreResponse = {
      risk_label: 'HIGH_RISK',
      final_score: 92,
      spectral_score: 94,
      intent_score: 88,
      caller_id: 'Unknown',
      caller_number: '+1 (800) 555-0199',
      timestamp: new Date().toISOString(),
    };

    fetchScoreMock.mockResolvedValueOnce(safeResponse);
    const second = deferred<ScoreResponse>();
    fetchScoreMock.mockReturnValueOnce(second.promise);

    const { result } = renderHook(() => useScoreEngine());

    await act(async () => {
      await result.current.analyzeCall('SAFE');
    });

    expect(result.current.callState.uiState).toBe('safe');
    expect(result.current.callState.data?.risk_label).toBe('SAFE');

    void act(() => {
      void result.current.analyzeCall('HIGH_RISK');
    });

    expect(result.current.callState.uiState).toBe('loading');
    expect(result.current.callState.data).toBeNull();
    expect(result.current.callState.error).toBeNull();

    second.resolve(highRiskResponse);

    await waitFor(() => {
      expect(result.current.callState.uiState).toBe('high_risk');
      expect(result.current.callState.data?.risk_label).toBe('HIGH_RISK');
    });
  });
});

