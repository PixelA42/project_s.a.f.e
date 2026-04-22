import type { ScoreResponse } from '@/types/score';

export const MOCK_HIGH_RISK: ScoreResponse = {
  risk_label: 'HIGH_RISK',
  final_score: 92,
  spectral_score: 94,
  intent_score: 88,
  caller_id: 'Unknown Caller',
  caller_number: '+1 (800) 555-0199',
  timestamp: new Date().toISOString(),
};

export const MOCK_PRANK: ScoreResponse = {
  risk_label: 'PRANK',
  final_score: 55,
  spectral_score: 78,
  intent_score: 12,
  caller_id: 'Unknown Caller',
  caller_number: '+1 (555) 867-5309',
  timestamp: new Date().toISOString(),
};

export const MOCK_SAFE: ScoreResponse = {
  risk_label: 'SAFE',
  final_score: 12,
  spectral_score: 8,
  intent_score: 5,
  caller_id: 'John Doe',
  caller_number: '+1 (555) 234-5678',
  timestamp: new Date().toISOString(),
};

/** Simulate a network fetch with a realistic delay */
export async function fetchMockScore(label: 'HIGH_RISK' | 'PRANK' | 'SAFE'): Promise<ScoreResponse> {
  const delay = 1500 + Math.random() * 800;
  await new Promise((res) => setTimeout(res, delay));

  const map = {
    HIGH_RISK: MOCK_HIGH_RISK,
    PRANK: MOCK_PRANK,
    SAFE: MOCK_SAFE,
  };

  return { ...map[label], timestamp: new Date().toISOString() };
}