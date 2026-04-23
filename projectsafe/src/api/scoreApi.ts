import { fetchMockScore } from '@/mocks/mockResponses';
import type { RiskLabel, ScoreResponse } from '@/types/score';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? '';
const EVALUATE_RISK_URL = `${API_BASE_URL}/api/evaluate-risk`;

async function parseApiError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { error?: string };
    if (payload.error) {
      return payload.error;
    }
  } catch {
    // Fall through to generic message.
  }

  return `Backend request failed with status ${response.status}`;
}

export async function fetchScore(label: RiskLabel): Promise<ScoreResponse> {
  try {
    const response = await fetch(EVALUATE_RISK_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ scenario: label }),
    });

    if (!response.ok) {
      throw new Error(await parseApiError(response));
    }

    return (await response.json()) as ScoreResponse;
  } catch (error) {
    if (import.meta.env.DEV) {
      console.warn('Backend unavailable, falling back to mock response.', error);
      return fetchMockScore(label);
    }

    throw error;
  }
}
