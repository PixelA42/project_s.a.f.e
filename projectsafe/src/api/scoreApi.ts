import { fetchMockScore } from '@/mocks/mockResponses';
import type { RiskLabel, ScoreResponse } from '@/types/score';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? 'http://localhost:5000';
const ANALYZE_URL = `${API_BASE_URL}/api/v1/analyze`;

async function parseApiError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { error?: string; description?: string };
    if (payload.error) {
      return payload.description || payload.error;
    }
  } catch {
    // Fall through to generic message.
  }

  return `Backend request failed with status ${response.status}`;
}

export async function fetchScore(label: RiskLabel): Promise<ScoreResponse> {
  try {
    const response = await fetch(ANALYZE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      // Use mock_label for dev/testing; backend will synthesize scores
      body: JSON.stringify({ mock_label: label }),
    });

    if (!response.ok) {
      throw new Error(await parseApiError(response));
    }

    const data = (await response.json()) as ScoreResponse;
    return data;
  } catch (error) {
    if (import.meta.env.DEV) {
      console.warn('Backend unavailable, falling back to mock response.', error);
      return fetchMockScore(label);
    }

    throw error;
  }
}
