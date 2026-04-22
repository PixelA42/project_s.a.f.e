export type RiskLabel = 'HIGH_RISK' | 'PRANK' | 'SAFE';

export type UIState = 'loading' | 'high_risk' | 'prank' | 'safe';

export interface ScoreResponse {
  risk_label: RiskLabel;
  final_score: number;   // 0–100
  spectral_score: number; // 0–100
  intent_score: number;   // 0–100
  caller_id?: string;
  caller_number?: string;
  timestamp?: string;
}

export interface CallState {
  uiState: UIState;
  data: ScoreResponse | null;
  error: string | null;
}