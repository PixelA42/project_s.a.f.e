import type { RiskLabel } from '@/types/score';

export function getRiskColor(label: RiskLabel): string {
  const colors: Record<RiskLabel, string> = {
    HIGH_RISK: '#fc8181',
    PRANK: '#ed8936',
    SAFE: '#68d391',
  };
  return colors[label];
}

export function getRiskBgGradient(label: RiskLabel): string {
  const gradients: Record<RiskLabel, string> = {
    HIGH_RISK: 'linear-gradient(160deg, #1a0000 0%, #2d0000 50%, #3d0505 100%)',
    PRANK: 'linear-gradient(160deg, #1a1400 0%, #261c00 50%, #332400 100%)',
    SAFE: 'linear-gradient(160deg, #0d1f18 0%, #0f2d20 50%, #1a3a2a 100%)',
  };
  return gradients[label];
}

export function scoreToPercent(score: number): string {
  return `${Math.min(100, Math.max(0, score))}%`;
}

export function formatScore(score: number): string {
  return `${Math.round(score)} / 100`;
}