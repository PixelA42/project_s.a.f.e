/**
 * Wraps the existing call analyzer UI (previously in App.tsx)
 * into a proper routed page.
 */
import { AnimatePresence } from 'framer-motion';
import { useScoreEngine } from '@/hooks/useScoreEngine';
import { PhoneFrame } from '@/components/shared/PhoneFrame';
import { LoadingState } from '@/components/CallScreen/LoadingState';
import { SafeState } from '@/components/CallScreen/SafeState';
import { PrankState } from '@/components/CallScreen/PrankState';
import { HighRiskState } from '@/components/CallScreen/HighRiskState';
import { DevControls } from '@/components/DevControls';

const BG_MAP = {
  loading:  'linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%)',
  safe:     'linear-gradient(160deg, #0d1f18 0%, #0f2d20 50%, #1a3a2a 100%)',
  prank:    'linear-gradient(160deg, #1a1400 0%, #261c00 50%, #332400 100%)',
  high_risk:'linear-gradient(160deg, #1a0000 0%, #2d0000 50%, #3d0505 100%)',
} as const;

export function AnalyzerPage() {
  const { callState, analyzeCall, reset } = useScoreEngine();
  const { uiState, data } = callState;
  const isLoading = uiState === 'loading';

  function renderScreen() {
    switch (uiState) {
      case 'loading':   return <LoadingState />;
      case 'safe':      return data ? <SafeState data={data} onDecline={reset} onAccept={reset} /> : null;
      case 'prank':     return data ? <PrankState data={data} onDecline={reset} onAccept={reset} /> : null;
      case 'high_risk': return data ? <HighRiskState data={data} onBlock={reset} /> : null;
    }
  }

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-4 py-8 pt-[80px]"
      style={{ background: 'radial-gradient(ellipse at 50% 0%, #1a1a2e 0%, #0a0a12 60%)' }}
    >
      <div className="text-center mb-6">
        <h1 className="font-display text-2xl font-extrabold tracking-tight text-white">
          Call{' '}
          <span
            style={{
              background: 'linear-gradient(90deg, #63b3ed, #68d391)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Analyzer
          </span>
        </h1>
        <p className="font-mono text-[10px] tracking-[0.15em] mt-1" style={{ color: 'rgba(255,255,255,0.25)' }}>
          SYNTHETIC AUDIO FRAUD ENGINE
        </p>
      </div>

      <AnimatePresence mode="wait">
        <PhoneFrame background={BG_MAP[uiState]} key={uiState}>
          {renderScreen()}
        </PhoneFrame>
      </AnimatePresence>

      <DevControls onSimulate={analyzeCall} isLoading={isLoading} />

      <p className="font-mono text-[9px] tracking-[0.15em] mt-6" style={{ color: 'rgba(255,255,255,0.15)' }}>
        MOCK MODE — BACKEND NOT CONNECTED
      </p>
    </div>
  );
}