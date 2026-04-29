import { motion } from 'framer-motion';
import { PulseRing } from '@/components/shared/PulseRing';
import { ScorePanel } from '@/components/shared/ScorePanel';
import type { ScoreResponse } from '@/types/score';

interface PrankStateProps {
  data: ScoreResponse;
  onDecline: () => void;
  onAccept: () => void;
}

export function PrankState({ data, onDecline, onAccept }: PrankStateProps) {
  const accent = '#ed8936';

  return (
    <motion.div
      className="flex flex-col min-h-[520px] px-7 py-2"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <span className="font-mono text-[10px] tracking-[0.15em]" style={{ color: 'rgba(255,255,255,0.3)' }}>
          S.A.F.E. ENGINE
        </span>
        <div
          className="px-2.5 py-1 rounded-full font-mono text-[10px] tracking-[0.1em]"
          style={{
            background: 'rgba(237,137,54,0.12)',
            border: '1px solid rgba(237,137,54,0.35)',
            color: accent,
          }}
        >
          DIGITAL AVATAR
        </div>
      </div>

      {/* Caller avatar */}
      <div className="text-center mb-5">
        <div className="mx-auto mb-4" style={{ width: 90, height: 90 }}>
          <PulseRing size={90} color={accent} speed="normal" rings={1}>
            <span className="text-3xl" role="img" aria-label="Robot">🤖</span>
          </PulseRing>
        </div>
        <h2 className="font-display text-[20px] font-bold mb-1" style={{ color: '#fbd38d' }}>
          {data.caller_id ?? 'Unknown Caller'}
        </h2>
        <p className="font-mono text-[12px]" style={{ color: 'rgba(255,255,255,0.4)' }}>{data.caller_number}</p>
      </div>

      {/* Threat card */}
      <motion.div
        className="rounded-2xl p-4 mb-5"
        style={{ background: 'rgba(237,137,54,0.07)', border: '1px solid rgba(237,137,54,0.22)' }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15 }}
      >
        <h3 className="font-display text-[13px] font-bold mb-2" style={{ color: accent }}>
          ⚡ AI-Generated Voice Detected
        </h3>
        <p className="font-mono text-[11px] leading-relaxed mb-3" style={{ color: 'rgba(255,255,255,0.5)' }}>
          AI-generated voice with harmless content. No financial coercion, threats, or urgency signals found.
        </p>
        <div
          className="rounded-xl px-3 py-2"
          style={{ background: 'rgba(237,137,54,0.1)' }}
        >
          <p className="font-mono text-[10px] mb-0.5" style={{ color: 'rgba(237,137,54,0.7)' }}>RISK ASSESSMENT</p>
          <p className="font-mono text-[11px]" style={{ color: 'rgba(255,255,255,0.5)' }}>
            Synthetic Voice Detected — Harmless Content
          </p>
        </div>
      </motion.div>

      {/* Scores */}
      <ScorePanel
        finalScore={data.final_score}
        spectralScore={data.spectral_score}
        intentScore={data.intent_score}
        accentColor={accent}
        trackColor="rgba(255,255,255,0.08)"
      />

      {/* Actions */}
      <div className="flex gap-3 mt-auto pt-5">
        <button
          onClick={onDecline}
          className="flex-1 py-3.5 rounded-full font-mono text-[12px] transition-all hover:bg-white/10 active:scale-95"
          style={{
            background: 'rgba(255,255,255,0.06)',
            border: '1px solid rgba(255,255,255,0.12)',
            color: 'rgba(255,255,255,0.65)',
          }}
          aria-label="Decline call"
        >
          ✕ DECLINE
        </button>
        <button
          onClick={onAccept}
          className="flex-1 py-3.5 rounded-full font-mono text-[12px] font-bold transition-all hover:brightness-110 active:scale-95"
          style={{ background: accent, border: 'none', color: '#1a1400' }}
          aria-label="Answer anyway"
        >
          ANSWER ANYWAY
        </button>
      </div>
    </motion.div>
  );
}