import { motion } from 'framer-motion';
import { ScorePanel } from '@/components/shared/ScorePanel';
import type { ScoreResponse } from '@/types/score';

interface SafeStateProps {
  data: ScoreResponse;
  onDecline: () => void;
  onAccept: () => void;
}

export function SafeState({ data, onDecline, onAccept }: SafeStateProps) {
  const accent = '#48bb78';
  const muted = '#68d391';

  return (
    <motion.div
      className="flex flex-col min-h-[520px] px-7 py-2"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Header */}
      <div className="flex justify-between items-center mb-7">
        <span className="font-mono text-[10px] tracking-[0.15em]" style={{ color: 'rgba(255,255,255,0.3)' }}>
          S.A.F.E. ENGINE
        </span>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full animate-blink" style={{ background: accent }} />
          <span className="font-mono text-[10px] tracking-widest" style={{ color: accent }}>VERIFIED SAFE</span>
        </div>
      </div>

      {/* Caller avatar */}
      <div className="text-center mb-6">
        <motion.div
          className="w-[90px] h-[90px] rounded-full mx-auto mb-4 flex items-center justify-center relative"
          style={{ border: `2px solid rgba(72,187,120,0.35)`, background: 'rgba(72,187,120,0.08)' }}
          initial={{ scale: 0.85 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.1, type: 'spring', stiffness: 200 }}
        >
          <div className="absolute inset-[-10px] rounded-full border border-[rgba(72,187,120,0.12)]" />
          <span className="font-display text-3xl font-bold" style={{ color: muted }}>
            {(data.caller_id ?? 'U').split(' ').map((w) => w[0]).join('').slice(0, 2).toUpperCase()}
          </span>
        </motion.div>
        <h2 className="font-display text-[22px] font-bold text-white mb-1">{data.caller_id ?? 'Unknown'}</h2>
        <p className="font-mono text-[12px]" style={{ color: 'rgba(255,255,255,0.4)' }}>{data.caller_number}</p>
        <p className="font-mono text-[10px] tracking-[0.1em] mt-1" style={{ color: 'rgba(255,255,255,0.2)' }}>INCOMING CALL</p>
      </div>

      {/* Risk badge */}
      <motion.div
        className="rounded-2xl p-4 mb-5"
        style={{ background: 'rgba(72,187,120,0.07)', border: '1px solid rgba(72,187,120,0.2)' }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full" style={{ background: accent }} />
          <span className="font-display text-[12px] font-semibold tracking-[0.1em]" style={{ color: muted }}>
            NO THREAT DETECTED
          </span>
        </div>
        <p className="font-mono text-[11px] leading-relaxed" style={{ color: 'rgba(255,255,255,0.45)' }}>
          Voice patterns match natural human speech. No synthetic markers or coercion signals identified.
        </p>
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
          className="flex-1 py-3.5 rounded-full font-mono text-[12px] text-white/80 transition-all hover:bg-white/10 active:scale-95"
          style={{ background: 'rgba(229,62,62,0.8)', border: 'none' }}
          aria-label="Decline call"
        >
          ✕ DECLINE
        </button>
        <button
          onClick={onAccept}
          className="flex-1 py-3.5 rounded-full font-mono text-[12px] font-bold transition-all hover:brightness-110 active:scale-95"
          style={{ background: accent, border: 'none', color: '#0d1f18' }}
          aria-label="Accept call"
        >
          ✓ ACCEPT
        </button>
      </div>
    </motion.div>
  );
}