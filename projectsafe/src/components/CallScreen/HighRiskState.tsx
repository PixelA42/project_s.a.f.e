import { motion } from 'framer-motion';
import { PulseRing } from '@/components/shared/PulseRing';
import { ScorePanel } from '@/components/shared/ScorePanel';
import type { ScoreResponse } from '@/types/score';

interface HighRiskStateProps {
  data: ScoreResponse;
  onBlock: () => void;
}

export function HighRiskState({ data, onBlock }: HighRiskStateProps) {
  const accent = '#fc8181';
  const accentDim = '#e53e3e';

  return (
    <motion.div
      className="relative flex flex-col min-h-[520px] px-7 py-2 overflow-hidden"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Scan line effect */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden opacity-[0.04]">
        <motion.div
          className="absolute left-0 right-0 h-16"
          style={{ background: 'linear-gradient(180deg, transparent, rgba(255,50,50,0.9), transparent)' }}
          animate={{ y: ['-100%', '200%'] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'linear' }}
        />
      </div>

      {/* Header */}
      <div className="flex justify-between items-center mb-5 relative">
        <span className="font-mono text-[10px] tracking-[0.15em]" style={{ color: 'rgba(255,100,100,0.45)' }}>
          S.A.F.E. ENGINE
        </span>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full animate-blink-fast" style={{ background: accent }} />
          <span className="font-mono text-[10px] tracking-widest animate-blink-fast" style={{ color: accent }}>
            LIVE THREAT
          </span>
        </div>
      </div>

      {/* Alert icon */}
      <div className="text-center mb-5 relative">
        <div className="mx-auto mb-4" style={{ width: 100, height: 100 }}>
          <PulseRing size={100} color={accentDim} speed="fast" rings={2}>
            <span className="font-display text-4xl font-extrabold" style={{ color: accent }}>!</span>
          </PulseRing>
        </div>
        <motion.h2
          className="font-display text-[26px] font-extrabold text-white mb-1 tracking-tight"
          animate={{ opacity: [1, 0.75, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        >
          SCAM LIKELY
        </motion.h2>
        <p className="font-mono text-[11px] tracking-[0.12em] uppercase" style={{ color: 'rgba(252,129,129,0.55)' }}>
          High-Risk Incoming Call
        </p>
      </div>

      {/* Threat breakdown */}
      <motion.div
        className="rounded-2xl p-4 mb-5 relative"
        style={{ background: 'rgba(229,62,62,0.08)', border: '1px solid rgba(229,62,62,0.32)' }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15 }}
      >
        <div className="flex items-center gap-2 mb-3">
          <motion.span
            style={{ color: accent, fontSize: 14 }}
            animate={{ opacity: [1, 0.2, 1] }}
            transition={{ duration: 0.8, repeat: Infinity }}
          >
            ▲
          </motion.span>
          <h3 className="font-display text-[14px] font-bold tracking-wide" style={{ color: accent }}>
            DUAL THREAT CONFIRMED
          </h3>
        </div>

        <div className="grid grid-cols-2 gap-2 mb-3">
          {[
            { label: 'SPECTRAL', value: 'SYNTHETIC VOICE' },
            { label: 'INTENT', value: 'COERCION' },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="rounded-xl p-2.5"
              style={{ background: 'rgba(229,62,62,0.1)', border: '1px solid rgba(229,62,62,0.18)' }}
            >
              <p className="font-mono text-[9px] mb-0.5" style={{ color: 'rgba(252,129,129,0.55)' }}>{label}</p>
              <p className="font-mono text-[11px] font-bold" style={{ color: accent }}>{value}</p>
            </div>
          ))}
        </div>

        <p className="font-mono text-[11px] leading-relaxed" style={{ color: 'rgba(255,200,200,0.5)' }}>
          Synthetic Voice + Coercion Detected. Financial urgency signals and AI voice markers both flagged simultaneously.
        </p>
      </motion.div>

      {/* Scores */}
      <ScorePanel
        finalScore={data.final_score}
        spectralScore={data.spectral_score}
        intentScore={data.intent_score}
        accentColor={accent}
        trackColor="rgba(255,255,255,0.06)"
      />

      {/* Block CTA */}
      <motion.button
        onClick={onBlock}
        className="w-full py-4 rounded-full font-mono text-[13px] font-bold tracking-[0.05em] text-white transition-all mt-auto pt-5"
        style={{ background: accentDim, border: `2px solid rgba(252,129,129,0.35)` }}
        whileHover={{ scale: 1.02, filter: 'brightness(1.1)' }}
        whileTap={{ scale: 0.97 }}
        aria-label="Block this call"
      >
        ⛔ BLOCK THIS CALL
      </motion.button>
    </motion.div>
  );
}