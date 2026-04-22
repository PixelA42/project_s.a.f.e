import { motion } from 'framer-motion';
import { formatScore, scoreToPercent } from '@/utils/riskUtils';

interface ScorePanelProps {
  finalScore: number;
  spectralScore: number;
  intentScore: number;
  accentColor: string;
  trackColor: string;
}

interface ScoreRowProps {
  label: string;
  score: number;
  accentColor: string;
  trackColor: string;
  delay: number;
}

function ScoreRow({ label, score, accentColor, trackColor, delay }: ScoreRowProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.4 }}
    >
      <div className="flex justify-between mb-1.5">
        <span className="text-[10px] tracking-widest" style={{ color: 'rgba(255,255,255,0.35)' }}>
          {label}
        </span>
        <span className="text-[11px] font-bold" style={{ color: accentColor }}>
          {formatScore(score)}
        </span>
      </div>
      <div className="h-1 rounded-full overflow-hidden" style={{ background: trackColor }}>
        <motion.div
          className="h-full rounded-full"
          style={{ background: accentColor }}
          initial={{ width: '0%' }}
          animate={{ width: scoreToPercent(score) }}
          transition={{ delay: delay + 0.1, duration: 0.7, ease: 'easeOut' }}
        />
      </div>
    </motion.div>
  );
}

export function ScorePanel({ finalScore, spectralScore, intentScore, accentColor, trackColor }: ScorePanelProps) {
  return (
    <div className="flex flex-col gap-3">
      <ScoreRow label="FINAL SCORE" score={finalScore} accentColor={accentColor} trackColor={trackColor} delay={0.1} />
      <ScoreRow label="SPECTRAL SCORE" score={spectralScore} accentColor={accentColor} trackColor={trackColor} delay={0.2} />
      <ScoreRow label="INTENT SCORE" score={intentScore} accentColor={accentColor} trackColor={trackColor} delay={0.3} />
    </div>
  );
}