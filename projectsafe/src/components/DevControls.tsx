import { motion } from 'framer-motion';
import type { RiskLabel } from '@/types/score';
import clsx from 'clsx';

interface DevControlsProps {
  onSimulate: (label: RiskLabel) => void;
  isLoading: boolean;
}

const SCENARIOS: { label: RiskLabel; display: string; color: string; bg: string }[] = [
  { label: 'SAFE', display: '✓ SAFE CALL', color: '#68d391', bg: 'rgba(72,187,120,0.1)' },
  { label: 'PRANK', display: '⚡ PRANK', color: '#ed8936', bg: 'rgba(237,137,54,0.1)' },
  { label: 'HIGH_RISK', display: '⚠ HIGH RISK', color: '#fc8181', bg: 'rgba(229,62,62,0.1)' },
];

export function DevControls({ onSimulate, isLoading }: DevControlsProps) {
  return (
    <motion.div
      className="mt-6 w-full max-w-[340px] mx-auto"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5 }}
    >
      <div
        className="rounded-2xl p-4"
        style={{
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.08)',
        }}
      >
        <p
          className="font-mono text-[9px] tracking-[0.2em] mb-3 text-center"
          style={{ color: 'rgba(255,255,255,0.25)' }}
        >
          ◆ DEV CONTROLS — SIMULATE CALL
        </p>
        <div className="flex gap-2">
          {SCENARIOS.map(({ label, display, color, bg }) => (
            <button
              key={label}
              onClick={() => onSimulate(label)}
              disabled={isLoading}
              className={clsx(
                'flex-1 py-2.5 rounded-xl font-mono text-[10px] tracking-wide transition-all',
                'hover:brightness-125 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed'
              )}
              style={{ background: bg, border: `1px solid ${color}40`, color }}
              aria-label={`Simulate ${label} call`}
            >
              {display}
            </button>
          ))}
        </div>
      </div>
    </motion.div>
  );
}