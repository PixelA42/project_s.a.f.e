import { motion } from 'framer-motion';
import { WaveBars } from '@/components/shared/WaveBar';

export function LoadingState() {
  return (
    <motion.div
      className="flex flex-col items-center justify-center min-h-[520px] px-8 gap-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      {/* Triple spinner */}
      <div className="relative w-20 h-20">
        <div className="absolute inset-0 rounded-full border border-[rgba(99,179,237,0.12)]" />
        <div
          className="absolute inset-0 rounded-full border-2 border-transparent animate-spin"
          style={{ borderTopColor: '#63b3ed' }}
        />
        <div
          className="absolute inset-2 rounded-full border-2 border-transparent"
          style={{
            borderTopColor: '#4299e1',
            animation: 'spin 1.4s linear infinite reverse',
          }}
        />
        <div
          className="absolute inset-[18px] rounded-full flex items-center justify-center"
          style={{ background: 'rgba(66,153,225,0.1)' }}
        >
          <div
            className="w-2.5 h-2.5 rounded-full animate-blink"
            style={{ background: '#63b3ed' }}
          />
        </div>
      </div>

      {/* Title */}
      <div className="text-center">
        <p className="font-mono text-[11px] tracking-[0.2em] mb-2" style={{ color: 'rgba(99,179,237,0.6)' }}>
          ANALYZING AUDIO SIGNAL
        </p>
        <h1 className="font-display text-2xl font-bold text-white mb-1">S.A.F.E.</h1>
        <p className="font-mono text-[10px] tracking-[0.15em]" style={{ color: 'rgba(255,255,255,0.25)' }}>
          SYNTHETIC AUDIO FRAUD ENGINE
        </p>
      </div>

      {/* Progress bars */}
      <div className="w-full max-w-[200px] space-y-3">
        {[
          { label: 'SPECTRAL LAYER' },
          { label: 'INTENT LAYER' },
        ].map(({ label }) => (
          <div key={label}>
            <div className="flex justify-between mb-1.5">
              <span className="font-mono text-[10px] tracking-widest" style={{ color: 'rgba(255,255,255,0.35)' }}>
                {label}
              </span>
              <span className="font-mono text-[10px]" style={{ color: 'rgba(99,179,237,0.5)' }}>—</span>
            </div>
            <div className="h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.08)' }}>
              <motion.div
                className="h-full rounded-full"
                style={{ background: 'linear-gradient(90deg, #2b6cb0, #63b3ed)' }}
                animate={{ width: ['0%', '65%', '30%', '85%', '50%', '90%'] }}
                transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Wave processing indicator */}
      <div className="flex items-center gap-3">
        <WaveBars color="#63b3ed" />
        <span className="font-mono text-[10px] tracking-[0.1em]" style={{ color: 'rgba(99,179,237,0.5)' }}>
          PROCESSING
        </span>
        <WaveBars color="#63b3ed" />
      </div>
    </motion.div>
  );
}