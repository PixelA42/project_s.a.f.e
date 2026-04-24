import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '@/context/AuthContext';
import { LoginModal } from '@/components/auth/LoginModal';
import { FeatureCard, type CardVariant } from '@/components/FeatureCard';

// ── Data ──────────────────────────────────────────────────────────────
const FEATURES: Array<{
  variant: CardVariant;
  icon: string;
  title: string;
  desc: string;
}> = [
  {
    variant: 'spectral',
    icon: '',
    title: 'Spectral Analysis',
    desc: 'MFCCs and prosody irregularities catch cloned voices before a word is spoken.',
  },
  {
    variant: 'intent',
    icon: '',
    title: 'Intent Detection',
    desc: 'NLP layer flags financial coercion, urgency signals, and authority impersonation.',
  },
  {
    variant: 'wolf',
    icon: '',
    title: 'No Cry-Wolf',
    desc: 'Dual-gate logic separates harmless AI pranks from real scam threats.',
  },
  {
    variant: 'realtime',
    icon: '',
    title: 'Real-Time Results',
    desc: 'Sub-second scoring with transparent final, spectral, and intent scores.',
  },
];

const STATS: Array<{ value: string; label: string }> = [
  { value: '99.2%', label: 'Detection accuracy' },
  { value: '<0.8s', label: 'Analysis latency' },
  { value: '2×',   label: 'Scoring layers' },
];

const HOW_IT_WORKS: Array<{
  step: string;
  title: string;
  desc: string;
  color: string;
}> = [
  {
    step: '01',
    title: 'Upload or receive call',
    desc: 'Feed in a live call stream or upload a recording. Any audio format works.',
    color: '#63b3ed',
  },
  {
    step: '02',
    title: 'Dual-layer analysis',
    desc: 'Spectral forensics and NLP intent detection run independently in parallel.',
    color: '#ed8936',
  },
  {
    step: '03',
    title: 'Score fusion',
    desc: 'Weighted scores are fused into a final_score and classified in milliseconds.',
    color: '#fc8181',
  },
  {
    step: '04',
    title: 'Instant verdict',
    desc: 'SAFE, PRANK, or HIGH_RISK — with full score transparency so you understand why.',
    color: '#48bb78',
  },
];

// ── Animation variants ────────────────────────────────────────────────
const STAGGER = {
  container: {
    animate: { transition: { staggerChildren: 0.08 } },
  },
};

const FADE_UP = {
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
};

// ── Component ─────────────────────────────────────────────────────────
export function LandingPage() {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();
  const [signupOpen, setSignupOpen] = useState(false);

  return (
    <main
      className="min-h-screen pt-[52px]"
      style={{ background: 'radial-gradient(ellipse at 50% 0%, #1a1a2e 0%, #0a0a12 60%)' }}
    >

      {/* ═══════════════════════════════════════════════════════
          HERO
      ═══════════════════════════════════════════════════════ */}
      <section className="relative px-6 pt-16 pb-14 overflow-hidden max-w-5xl mx-auto">

        {/* Ambient background glows */}
        <div className="absolute inset-0 pointer-events-none" aria-hidden="true">
          <div
            className="absolute top-0 left-1/4 w-[480px] h-[480px] rounded-full"
            style={{
              background: 'radial-gradient(circle, rgba(43,108,176,0.1), transparent 70%)',
              transform: 'translate(-50%, -30%)',
            }}
          />
          <div
            className="absolute top-0 right-1/4 w-96 h-96 rounded-full"
            style={{
              background: 'radial-gradient(circle, rgba(39,103,73,0.08), transparent 70%)',
              transform: 'translate(50%, -20%)',
            }}
          />
        </div>

        <motion.div
          className="relative"
          variants={STAGGER.container}
          initial="initial"
          animate="animate"
        >
          {/* Live badge */}
          <motion.div
            variants={FADE_UP}
            className="inline-flex items-center gap-2 mb-6 px-3 py-1.5 rounded-full"
            style={{
              background: 'rgba(99,179,237,0.08)',
              border: '1px solid rgba(99,179,237,0.2)',
            }}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-[#63b3ed] animate-blink" />
            <span
              className="font-mono text-[9px] tracking-[0.2em]"
              style={{ color: 'rgba(99,179,237,0.85)' }}
            >
              SAVE YOURSELF FROM SCAMS 
            </span>
          </motion.div>

          {/* Headline */}
          <motion.h1
            variants={FADE_UP}
            className="font-display font-extrabold text-white leading-[1.1] tracking-tight mb-5 max-w-2xl"
            style={{ fontSize: 'clamp(2rem, 5vw, 3.25rem)' }}
          >
            Stop{' '}
            <span
              style={{
                background: 'linear-gradient(90deg, #fc8181, #ed8936)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              AI voice scams
            </span>
            <br />
            before they start.
          </motion.h1>

          {/* Sub-headline */}
          <motion.p
            variants={FADE_UP}
            className="font-mono text-[12px] leading-[1.85] mb-8 max-w-xl"
            style={{ color: 'rgba(255,255,255,0.42)' }}
          >
            Project S.A.F.E. analyzes every incoming call through two independent AI layers —
            spectral forensics and intent detection — to protect you from synthetic voice fraud
            without crying wolf on harmless calls.
          </motion.p>

          {/* CTA row */}
          <motion.div variants={FADE_UP} className="flex gap-3 flex-wrap">
            <motion.button
              onClick={() => navigate('/analyzer')}
              className="px-6 py-3 rounded-lg font-mono text-[12px] font-bold text-white"
              style={{
                background: 'linear-gradient(135deg, #2b6cb0, #276749)',
                border: 'none',
                cursor: 'pointer',
              }}
              whileHover={{ scale: 1.02, filter: 'brightness(1.12)' }}
              whileTap={{ scale: 0.97 }}
            >
              ▶ Open Analyzer
            </motion.button>

            <motion.button
              onClick={() => navigate('/analyse')}
              className="px-6 py-3 rounded-lg font-mono text-[12px] font-bold"
              style={{
                background: 'rgba(99,179,237,0.08)',
                border: '1px solid rgba(99,179,237,0.25)',
                color: '#63b3ed',
                cursor: 'pointer',
              }}
              whileHover={{ scale: 1.02, background: 'rgba(99,179,237,0.14)' }}
              whileTap={{ scale: 0.97 }}
            >
              🎙️ Analyse a Recording
            </motion.button>

            {!isAuthenticated && (
              <motion.button
                onClick={() => setSignupOpen(true)}
                className="px-6 py-3 rounded-lg font-mono text-[12px]"
                style={{
                  background: 'rgba(255,255,255,0.05)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  color: 'rgba(255,255,255,0.6)',
                  cursor: 'pointer',
                }}
                whileHover={{ background: 'rgba(255,255,255,0.09)', color: '#fff' }}
                whileTap={{ scale: 0.97 }}
              >
                Create Free Account →
              </motion.button>
            )}
          </motion.div>
        </motion.div>
      </section>

      {/* ═══════════════════════════════════════════════════════
          STATS
      ═══════════════════════════════════════════════════════ */}
      <section className="px-6 pb-12 max-w-5xl mx-auto">
        <motion.div
          className="grid grid-cols-3 gap-4"
          initial="initial"
          whileInView="animate"
          viewport={{ once: true, margin: '-60px' }}
          variants={STAGGER.container}
        >
          {STATS.map(({ value, label }) => (
            <motion.div
              key={label}
              variants={FADE_UP}
              className="rounded-xl p-5 text-center"
              style={{
                background: 'rgba(255,255,255,0.025)',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
            >
              <div className="font-display text-3xl font-extrabold text-white mb-1.5">
                {value}
              </div>
              <div
                className="font-mono text-[9px] tracking-[0.15em] uppercase"
                style={{ color: 'rgba(255,255,255,0.3)' }}
              >
                {label}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* ═══════════════════════════════════════════════════════
          FEATURES — animated canvas cards
      ═══════════════════════════════════════════════════════ */}
      <section className="px-6 pb-14 max-w-5xl mx-auto">
        <motion.p
          className="font-mono text-[9px] tracking-[0.22em] mb-6"
          style={{ color: 'rgba(255,255,255,0.2)' }}
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          ◆ HOW IT WORKS
        </motion.p>

        <motion.div
          className="grid grid-cols-2 gap-4"
          initial="initial"
          whileInView="animate"
          viewport={{ once: true, margin: '-60px' }}
          variants={STAGGER.container}
        >
          {FEATURES.map(({ variant, icon, title, desc }) => (
            <motion.div key={title} variants={FADE_UP}>
              <FeatureCard
                variant={variant}
                icon={icon}
                title={title}
                desc={desc}
              />
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* ═══════════════════════════════════════════════════════
          HOW IT WORKS — step-by-step
      ═══════════════════════════════════════════════════════ */}
      <section className="px-6 pb-14 max-w-5xl mx-auto">
        <motion.p
          className="font-mono text-[9px] tracking-[0.22em] mb-6"
          style={{ color: 'rgba(255,255,255,0.2)' }}
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          ◆ THE PIPELINE
        </motion.p>

        <motion.div
          className="grid grid-cols-2 gap-4"
          initial="initial"
          whileInView="animate"
          viewport={{ once: true, margin: '-60px' }}
          variants={STAGGER.container}
        >
          {HOW_IT_WORKS.map(({ step, title, desc, color }) => (
            <motion.div
              key={step}
              variants={FADE_UP}
              className="rounded-xl p-5 flex gap-4"
              style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
            >
              <div
                className="font-display text-xl font-extrabold flex-shrink-0 mt-0.5 w-9"
                style={{ color }}
              >
                {step}
              </div>
              <div>
                <h3 className="font-display text-sm font-bold text-white mb-1.5">{title}</h3>
                <p className="font-mono text-[10px] leading-relaxed" style={{ color: 'rgba(255,255,255,0.35)' }}>
                  {desc}
                </p>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* ═══════════════════════════════════════════════════════
          QUICK ACTIONS
      ═══════════════════════════════════════════════════════ */}
      <section className="px-6 pb-14 max-w-5xl mx-auto">
        <motion.div
          className="grid grid-cols-2 gap-4"
          initial="initial"
          whileInView="animate"
          viewport={{ once: true, margin: '-40px' }}
          variants={STAGGER.container}
        >
          {/* Analyzer card */}
          <motion.div
            variants={FADE_UP}
            className="rounded-2xl p-6 cursor-pointer"
            style={{
              background: 'linear-gradient(135deg, rgba(43,108,176,0.12), rgba(39,103,73,0.1))',
              border: '1px solid rgba(99,179,237,0.15)',
            }}
            whileHover={{ borderColor: 'rgba(99,179,237,0.35)', scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            onClick={() => navigate('/analyzer')}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && navigate('/analyzer')}
            aria-label="Open live call analyzer"
          >
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center text-lg mb-4"
              style={{ background: 'rgba(99,179,237,0.12)', border: '1px solid rgba(99,179,237,0.2)' }}
            >
              📞
            </div>
            <h3 className="font-display text-base font-bold text-white mb-1.5">
              Live Call Analyzer
            </h3>
            <p className="font-mono text-[10px] leading-relaxed mb-4" style={{ color: 'rgba(255,255,255,0.38)' }}>
              Simulate or connect a live call and see real-time risk scoring across all four states.
            </p>
            <span className="font-mono text-[10px]" style={{ color: '#63b3ed' }}>
              Open analyzer →
            </span>
          </motion.div>

          {/* Recordings card */}
          <motion.div
            variants={FADE_UP}
            className="rounded-2xl p-6 cursor-pointer"
            style={{
              background: 'linear-gradient(135deg, rgba(99,179,237,0.08), rgba(43,108,176,0.06))',
              border: '1px solid rgba(99,179,237,0.12)',
            }}
            whileHover={{ borderColor: 'rgba(99,179,237,0.3)', scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            onClick={() => navigate('/analyse')}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && navigate('/analyse')}
            aria-label="Upload and analyse a recording"
          >
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center text-lg mb-4"
              style={{ background: 'rgba(99,179,237,0.1)', border: '1px solid rgba(99,179,237,0.18)' }}
            >
              🎙️
            </div>
            <h3 className="font-display text-base font-bold text-white mb-1.5">
              Analyse a Recording
            </h3>
            <p className="font-mono text-[10px] leading-relaxed mb-4" style={{ color: 'rgba(255,255,255,0.38)' }}>
              Upload any audio file — MP3, WAV, M4A, FLAC — and get a full spectral + intent report.
            </p>
            <span className="font-mono text-[10px]" style={{ color: '#63b3ed' }}>
              Upload recording →
            </span>
          </motion.div>
        </motion.div>
      </section>

      {/* ═══════════════════════════════════════════════════════
          CTA BANNER (only shown when logged out)
      ═══════════════════════════════════════════════════════ */}
      {!isAuthenticated && (
        <section className="px-6 pb-16 max-w-5xl mx-auto">
          <motion.div
            className="rounded-2xl p-10 text-center relative overflow-hidden"
            style={{
              background: 'linear-gradient(135deg, rgba(43,108,176,0.14), rgba(39,103,73,0.12))',
              border: '1px solid rgba(255,255,255,0.08)',
            }}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            {/* Subtle glow */}
            <div
              className="absolute inset-0 pointer-events-none"
              style={{
                background: 'radial-gradient(ellipse at 50% 0%, rgba(99,179,237,0.06), transparent 60%)',
              }}
              aria-hidden="true"
            />
            <div className="relative">
              <div
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full mb-5"
                style={{
                  background: 'rgba(72,187,120,0.1)',
                  border: '1px solid rgba(72,187,120,0.2)',
                }}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-[#48bb78] animate-blink" />
                <span className="font-mono text-[9px] tracking-[0.15em]" style={{ color: '#68d391' }}>
                  FREE TO GET STARTED
                </span>
              </div>

              <h2 className="font-display text-2xl font-extrabold text-white mb-2 tracking-tight">
                Ready to protect your calls?
              </h2>
              <p className="font-mono text-[11px] mb-7" style={{ color: 'rgba(255,255,255,0.4)' }}>
                Free account. No credit card. Full access to both scoring layers.
              </p>

              <div className="flex gap-3 justify-center">
                <motion.button
                  onClick={() => setSignupOpen(true)}
                  className="px-7 py-3 rounded-lg font-mono text-[12px] font-bold text-white"
                  style={{
                    background: 'linear-gradient(135deg, #2b6cb0, #276749)',
                    border: 'none',
                    cursor: 'pointer',
                  }}
                  whileHover={{ scale: 1.02, filter: 'brightness(1.12)' }}
                  whileTap={{ scale: 0.97 }}
                >
                  Get Started Free →
                </motion.button>
                <motion.button
                  onClick={() => navigate('/analyzer')}
                  className="px-7 py-3 rounded-lg font-mono text-[12px]"
                  style={{
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: 'rgba(255,255,255,0.6)',
                    cursor: 'pointer',
                  }}
                  whileHover={{ color: '#fff', background: 'rgba(255,255,255,0.09)' }}
                  whileTap={{ scale: 0.97 }}
                >
                  Try Without Account
                </motion.button>
              </div>
            </div>
          </motion.div>
        </section>
      )}

      {/* ═══════════════════════════════════════════════════════
          FOOTER
      ═══════════════════════════════════════════════════════ */}
      <footer
        className="px-6 py-6 max-w-5xl mx-auto flex items-center justify-between"
        style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}
      >
        <span className="font-display text-sm font-extrabold text-white/40 tracking-tight">
          Project S.A.F.E.
        </span>
        <span className="font-mono text-[9px] tracking-[0.15em]" style={{ color: 'rgba(255,255,255,0.18)' }}>
          SYNTHETIC AUDIO FRAUD ENGINE · v1.0
        </span>
      </footer>

      {/* Auth modal */}
      <LoginModal
        isOpen={signupOpen}
        initialTab="signup"
        onClose={() => setSignupOpen(false)}
      />
    </main>
  );
}