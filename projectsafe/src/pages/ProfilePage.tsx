import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '@/context/AuthContext';
import { LoginModal } from '@/components/auth/LoginModal';

const MOCK_ACTIVITY = [
  { color: '#fc8181', label: 'HIGH_RISK call blocked', detail: '+1 (800) 555-0199', time: '2m ago' },
  { color: '#ed8936', label: 'PRANK detected — answered', detail: '+1 (555) 867-5309', time: '1h ago' },
  { color: '#68d391', label: 'SAFE call from John Doe', detail: '+1 (555) 234-5678', time: '3h ago' },
  { color: '#68d391', label: 'SAFE call verified', detail: '+1 (555) 100-2020', time: '5h ago' },
  { color: '#fc8181', label: 'HIGH_RISK call blocked', detail: '+44 800 555 0100', time: 'Yesterday' },
] as const;

export function ProfilePage() {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();
  const [loginOpen, setLoginOpen] = useState(false);

  const initials = user?.name
    .split(' ')
    .map((w) => w[0])
    .join('')
    .slice(0, 2)
    .toUpperCase() ?? '??';

  // ── Not logged in ──────────────────────────────────────────
  if (!isAuthenticated) {
    return (
      <main
        className="min-h-screen pt-[52px] flex items-center justify-center px-6"
        style={{ background: 'radial-gradient(ellipse at 50% 0%, #1a1a2e 0%, #0a0a12 60%)' }}
      >
        <motion.div
          className="text-center max-w-sm"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <div
            className="w-16 h-16 rounded-full flex items-center justify-center text-2xl mx-auto mb-5"
            style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
          >
            🔒
          </div>
          <h2 className="font-display text-xl font-extrabold text-white mb-2">
            Sign in to view your profile
          </h2>
          <p className="font-mono text-[11px] leading-relaxed mb-6" style={{ color: 'rgba(255,255,255,0.35)' }}>
            Your call history, threat stats, and account settings are saved here once you create an account.
          </p>
          <div className="flex gap-3 justify-center">
            <button
              onClick={() => setLoginOpen(true)}
              className="px-5 py-2.5 rounded-lg font-mono text-[11px] transition-all"
              style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.65)', cursor: 'pointer' }}
            >
              Log in
            </button>
            <button
              onClick={() => setLoginOpen(true)}
              className="px-5 py-2.5 rounded-lg font-mono text-[11px] font-bold text-white hover:brightness-110 transition-all"
              style={{ background: 'linear-gradient(135deg, #2b6cb0, #276749)', border: 'none', cursor: 'pointer' }}
            >
              Create account
            </button>
          </div>
        </motion.div>
        <LoginModal isOpen={loginOpen} initialTab="signup" onClose={() => setLoginOpen(false)} />
      </main>
    );
  }

  // ── Logged in ──────────────────────────────────────────────
  return (
    <main
      className="min-h-screen pt-[52px]"
      style={{ background: 'radial-gradient(ellipse at 50% 0%, #1a1a2e 0%, #0a0a12 60%)' }}
    >
      <div className="max-w-2xl mx-auto px-6 py-8">

        {/* Profile header card */}
        <motion.div
          className="rounded-2xl overflow-hidden mb-5"
          style={{ border: '1px solid rgba(255,255,255,0.08)' }}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <div
            className="px-7 py-7 relative overflow-hidden"
            style={{ background: 'linear-gradient(160deg, #0d1a2e 0%, #0f2d20 100%)' }}
          >
            <div className="absolute inset-0 pointer-events-none" style={{ background: 'radial-gradient(ellipse at 80% 50%, rgba(72,187,120,0.08), transparent 60%)' }} />
            <div className="relative flex items-start gap-5">
              <motion.div
                className="w-16 h-16 rounded-full flex items-center justify-center font-display text-xl font-extrabold text-white flex-shrink-0"
                style={{ background: 'linear-gradient(135deg, #2b6cb0, #276749)', border: '2px solid rgba(255,255,255,0.15)' }}
                animate={{ y: [0, -5, 0] }}
                transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
              >
                {initials}
              </motion.div>
              <div className="pt-1">
                <h1 className="font-display text-xl font-extrabold text-white mb-0.5">{user?.name}</h1>
                <p className="font-mono text-[11px] mb-3" style={{ color: 'rgba(255,255,255,0.4)' }}>{user?.email}</p>
                <div
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full font-mono text-[9px] tracking-[0.1em]"
                  style={{ background: 'rgba(72,187,120,0.1)', border: '1px solid rgba(72,187,120,0.22)', color: '#68d391' }}
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-[#68d391] animate-blink" />
                  Active · {user?.plan === 'free' ? 'Free Plan' : 'Pro Plan'}
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Stats grid */}
        <motion.div
          className="grid grid-cols-4 gap-3 mb-5"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.4 }}
        >
          {[
            { label: 'Member since', value: user?.joinedAt ?? '—' },
            { label: 'Plan', value: user?.plan === 'free' ? 'Free' : 'Pro' },
            { label: 'Calls analyzed', value: user?.callsAnalyzed.toString() ?? '0' },
            { label: 'Threats blocked', value: user?.threatsBlocked.toString() ?? '0' },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="rounded-xl p-3.5"
              style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
            >
              <p className="font-mono text-[9px] tracking-[0.1em] mb-1.5" style={{ color: 'rgba(255,255,255,0.3)' }}>
                {label.toUpperCase()}
              </p>
              <p className="font-display text-base font-bold text-white">{value}</p>
            </div>
          ))}
        </motion.div>

        {/* Recent activity */}
        <motion.div
          className="rounded-2xl p-5 mb-5"
          style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.4 }}
        >
          <p className="font-mono text-[9px] tracking-[0.2em] mb-4" style={{ color: 'rgba(255,255,255,0.25)' }}>
            ◆ RECENT ACTIVITY
          </p>
          <div className="flex flex-col gap-2.5">
            {MOCK_ACTIVITY.map(({ color, label, detail, time }, i) => (
              <motion.div
                key={i}
                className="flex items-center gap-3 px-3 py-2.5 rounded-lg"
                style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.25 + i * 0.06 }}
              >
                <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                <div className="flex-1 min-w-0">
                  <p className="font-mono text-[10px] text-white/60 truncate">{label}</p>
                  <p className="font-mono text-[9px] truncate" style={{ color: 'rgba(255,255,255,0.25)' }}>{detail}</p>
                </div>
                <span className="font-mono text-[9px] flex-shrink-0" style={{ color: 'rgba(255,255,255,0.2)' }}>{time}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Account actions */}
        <motion.div
          className="flex gap-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <button
            onClick={() => navigate('/analyzer')}
            className="flex-1 py-2.5 rounded-xl font-mono text-[11px] transition-all hover:brightness-110"
            style={{ background: 'linear-gradient(135deg, #2b6cb0, #276749)', border: 'none', color: '#fff', cursor: 'pointer' }}
          >
            ▶ Open Analyzer
          </button>
          <button
            onClick={() => { logout(); navigate('/'); }}
            className="px-5 py-2.5 rounded-xl font-mono text-[11px] transition-all"
            style={{
              background: 'rgba(229,62,62,0.07)',
              border: '1px solid rgba(229,62,62,0.2)',
              color: 'rgba(252,129,129,0.7)',
              cursor: 'pointer',
            }}
            onMouseEnter={(e) => { (e.target as HTMLElement).style.color = '#fc8181'; }}
            onMouseLeave={(e) => { (e.target as HTMLElement).style.color = 'rgba(252,129,129,0.7)'; }}
          >
            ↩ Sign out
          </button>
        </motion.div>

      </div>
    </main>
  );
}