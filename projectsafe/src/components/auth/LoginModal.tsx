import { useState, type FormEvent } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '@/context/AuthContext';

interface LoginModalProps {
  isOpen: boolean;
  initialTab?: 'login' | 'signup';
  onClose: () => void;
}

type Tab = 'login' | 'signup';

export function LoginModal({ isOpen, initialTab = 'login', onClose }: LoginModalProps) {
  const { login, signup } = useAuth();
  const [tab, setTab] = useState<Tab>(initialTab);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);
    try {
      if (tab === 'login') {
        await login(email, password);
      } else {
        if (!name.trim()) { setError('Name is required'); setIsSubmitting(false); return; }
        await signup(name, email, password);
      }
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center px-4"
          style={{ background: 'rgba(0,0,0,0.75)' }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={(e) => e.target === e.currentTarget && onClose()}
        >
          <motion.div
            className="relative w-full max-w-sm rounded-2xl p-7"
            style={{ background: '#0f0f1a', border: '1px solid rgba(255,255,255,0.1)' }}
            initial={{ opacity: 0, y: 20, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.97 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
          >
            {/* Close */}
            <button
              onClick={onClose}
              className="absolute top-4 right-4 w-7 h-7 flex items-center justify-center rounded-md font-mono text-sm transition-all"
              style={{ color: 'rgba(255,255,255,0.35)', background: 'transparent' }}
              onMouseEnter={(e) => { (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.07)'; (e.target as HTMLElement).style.color = '#fff'; }}
              onMouseLeave={(e) => { (e.target as HTMLElement).style.background = 'transparent'; (e.target as HTMLElement).style.color = 'rgba(255,255,255,0.35)'; }}
              aria-label="Close"
            >
              ✕
            </button>

            {/* Header */}
            <h2 className="font-display text-xl font-extrabold text-white mb-0.5">
              {tab === 'login' ? 'Welcome back' : 'Create account'}
            </h2>
            <p className="font-mono text-[9px] tracking-[0.18em] mb-5" style={{ color: 'rgba(255,255,255,0.3)' }}>
              {tab === 'login' ? 'SIGN IN TO PROJECT S.A.F.E.' : 'JOIN PROJECT S.A.F.E.'}
            </p>

            {/* Tabs */}
            <div className="flex gap-1 p-1 rounded-lg mb-5" style={{ background: 'rgba(255,255,255,0.04)' }}>
              {(['login', 'signup'] as Tab[]).map((t) => (
                <button
                  key={t}
                  onClick={() => { setTab(t); setError(''); }}
                  className="flex-1 py-1.5 rounded-md font-mono text-[10px] transition-all"
                  style={{
                    background: tab === t ? 'rgba(255,255,255,0.1)' : 'transparent',
                    color: tab === t ? '#fff' : 'rgba(255,255,255,0.35)',
                    border: 'none',
                    cursor: 'pointer',
                  }}
                >
                  {t === 'login' ? 'Log in' : 'Sign up'}
                </button>
              ))}
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit} className="flex flex-col gap-3.5">
              <AnimatePresence mode="wait">
                {tab === 'signup' && (
                  <motion.div
                    key="name-field"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <InputField
                      label="FULL NAME"
                      type="text"
                      placeholder="Arjun Kumar"
                      value={name}
                      onChange={setName}
                      autoFocus
                    />
                  </motion.div>
                )}
              </AnimatePresence>

              <InputField
                label="EMAIL"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={setEmail}
                autoFocus={tab === 'login'}
              />

              <InputField
                label="PASSWORD"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={setPassword}
              />

              {error && (
                <p className="font-mono text-[10px] text-center" style={{ color: '#fc8181' }}>
                  {error}
                </p>
              )}

              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full py-3 rounded-lg font-mono text-[11px] font-bold tracking-[0.05em] text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed mt-1"
                style={{ background: 'linear-gradient(135deg, #2b6cb0, #276749)', border: 'none', cursor: 'pointer' }}
              >
                {isSubmitting
                  ? '— Verifying —'
                  : tab === 'login'
                  ? '→ Sign in'
                  : '→ Create account'}
              </button>
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ── Reusable input ────────────────────────────────────────────────────
function InputField({
  label,
  type,
  placeholder,
  value,
  onChange,
  autoFocus,
}: {
  label: string;
  type: string;
  placeholder: string;
  value: string;
  onChange: (v: string) => void;
  autoFocus?: boolean;
}) {
  return (
    <div>
      <label className="block font-mono text-[9px] tracking-[0.12em] mb-1.5" style={{ color: 'rgba(255,255,255,0.35)' }}>
        {label}
      </label>
      <input
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        autoFocus={autoFocus}
        required
        className="w-full px-3 py-2.5 rounded-lg font-mono text-[11px] text-white transition-all outline-none"
        style={{
          background: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.1)',
          color: '#fff',
        }}
        onFocus={(e) => (e.target.style.borderColor = 'rgba(99,179,237,0.5)')}
        onBlur={(e) => (e.target.style.borderColor = 'rgba(255,255,255,0.1)')}
      />
    </div>
  );
}