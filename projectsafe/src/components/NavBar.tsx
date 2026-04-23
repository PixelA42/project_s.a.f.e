import { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '@/context/AuthContext';
import { LoginModal } from '@/components/auth/LoginModal';

export function NavBar() {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();
  const [modalOpen, setModalOpen] = useState(false);
  const [modalTab, setModalTab] = useState<'login' | 'signup'>('login');
  const [avatarMenuOpen, setAvatarMenuOpen] = useState(false);

  function openLogin() { setModalTab('login'); setModalOpen(true); }
  function openSignup() { setModalTab('signup'); setModalOpen(true); }

  const initials = user?.name
    .split(' ')
    .map((w) => w[0])
    .join('')
    .slice(0, 2)
    .toUpperCase() ?? '??';

  const navLinkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-1.5 rounded-md font-mono text-[11px] tracking-wide transition-all ${
      isActive
        ? 'text-white bg-white/[0.08]'
        : 'text-white/40 hover:text-white/80 hover:bg-white/[0.04]'
    }`;

  return (
    <>
      <nav
        className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-6 h-[52px]"
        style={{
          background: 'rgba(10,10,18,0.92)',
          backdropFilter: 'blur(16px)',
          borderBottom: '1px solid rgba(255,255,255,0.07)',
        }}
        role="navigation"
        aria-label="Main navigation"
      >
        {/* Logo */}
        <button
          onClick={() => navigate('/')}
          className="font-display text-[15px] font-extrabold text-white tracking-tight"
          style={{ background: 'none', border: 'none', cursor: 'pointer' }}
          aria-label="Go to home"
        >
          Project{' '}
          <span
            style={{
              background: 'linear-gradient(90deg, #63b3ed, #68d391)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            S.A.F.E.
          </span>
        </button>

        {/* Nav links */}
        <div className="flex items-center gap-1" role="menubar">
          <NavLink to="/" end className={navLinkClass} role="menuitem">Home</NavLink>
          <NavLink to="/analyzer" className={navLinkClass} role="menuitem">Analyzer</NavLink>
          <NavLink to="/profile" className={navLinkClass} role="menuitem">Profile</NavLink>
        </div>

        {/* Auth area */}
        <div className="flex items-center gap-2">
          <AnimatePresence mode="wait">
            {isAuthenticated ? (
              <motion.div
                key="avatar"
                className="relative"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
              >
                <button
                  onClick={() => setAvatarMenuOpen((v) => !v)}
                  className="w-8 h-8 rounded-full flex items-center justify-center font-display text-xs font-bold text-white transition-all hover:ring-2 hover:ring-white/20"
                  style={{
                    background: 'linear-gradient(135deg, #2b6cb0, #276749)',
                    border: '2px solid rgba(255,255,255,0.15)',
                  }}
                  aria-label="Open profile menu"
                  aria-haspopup="true"
                  aria-expanded={avatarMenuOpen}
                >
                  {initials}
                </button>

                <AnimatePresence>
                  {avatarMenuOpen && (
                    <>
                      {/* Click-outside overlay */}
                      <div
                        className="fixed inset-0 z-40"
                        onClick={() => setAvatarMenuOpen(false)}
                      />
                      <motion.div
                        className="absolute right-0 top-10 w-44 rounded-xl py-1.5 z-50"
                        style={{
                          background: '#141420',
                          border: '1px solid rgba(255,255,255,0.1)',
                          boxShadow: '0 16px 48px rgba(0,0,0,0.5)',
                        }}
                        initial={{ opacity: 0, y: -6, scale: 0.97 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -6, scale: 0.97 }}
                        transition={{ duration: 0.15 }}
                        role="menu"
                      >
                        <div className="px-3 py-2 border-b" style={{ borderColor: 'rgba(255,255,255,0.07)' }}>
                          <p className="font-display text-xs font-bold text-white truncate">{user?.name}</p>
                          <p className="font-mono text-[10px] truncate" style={{ color: 'rgba(255,255,255,0.35)' }}>
                            {user?.email}
                          </p>
                        </div>
                        {[
                          { label: 'View profile', action: () => { navigate('/profile'); setAvatarMenuOpen(false); } },
                          { label: 'Analyzer', action: () => { navigate('/analyzer'); setAvatarMenuOpen(false); } },
                        ].map(({ label, action }) => (
                          <button
                            key={label}
                            onClick={action}
                            className="w-full text-left px-3 py-1.5 font-mono text-[11px] transition-all"
                            style={{ color: 'rgba(255,255,255,0.55)', background: 'none', border: 'none', cursor: 'pointer' }}
                            onMouseEnter={(e) => { (e.target as HTMLElement).style.color = '#fff'; (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.05)'; }}
                            onMouseLeave={(e) => { (e.target as HTMLElement).style.color = 'rgba(255,255,255,0.55)'; (e.target as HTMLElement).style.background = 'none'; }}
                            role="menuitem"
                          >
                            {label}
                          </button>
                        ))}
                        <div className="border-t mt-1 pt-1" style={{ borderColor: 'rgba(255,255,255,0.07)' }}>
                          <button
                            onClick={() => { logout(); setAvatarMenuOpen(false); navigate('/'); }}
                            className="w-full text-left px-3 py-1.5 font-mono text-[11px] transition-all"
                            style={{ color: 'rgba(252,129,129,0.65)', background: 'none', border: 'none', cursor: 'pointer' }}
                            onMouseEnter={(e) => { (e.target as HTMLElement).style.color = '#fc8181'; (e.target as HTMLElement).style.background = 'rgba(229,62,62,0.08)'; }}
                            onMouseLeave={(e) => { (e.target as HTMLElement).style.color = 'rgba(252,129,129,0.65)'; (e.target as HTMLElement).style.background = 'none'; }}
                            role="menuitem"
                          >
                            ↩ Sign out
                          </button>
                        </div>
                      </motion.div>
                    </>
                  )}
                </AnimatePresence>
              </motion.div>
            ) : (
              <motion.div
                key="auth-btns"
                className="flex items-center gap-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <button
                  onClick={openLogin}
                  className="px-3 py-1.5 rounded-md font-mono text-[11px] transition-all"
                  style={{
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: 'rgba(255,255,255,0.65)',
                    cursor: 'pointer',
                  }}
                  onMouseEnter={(e) => { (e.target as HTMLElement).style.color = '#fff'; }}
                  onMouseLeave={(e) => { (e.target as HTMLElement).style.color = 'rgba(255,255,255,0.65)'; }}
                >
                  Log in
                </button>
                <button
                  onClick={openSignup}
                  className="px-3 py-1.5 rounded-md font-mono text-[11px] font-bold text-white transition-all hover:brightness-110"
                  style={{
                    background: 'linear-gradient(135deg, #2b6cb0, #276749)',
                    border: 'none',
                    cursor: 'pointer',
                  }}
                >
                  Sign up
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </nav>

      <LoginModal
        isOpen={modalOpen}
        initialTab={modalTab}
        onClose={() => setModalOpen(false)}
      />
      
<NavLink to="/analyse" className={navLinkClass} role="menuitem">Analyse</NavLink>
    </>
  );
}