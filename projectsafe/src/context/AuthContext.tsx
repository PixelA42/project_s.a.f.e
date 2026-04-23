/**
 * Auth context — manages login state across the whole app.
 * 
 * Currently uses localStorage for persistence (survives page refresh).
 * When your backend auth is ready, replace the mock functions below
 * with real API calls to your Flask /api/v1/auth/* endpoints.
 */
import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from 'react';

// ── Types ─────────────────────────────────────────────────────────────
export interface User {
  id: string;
  name: string;
  email: string;
  plan: 'free' | 'pro';
  joinedAt: string;
  callsAnalyzed: number;
  threatsBlocked: number;
}

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
}

interface AuthContextValue extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  signup: (name: string, email: string, password: string) => Promise<void>;
  logout: () => void;
}

// ── Context ───────────────────────────────────────────────────────────
const AuthContext = createContext<AuthContextValue | null>(null);

const STORAGE_KEY = 'safe_auth_user';

// ── Mock user (replace with real API calls) ───────────────────────────
function buildMockUser(name: string, email: string): User {
  return {
    id: `usr_${Math.random().toString(36).slice(2, 10)}`,
    name,
    email,
    plan: 'free',
    joinedAt: new Date().toLocaleDateString('en-US', { month: 'short', year: 'numeric' }),
    callsAnalyzed: Math.floor(Math.random() * 200 + 50),
    threatsBlocked: Math.floor(Math.random() * 15 + 1),
  };
}

// ── Provider ──────────────────────────────────────────────────────────
export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    isLoading: true,
    isAuthenticated: false,
  });

  // Restore session on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const user: User = JSON.parse(stored);
        setState({ user, isLoading: false, isAuthenticated: true });
      } else {
        setState((s) => ({ ...s, isLoading: false }));
      }
    } catch {
      setState((s) => ({ ...s, isLoading: false }));
    }
  }, []);

  const login = useCallback(async (email: string, _password: string) => {
    /**
     * ── REPLACE WITH REAL API CALL ─────────────────────────────────────
     * const res = await fetch('/api/v1/auth/login', {
     *   method: 'POST',
     *   headers: { 'Content-Type': 'application/json' },
     *   body: JSON.stringify({ email, password }),
     * });
     * if (!res.ok) throw new Error('Invalid credentials');
     * const { user, token } = await res.json();
     * localStorage.setItem('safe_auth_token', token);
     * ──────────────────────────────────────────────────────────────────
     */
    await new Promise((r) => setTimeout(r, 800)); // simulate network
    const name = email.split('@')[0].replace(/[._]/g, ' ');
    const user = buildMockUser(name, email);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
    setState({ user, isLoading: false, isAuthenticated: true });
  }, []);

  const signup = useCallback(async (name: string, email: string, _password: string) => {
    /**
     * ── REPLACE WITH REAL API CALL ─────────────────────────────────────
     * const res = await fetch('/api/v1/auth/signup', {
     *   method: 'POST',
     *   headers: { 'Content-Type': 'application/json' },
     *   body: JSON.stringify({ name, email, password }),
     * });
     * if (!res.ok) throw new Error('Signup failed');
     * const { user, token } = await res.json();
     * localStorage.setItem('safe_auth_token', token);
     * ──────────────────────────────────────────────────────────────────
     */
    await new Promise((r) => setTimeout(r, 1000));
    const user = buildMockUser(name, email);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
    setState({ user, isLoading: false, isAuthenticated: true });
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setState({ user: null, isLoading: false, isAuthenticated: false });
  }, []);

  return (
    <AuthContext.Provider value={{ ...state, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

// ── Hook ──────────────────────────────────────────────────────────────
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>');
  return ctx;
}