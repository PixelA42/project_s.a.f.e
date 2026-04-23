import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from '@/context/AuthContext';
import { NavBar } from '@/components/NavBar';
import { LandingPage } from '@/pages/LandingPage';
import { ProfilePage } from '@/pages/ProfilePage';

// ── Lazy import the existing call analyzer page ────────────────────
// Move the content of your current App.tsx into this component
import { AnalyzerPage } from '@/pages/AnalyzerPage';

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <NavBar />
        <Routes>
          <Route path="/"          element={<LandingPage />} />
          <Route path="/analyzer"  element={<AnalyzerPage />} />
          <Route path="/profile"   element={<ProfilePage />} />
          <Route path="/analyse" element={<AnalyseRecordingsPage />} />
          <Route path="*"          element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}