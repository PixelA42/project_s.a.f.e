import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useScoreEngine } from '@/hooks/useScoreEngine';
import type { RiskLabel } from '@/types/score';

interface LoadedFile {
  name: string;
  ext: string;
  sizeMB: string;
  url: string;
  duration: string;
}

const ACCEPTED = ['audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/ogg', 'audio/flac', 'audio/x-flac', 'audio/aac'];
const MAX_MB = 50;

export function AnalyseRecordingsPage() {
  const navigate = useNavigate();
  const { callState, analyzeCall, reset } = useScoreEngine();
  const [loadedFile, setLoadedFile] = useState<LoadedFile | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [fileError, setFileError] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);

  // ── File handling ───────────────────────────────────────
  function handleFiles(files: FileList | null) {
    setFileError('');
    const file = files?.[0];
    if (!file) return;

    if (!file.type.startsWith('audio/') && !ACCEPTED.includes(file.type)) {
      setFileError('Unsupported format. Please upload an audio file (MP3, WAV, M4A, OGG, FLAC).');
      return;
    }

    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    if (parseFloat(sizeMB) > MAX_MB) {
      setFileError(`File too large (${sizeMB} MB). Maximum is ${MAX_MB} MB.`);
      return;
    }

    const ext = file.name.split('.').pop()?.toUpperCase() ?? 'AUDIO';
    const url = URL.createObjectURL(file);

    const audio = new Audio(url);
    audioRef.current = audio;

    audio.addEventListener('loadedmetadata', () => {
      const dur = audio.duration;
      const m = Math.floor(dur / 60);
      const s = Math.floor(dur % 60).toString().padStart(2, '0');
      setLoadedFile({ name: file.name, ext, sizeMB, url, duration: `${m}:${s}` });
      reset();
      setTimeout(drawWaveform, 150);
    });

    audio.addEventListener('timeupdate', () => {
      setProgress((audio.currentTime / (audio.duration || 1)) * 100);
    });

    audio.addEventListener('ended', () => {
      setIsPlaying(false);
      setProgress(0);
    });
  }

  function drawWaveform() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    canvas.width = canvas.offsetWidth * devicePixelRatio;
    canvas.height = 64 * devicePixelRatio;
    const w = canvas.width, h = canvas.height;
    const bars = 130;
    const mockAmps = Array.from({ length: bars }, () => 0.05 + Math.random() * 0.95);

    ctx.clearRect(0, 0, w, h);
    const bw = w / bars;
    mockAmps.forEach((amp, i) => {
      const bh = amp * h * 0.82;
      const y = (h - bh) / 2;
      ctx.fillStyle = `rgba(99,179,237,${0.12 + amp * 0.18})`;
      ctx.fillRect(i * bw + 0.5, y, bw - 1, bh);
    });
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  }, []);

  function togglePlay() {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play();
      setIsPlaying(true);
    }
  }

  function seekTo(e: React.MouseEvent<HTMLDivElement>) {
    const audio = audioRef.current;
    if (!audio) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    audio.currentTime = pct * audio.duration;
    setProgress(pct * 100);
  }

  function clearFile() {
    audioRef.current?.pause();
    if (loadedFile) URL.revokeObjectURL(loadedFile.url);
    setLoadedFile(null);
    setIsPlaying(false);
    setProgress(0);
    cancelAnimationFrame(animFrameRef.current);
    reset();
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  // ── Mock analysis trigger ────────────────────────────────
  // In production: send audio to POST /api/v1/analyze with audio_b64
  async function runAnalysis() {
    const labels: RiskLabel[] = ['HIGH_RISK', 'PRANK', 'SAFE'];
    const label = labels[Math.floor(Math.random() * 3)];
    await analyzeCall(label);
  }

  const { uiState, data } = callState;
  const isAnalyzing = uiState === 'loading' && loadedFile !== null;

  return (
    <main
      className="min-h-screen pt-[52px]"
      style={{ background: 'radial-gradient(ellipse at 50% 0%, #1a1a2e 0%, #0a0a12 60%)' }}
    >
      <div className="max-w-2xl mx-auto px-6 py-8">

        {/* Header */}
        <motion.div
          className="mb-7"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <div className="flex items-center gap-3 mb-1">
            <button
              onClick={() => navigate(-1)}
              className="font-mono text-[10px] transition-all"
              style={{ color: 'rgba(255,255,255,0.3)', background: 'none', border: 'none', cursor: 'pointer' }}
              onMouseEnter={(e) => ((e.target as HTMLElement).style.color = 'rgba(255,255,255,0.7)')}
              onMouseLeave={(e) => ((e.target as HTMLElement).style.color = 'rgba(255,255,255,0.3)')}
            >
              ← Back
            </button>
          </div>
          <h1 className="font-display text-2xl font-extrabold text-white tracking-tight mb-1">
            Analyse{' '}
            <span style={{ background: 'linear-gradient(90deg,#63b3ed,#68d391)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Recording
            </span>
          </h1>
          <p className="font-mono text-[10px] tracking-[0.12em]" style={{ color: 'rgba(255,255,255,0.25)' }}>
            UPLOAD AN AUDIO FILE — SPECTRAL + INTENT ANALYSIS
          </p>
        </motion.div>

        {/* Upload zone */}
        <AnimatePresence mode="wait">
          {!loadedFile ? (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              <div
                role="button"
                tabIndex={0}
                aria-label="Upload audio file"
                className="rounded-2xl text-center cursor-pointer transition-all"
                style={{
                  border: `2px dashed ${isDragging ? 'rgba(99,179,237,0.6)' : 'rgba(99,179,237,0.22)'}`,
                  background: isDragging ? 'rgba(99,179,237,0.07)' : 'rgba(99,179,237,0.025)',
                  padding: '48px 24px',
                }}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
              >
                <motion.div
                  className="w-14 h-14 rounded-full flex items-center justify-center text-2xl mx-auto mb-4"
                  style={{ background: 'rgba(99,179,237,0.1)', border: '1px solid rgba(99,179,237,0.2)' }}
                  animate={{ y: isDragging ? -4 : 0 }}
                >
                  🎙️
                </motion.div>
                <h3 className="font-display text-base font-bold text-white mb-1.5">
                  {isDragging ? 'Drop to analyse' : 'Drop your recording here'}
                </h3>
                <p className="font-mono text-[10px] mb-4" style={{ color: 'rgba(255,255,255,0.35)' }}>
                  or click to browse your device
                </p>
                <span
                  className="inline-block px-4 py-2 rounded-lg font-mono text-[10px] transition-all"
                  style={{ background: 'rgba(99,179,237,0.1)', border: '1px solid rgba(99,179,237,0.28)', color: '#63b3ed' }}
                >
                  Choose File
                </span>
                <p className="font-mono text-[9px] mt-4 tracking-[0.1em]" style={{ color: 'rgba(255,255,255,0.18)' }}>
                  MP3 · WAV · M4A · OGG · FLAC · UP TO 50MB
                </p>
              </div>

              {fileError && (
                <p className="font-mono text-[10px] text-center mt-3" style={{ color: '#fc8181' }}>
                  {fileError}
                </p>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(e) => handleFiles(e.target.files)}
                aria-label="Audio file input"
              />
            </motion.div>
          ) : (
            /* ── Loaded file UI ─────────────────────────────── */
            <motion.div
              key="loaded"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
              className="space-y-4"
            >
              {/* File card */}
              <div className="rounded-2xl p-5" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>

                {/* File info row */}
                <div className="flex items-center gap-3 mb-4">
                  <div
                    className="w-11 h-11 rounded-xl flex items-center justify-center text-xl flex-shrink-0"
                    style={{ background: 'rgba(99,179,237,0.1)', border: '1px solid rgba(99,179,237,0.2)' }}
                  >
                    🎵
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-display text-sm font-bold text-white truncate">{loadedFile.name}</p>
                    <p className="font-mono text-[9px]" style={{ color: 'rgba(255,255,255,0.3)' }}>
                      {loadedFile.ext} · {loadedFile.sizeMB} MB · {loadedFile.duration}
                    </p>
                  </div>
                  <button
                    onClick={clearFile}
                    className="w-7 h-7 flex items-center justify-center rounded-md font-mono text-xs transition-all flex-shrink-0"
                    style={{ color: 'rgba(255,255,255,0.3)', background: 'none', border: 'none', cursor: 'pointer' }}
                    onMouseEnter={(e) => { (e.target as HTMLElement).style.color = '#fff'; (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.07)'; }}
                    onMouseLeave={(e) => { (e.target as HTMLElement).style.color = 'rgba(255,255,255,0.3)'; (e.target as HTMLElement).style.background = 'none'; }}
                    aria-label="Remove file"
                  >
                    ✕
                  </button>
                </div>

                {/* Waveform */}
                <div className="rounded-lg overflow-hidden mb-4" style={{ background: 'rgba(0,0,0,0.3)', height: 64, position: 'relative' }}>
                  <canvas ref={canvasRef} className="w-full h-full block" aria-label="Audio waveform" />
                  {/* Progress overlay */}
                  <div
                    className="absolute inset-y-0 left-0 pointer-events-none rounded-lg"
                    style={{
                      width: `${progress}%`,
                      background: 'linear-gradient(90deg, rgba(99,179,237,0.15), rgba(99,179,237,0.05))',
                      borderRight: '1px solid rgba(99,179,237,0.4)',
                      transition: 'width 0.1s linear',
                    }}
                  />
                </div>

                {/* Playback controls */}
                <div className="flex items-center gap-3 mb-4">
                  <button
                    onClick={togglePlay}
                    className="w-9 h-9 rounded-full flex items-center justify-center text-xs flex-shrink-0 transition-all"
                    style={{ background: 'rgba(99,179,237,0.12)', border: '1px solid rgba(99,179,237,0.28)', color: '#63b3ed', cursor: 'pointer' }}
                    onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.background = 'rgba(99,179,237,0.22)')}
                    onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.background = 'rgba(99,179,237,0.12)')}
                    aria-label={isPlaying ? 'Pause' : 'Play'}
                  >
                    {isPlaying ? '⏸' : '▶'}
                  </button>

                  <div
                    className="flex-1 h-1 rounded-full cursor-pointer relative"
                    style={{ background: 'rgba(255,255,255,0.08)' }}
                    onClick={seekTo}
                    role="slider"
                    aria-label="Seek"
                    aria-valuenow={Math.round(progress)}
                  >
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${progress}%`,
                        background: 'linear-gradient(90deg,#2b6cb0,#63b3ed)',
                        transition: 'width 0.1s linear',
                      }}
                    />
                  </div>

                  <span className="font-mono text-[9px] flex-shrink-0" style={{ color: 'rgba(255,255,255,0.3)' }}>
                    {loadedFile.duration}
                  </span>
                </div>

                {/* File metadata chips */}
                <div className="grid grid-cols-3 gap-2.5 mb-4">
                  {[
                    { label: 'Format', value: loadedFile.ext },
                    { label: 'Size', value: `${loadedFile.sizeMB} MB` },
                    { label: 'Duration', value: loadedFile.duration },
                  ].map(({ label, value }) => (
                    <div key={label} className="rounded-lg p-2.5" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <p className="font-mono text-[8px] tracking-[0.1em] mb-1" style={{ color: 'rgba(255,255,255,0.25)' }}>{label.toUpperCase()}</p>
                      <p className="font-display text-sm font-bold text-white">{value}</p>
                    </div>
                  ))}
                </div>

                {/* Analyse button */}
                <motion.button
                  onClick={runAnalysis}
                  disabled={isAnalyzing}
                  className="w-full py-3.5 rounded-xl font-mono text-[12px] font-bold tracking-[0.05em] text-white disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ background: 'linear-gradient(135deg, #2b6cb0, #276749)', border: 'none', cursor: 'pointer' }}
                  whileHover={{ scale: isAnalyzing ? 1 : 1.01, filter: isAnalyzing ? 'none' : 'brightness(1.12)' }}
                  whileTap={{ scale: isAnalyzing ? 1 : 0.98 }}
                >
                  {isAnalyzing ? '— Analysing audio —' : '⚡ Analyse Recording'}
                </motion.button>
              </div>

              {/* ── Analysing spinner ────────────────────────── */}
              <AnimatePresence>
                {isAnalyzing && (
                  <motion.div
                    className="rounded-2xl p-6 text-center"
                    style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <div
                      className="w-10 h-10 rounded-full mx-auto mb-3"
                      style={{
                        border: '2px solid transparent',
                        borderTopColor: '#63b3ed',
                        animation: 'spin 0.9s linear infinite',
                      }}
                    />
                    <p className="font-mono text-[10px] tracking-[0.12em]" style={{ color: 'rgba(99,179,237,0.6)' }}>
                      Running spectral + intent analysis...
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* ── Result card ──────────────────────────────── */}
              <AnimatePresence>
                {data && !isAnalyzing && (
                  <ResultCard data={data} label={callState.uiState} />
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}

// ── Result display card ───────────────────────────────────────────────
function ResultCard({ data, label }: { data: NonNullable<typeof import('@/types/score').ScoreResponse extends infer T ? T : never>; label: string }) {
  const cfg = {
    high_risk: {
      bg: 'rgba(229,62,62,0.08)',
      border: 'rgba(229,62,62,0.3)',
      accent: '#fc8181',
      badgeBg: 'rgba(229,62,62,0.15)',
      badgeLabel: 'HIGH RISK',
      msg: 'Synthetic Voice + Coercion Detected. This call shows simultaneous AI voice cloning and financial urgency signals.',
    },
    prank: {
      bg: 'rgba(237,137,54,0.07)',
      border: 'rgba(237,137,54,0.28)',
      accent: '#ed8936',
      badgeBg: 'rgba(237,137,54,0.15)',
      badgeLabel: 'PRANK',
      msg: 'Synthetic voice detected but content is harmless. No coercion or financial urgency signals found.',
    },
    safe: {
      bg: 'rgba(72,187,120,0.07)',
      border: 'rgba(72,187,120,0.25)',
      accent: '#48bb78',
      badgeBg: 'rgba(72,187,120,0.12)',
      badgeLabel: 'SAFE',
      msg: 'No synthetic voice markers detected. Voice patterns match natural human speech.',
    },
  } as const;

  const c = cfg[label as keyof typeof cfg] ?? cfg.safe;

  return (
    <motion.div
      className="rounded-2xl p-5"
      style={{ background: c.bg, border: `1px solid ${c.border}` }}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-display text-base font-extrabold text-white">Analysis Complete</h3>
        <span
          className="px-2.5 py-1 rounded-full font-mono text-[9px] font-bold tracking-[0.08em]"
          style={{ background: c.badgeBg, border: `1px solid ${c.accent}40`, color: c.accent }}
        >
          {c.badgeLabel}
        </span>
      </div>

      <p className="font-mono text-[10px] leading-relaxed mb-4 px-3 py-2.5 rounded-lg" style={{ background: 'rgba(0,0,0,0.2)', color: 'rgba(255,255,255,0.5)' }}>
        {c.msg}
      </p>

      {[
        { label: 'Final Score', value: data.final_score },
        { label: 'Spectral Score', value: data.spectral_score },
        { label: 'Intent Score', value: data.intent_score },
      ].map(({ label: scoreLabel, value }) => (
        <div key={scoreLabel} className="flex items-center gap-3 mb-2.5">
          <span className="font-mono text-[9px] tracking-[0.05em] flex-shrink-0 w-24" style={{ color: 'rgba(255,255,255,0.35)' }}>
            {scoreLabel.toUpperCase()}
          </span>
          <div className="flex-1 h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.07)' }}>
            <motion.div
              className="h-full rounded-full"
              style={{ background: c.accent }}
              initial={{ width: '0%' }}
              animate={{ width: `${value}%` }}
              transition={{ duration: 0.7, ease: 'easeOut', delay: 0.1 }}
            />
          </div>
          <span className="font-mono text-[10px] font-bold flex-shrink-0 w-14 text-right" style={{ color: c.accent }}>
            {Math.round(value)} / 100
          </span>
        </div>
      ))}
    </motion.div>
  );
}