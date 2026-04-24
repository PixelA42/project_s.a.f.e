import { useState, useRef, useEffect } from 'react';
import { useScoreEngine } from '@/hooks/useScoreEngine';
import type { RiskLabel } from '@/types/score';

interface LoadedFile {
  name: string;
  ext: string;
  sizeMB: string;
  url: string;
  duration: string;
}

const ACCEPTED_TYPES = [
  'audio/mpeg',
  'audio/wav',
  'audio/mp4',
  'audio/ogg',
  'audio/flac',
  'audio/x-flac',
  'audio/aac'
];

const MAX_MB = 50;

export function AnalyseRecordingsPage() {
  const { callState, analyzeCall, reset } = useScoreEngine();

  const [loadedFile, setLoadedFile] = useState<LoadedFile | null>(null);
  const [rawFile, setRawFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  const { uiState, data } = callState;
  const isAnalyzing = uiState === 'loading';

  // ================= FILE HANDLING =================
  function handleFile(file: File) {
    setFileError('');

    if (!ACCEPTED_TYPES.includes(file.type)) {
      setFileError('Unsupported audio format');
      return;
    }

    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > MAX_MB) {
      setFileError(`File too large (${sizeMB.toFixed(1)} MB)`);
      return;
    }

    const url = URL.createObjectURL(file);
    const audio = new Audio(url);

    // Cleanup previous audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
    }

    audioRef.current = audio;

    audio.onloadedmetadata = () => {
      if (!isFinite(audio.duration)) return;

      const m = Math.floor(audio.duration / 60);
      const s = Math.floor(audio.duration % 60)
        .toString()
        .padStart(2, '0');

      setLoadedFile({
        name: file.name,
        ext: file.name.split('.').pop()?.toUpperCase() ?? 'AUDIO',
        sizeMB: sizeMB.toFixed(1),
        url,
        duration: `${m}:${s}`
      });

      setRawFile(file);
      setProgress(0);
      setIsPlaying(false);
      reset();
    };

    audio.ontimeupdate = () => {
      if (!isFinite(audio.duration)) return;
      setProgress((audio.currentTime / audio.duration) * 100);
    };

    audio.onended = () => {
      setIsPlaying(false);
      setProgress(0);
    };
  }

  // Cleanup object URL
  useEffect(() => {
    return () => {
      if (loadedFile?.url) {
        URL.revokeObjectURL(loadedFile.url);
      }
    };
  }, [loadedFile]);

  // ================= BACKEND =================
  async function runAnalysis() {
    if (!rawFile || isAnalyzing) return;

    const formData = new FormData();
    formData.append('file', rawFile);

    try {
      const res = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const result = await res.json();

      const labelMap: Record<string, RiskLabel> = {
        high_risk: 'HIGH_RISK',
        prank: 'PRANK',
        safe: 'SAFE'
      };

      const mappedLabel = labelMap[result.label] ?? 'SAFE';

      await analyzeCall(mappedLabel);
    } catch (err) {
      console.error('Analysis failed:', err);
      setFileError('Analysis failed. Please try again.');
    }
  }

  // ================= AUDIO CONTROL =================
  function togglePlayback() {
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

  // ================= UI =================
  return (
    <main className="min-h-screen p-6 text-white">
      <h1 className="text-xl font-bold mb-4">Analyse Recording</h1>

      {!loadedFile ? (
        <div>
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
          {fileError && <p className="text-red-400 mt-2">{fileError}</p>}
        </div>
      ) : (
        <div className="space-y-4">
          <div>
            <p className="font-semibold">{loadedFile.name}</p>
            <p className="text-sm text-gray-400">
              {loadedFile.ext} • {loadedFile.sizeMB} MB • {loadedFile.duration}
            </p>
          </div>

          <div className="flex items-center gap-4">
            <button onClick={togglePlayback}>
              {isPlaying ? 'Pause' : 'Play'}
            </button>

            <div className="w-full bg-gray-700 h-2 rounded">
              <div
                className="bg-blue-500 h-2 rounded"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          <button
            onClick={runAnalysis}
            disabled={isAnalyzing}
            className="bg-blue-600 px-4 py-2 rounded disabled:opacity-50"
          >
            {isAnalyzing ? 'Analyzing...' : 'Analyse Recording'}
          </button>
        </div>
      )}

      {data && (
        <div className="mt-6">
          <h2 className="font-semibold mb-2">Result</h2>
          <p>Final Score: {data.final_score}</p>
          <p>Spectral Score: {data.spectral_score}</p>
          <p>Intent Score: {data.intent_score}</p>
        </div>
      )}
    </main>
  );
}