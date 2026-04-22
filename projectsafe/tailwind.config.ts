import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"Space Mono"', 'monospace'],
        display: ['Syne', 'sans-serif'],
      },
      colors: {
        safe: {
          bg: '#0d1f18',
          accent: '#48bb78',
          muted: '#68d391',
          border: 'rgba(72,187,120,0.25)',
        },
        prank: {
          bg: '#1a1400',
          accent: '#ed8936',
          muted: '#fbd38d',
          border: 'rgba(237,137,54,0.3)',
        },
        threat: {
          bg: '#1a0000',
          accent: '#e53e3e',
          muted: '#fc8181',
          border: 'rgba(229,62,62,0.4)',
        },
        load: {
          bg: '#0f0f1a',
          accent: '#63b3ed',
        },
      },
      animation: {
        'pulse-ring': 'pulse-ring 1.4s ease-in-out infinite',
        'pulse-ring-fast': 'pulse-ring 0.9s ease-in-out infinite',
        'scan': 'scan 2.5s linear infinite',
        'blink': 'blink 1.2s ease-in-out infinite',
        'blink-fast': 'blink 0.7s ease-in-out infinite',
        'spin-slow': 'spin 3s linear infinite',
        'fade-in-up': 'fadeInUp 0.5s ease forwards',
      },
      keyframes: {
        'pulse-ring': {
          '0%, 100%': { transform: 'scale(1)', opacity: '0.8' },
          '50%': { transform: 'scale(1.18)', opacity: '0.25' },
        },
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(200%)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.15' },
        },
        fadeInUp: {
          from: { opacity: '0', transform: 'translateY(12px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;