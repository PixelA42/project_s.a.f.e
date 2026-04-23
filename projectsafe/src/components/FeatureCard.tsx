/**
 * Feature card with animated canvas background.
 * Each card draws a unique animation that matches its category.
 */
import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';

export type CardVariant = 'spectral' | 'intent' | 'wolf' | 'realtime';

interface FeatureCardProps {
  variant: CardVariant;
  icon: string;
  title: string;
  desc: string;
  className?: string;
}

export function FeatureCard({ variant, icon, title, desc, className }: FeatureCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;

    const setSize = () => {
      canvas.width = canvas.offsetWidth * devicePixelRatio;
      canvas.height = canvas.offsetHeight * devicePixelRatio;
    };
    setSize();

    const w = () => canvas.width;
    const h = () => canvas.height;

    if (variant === 'spectral') {
      const BAR_COUNT = 52;
      const heights = Array.from({ length: BAR_COUNT }, (_, i) =>
        0.2 + 0.8 * Math.abs(Math.sin(i * 0.42))
      );
      const phases = Array.from({ length: BAR_COUNT }, () => Math.random() * Math.PI * 2);
      const speeds = Array.from({ length: BAR_COUNT }, () => 0.018 + Math.random() * 0.028);

      const draw = (t: number) => {
        ctx.clearRect(0, 0, w(), h());
        const bg = ctx.createLinearGradient(0, 0, w(), 0);
        bg.addColorStop(0, 'rgba(10,14,26,1)');
        bg.addColorStop(0.5, 'rgba(15,25,48,1)');
        bg.addColorStop(1, 'rgba(10,14,26,1)');
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, w(), h());
        const bw = w() / BAR_COUNT;
        heights.forEach((base, i) => {
          const amp = base * (0.45 + 0.55 * Math.sin(t * speeds[i] + phases[i]));
          const bh = amp * h() * 0.76;
          const x = i * bw;
          const y = (h() - bh) / 2;
          const intensity = amp;
          const r = Math.round(40 + intensity * 60);
          const g = Math.round(100 + intensity * 80);
          const b = Math.round(190 + intensity * 55);
          ctx.fillStyle = `rgba(${r},${g},${b},${0.35 + intensity * 0.55})`;
          ctx.fillRect(x + 1, y, bw - 2, bh);
        });
        ctx.strokeStyle = 'rgba(99,179,237,0.05)';
        ctx.lineWidth = 0.5;
        [0.25, 0.5, 0.75].forEach((frac) => {
          ctx.beginPath();
          ctx.moveTo(0, h() * frac);
          ctx.lineTo(w(), h() * frac);
          ctx.stroke();
        });
        rafRef.current = requestAnimationFrame(draw);
      };
      rafRef.current = requestAnimationFrame(draw);
    }

    if (variant === 'intent') {
      const nodes = Array.from({ length: 24 }, () => ({
        x: 0.08 * w() + Math.random() * 0.84 * w(),
        y: 0.08 * h() + Math.random() * 0.84 * h(),
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        r: 1.5 + Math.random() * 2.5,
        pulse: Math.random() * Math.PI * 2,
      }));

      const draw = () => {
        ctx.clearRect(0, 0, w(), h());
        ctx.fillStyle = 'rgba(26,14,0,1)';
        ctx.fillRect(0, 0, w(), h());
        const CONNECT_DIST = w() * 0.23;
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const dx = nodes[i].x - nodes[j].x;
            const dy = nodes[i].y - nodes[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < CONNECT_DIST) {
              ctx.strokeStyle = `rgba(237,137,54,${(1 - dist / CONNECT_DIST) * 0.32})`;
              ctx.lineWidth = 0.6;
              ctx.beginPath();
              ctx.moveTo(nodes[i].x, nodes[i].y);
              ctx.lineTo(nodes[j].x, nodes[j].y);
              ctx.stroke();
            }
          }
        }
        nodes.forEach((n) => {
          n.x += n.vx; n.y += n.vy;
          if (n.x < 0 || n.x > w()) n.vx *= -1;
          if (n.y < 0 || n.y > h()) n.vy *= -1;
          n.pulse += 0.035;
          const glow = 0.5 + 0.5 * Math.sin(n.pulse);
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.r * (0.85 + 0.3 * glow), 0, Math.PI * 2);
          ctx.fillStyle = `rgba(237,${Math.round(130 + glow * 40)},54,${0.45 + glow * 0.55})`;
          ctx.fill();
        });
        rafRef.current = requestAnimationFrame(draw);
      };
      rafRef.current = requestAnimationFrame(draw);
    }

    if (variant === 'wolf') {
      const RINGS = [
        { r: 0.14, speed: 0.007, dotCount: 12 },
        { r: 0.24, speed: -0.005, dotCount: 20 },
        { r: 0.34, speed: 0.004, dotCount: 28 },
      ];
      const ringDots = RINGS.flatMap((ring) =>
        Array.from({ length: ring.dotCount }, (_, i) => ({
          ring,
          baseAngle: (i / ring.dotCount) * Math.PI * 2,
        }))
      );
      let t = 0;

      const draw = () => {
        t += 0.01;
        ctx.clearRect(0, 0, w(), h());
        ctx.fillStyle = 'rgba(10,24,16,1)';
        ctx.fillRect(0, 0, w(), h());
        const cx = w() / 2, cy = h() / 2;
        RINGS.forEach((ring) => {
          ctx.beginPath();
          ctx.arc(cx, cy, ring.r * w(), 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(72,187,120,0.07)';
          ctx.lineWidth = 0.5;
          ctx.stroke();
        });
        ringDots.forEach((d) => {
          const angle = d.baseAngle + t * d.ring.speed * 10;
          const r = d.ring.r * w();
          const x = cx + Math.cos(angle) * r;
          const y = cy + Math.sin(angle) * r;
          const glow = 0.4 + 0.6 * Math.abs(Math.sin(angle * 2 + t));
          ctx.beginPath();
          ctx.arc(x, y, 1.5, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(72,187,120,${0.25 + glow * 0.65})`;
          ctx.fill();
        });
        ctx.beginPath();
        ctx.arc(cx, cy, w() * 0.07, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(72,187,120,${0.05 + 0.03 * Math.sin(t * 2)})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(72,187,120,${0.4 + 0.2 * Math.sin(t * 2)})`;
        ctx.lineWidth = 1;
        ctx.stroke();
        rafRef.current = requestAnimationFrame(draw);
      };
      rafRef.current = requestAnimationFrame(draw);
    }

    if (variant === 'realtime') {
      const LINE_COUNT = 5;
      const POINT_COUNT = 90;
      const lineData = Array.from({ length: LINE_COUNT }, (_, i) => ({
        points: new Array(POINT_COUNT).fill(h() / 2) as number[],
        speed: 0.7 + Math.random() * 0.8,
        color: i === 2 ? [252, 129, 129] : i === 1 || i === 3 ? [237, 137, 54] : [99, 179, 237],
        opacity: i === 2 ? 0.9 : 0.3,
        lineWidth: i === 2 ? 1.5 : 0.8,
      }));

      const draw = () => {
        ctx.clearRect(0, 0, w(), h());
        ctx.fillStyle = 'rgba(26,0,0,1)';
        ctx.fillRect(0, 0, w(), h());
        ctx.strokeStyle = 'rgba(252,129,129,0.04)';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= 5; i++) {
          ctx.beginPath();
          ctx.moveTo(0, (h() * i) / 5);
          ctx.lineTo(w(), (h() * i) / 5);
          ctx.stroke();
        }
        lineData.forEach((line) => {
          const newVal = h() * 0.5 + (Math.random() - 0.5) * h() * 0.38 * line.speed;
          line.points.push(newVal);
          line.points.shift();
          ctx.beginPath();
          ctx.moveTo(0, line.points[0]);
          line.points.forEach((p, idx) => {
            ctx.lineTo((idx * w()) / (POINT_COUNT - 1), p);
          });
          ctx.strokeStyle = `rgba(${line.color.join(',')},${line.opacity})`;
          ctx.lineWidth = line.lineWidth;
          ctx.stroke();
        });
        const scanX = ((Date.now() % 2200) / 2200) * w();
        ctx.strokeStyle = 'rgba(252,129,129,0.45)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(scanX, 0);
        ctx.lineTo(scanX, h());
        ctx.stroke();
        rafRef.current = requestAnimationFrame(draw);
      };
      rafRef.current = requestAnimationFrame(draw);
    }

    return () => cancelAnimationFrame(rafRef.current);
  }, [variant]);

  return (
    <motion.div
      className={clsx('rounded-xl overflow-hidden transition-all', className)}
      style={{ border: '1px solid rgba(255,255,255,0.07)', background: '#0a0a12' }}
      whileHover={{ borderColor: 'rgba(255,255,255,0.18)', y: -2 }}
      transition={{ duration: 0.2 }}
    >
      <canvas
        ref={canvasRef}
        className="w-full block"
        style={{ height: 120 }}
        aria-hidden="true"
      />
      <div className="px-5 py-4">
        <span className="text-base mb-3 block" aria-hidden="true">{icon}</span>
        <h3 className="font-display text-[15px] font-bold text-white mb-1.5">{title}</h3>
        <p className="font-mono text-[10px] leading-relaxed" style={{ color: 'rgba(255,255,255,0.38)' }}>
          {desc}
        </p>
      </div>
    </motion.div>
  );
}