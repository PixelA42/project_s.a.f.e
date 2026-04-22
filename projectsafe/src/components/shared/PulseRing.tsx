import clsx from 'clsx';

interface PulseRingProps {
  size: number;
  color: string;
  speed?: 'normal' | 'fast';
  rings?: number;
  children?: React.ReactNode;
}

export function PulseRing({ size, color, speed = 'normal', rings = 2, children }: PulseRingProps) {
  const animClass = speed === 'fast' ? 'animate-pulse-ring-fast' : 'animate-pulse-ring';

  return (
    <div className="relative" style={{ width: size, height: size }}>
      {Array.from({ length: rings }).map((_, i) => (
        <div
          key={i}
          className={clsx('absolute rounded-full border', animClass)}
          style={{
            inset: -(i + 1) * 8,
            borderColor: color,
            opacity: 0.25 - i * 0.05,
            animationDelay: `${i * 0.25}s`,
          }}
        />
      ))}
      <div
        className="relative w-full h-full rounded-full flex items-center justify-center overflow-hidden"
        style={{ border: `2px solid ${color}`, background: `${color}18` }}
      >
        {children}
      </div>
    </div>
  );
}