interface WaveBarsProps {
  color: string;
  count?: number;
}

export function WaveBars({ color, count = 5 }: WaveBarsProps) {
  const heights = [35, 70, 100, 60, 45];
  const delays = [0, 0.1, 0.2, 0.15, 0.05];

  return (
    <div className="flex items-end gap-[3px]" style={{ height: 28 }}>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="w-1 rounded-sm animate-blink"
          style={{
            height: `${heights[i % heights.length]}%`,
            background: color,
            animationDelay: `${delays[i % delays.length]}s`,
          }}
        />
      ))}
    </div>
  );
}