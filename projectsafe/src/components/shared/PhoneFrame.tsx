import { type ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';

interface PhoneFrameProps {
  background: string;
  children: ReactNode;
  className?: string;
}

export function PhoneFrame({ background, children, className }: PhoneFrameProps) {
  return (
    <div
      className={clsx(
        'relative mx-auto w-[340px] min-h-[580px] rounded-[40px] overflow-hidden',
        'shadow-[0_40px_80px_rgba(0,0,0,0.6),0_0_0_1px_rgba(255,255,255,0.06)]',
        className
      )}
      style={{ background }}
    >
      {/* Notch bar */}
      <div className="absolute top-3 left-1/2 -translate-x-1/2 w-[90px] h-[6px] rounded-full bg-white/10 z-10" />

      {/* Content area */}
      <AnimatePresence mode="wait">
        <motion.div
          key={background}
          className="pt-7 h-full"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.35, ease: 'easeInOut' }}
        >
          {children}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}