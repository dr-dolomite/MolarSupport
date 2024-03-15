import { useEffect, useRef, useState } from "react";
import { AnimatedText } from "@/components/animation/animated-text";

type DelayedAnimatedTextProps = {
  text: string | string[];
  el?: keyof JSX.IntrinsicElements;
  className?: string;
  once?: boolean;
  onAnimationComplete?: () => void;
  delay?: number;
};

export const DelayedAnimatedText = ({
  delay = 1800,
  ...props
}: DelayedAnimatedTextProps) => {
  const [isDisplayed, setIsDisplayed] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsDisplayed(true);
    }, delay);

    return () => clearTimeout(timer);
  }, []);

  return isDisplayed ? <AnimatedText {...props} /> : null;
};