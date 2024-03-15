import { useEffect, useRef } from "react";
import { motion, useAnimation, useInView } from "framer-motion";

type AnimatedTextProps = {
  text: string | string[];
  el?: keyof JSX.IntrinsicElements;
  className?: string;
  once?: boolean;
  onAnimationComplete?: () => void;
};

export const AnimatedText = ({
  text,
  el: Wrapper = "p",
  className,
  once,
  onAnimationComplete,
}: AnimatedTextProps) => {
  const defaultAnimations = {
    hidden: { opacity: 0, y: 10 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.1,
      },
    },
  };

  const control = useAnimation();
  const ref = useRef(null);
  const textArray = Array.isArray(text) ? text : [text];
  const isInView = useInView(ref, { amount: 0.1, once });

  useEffect(() => {
    if (isInView) {
      control.start("visible");
    } else {
      control.start("hidden");
    }
  }, [isInView]);

  useEffect(() => {
    if (once && onAnimationComplete && isInView) {
      onAnimationComplete();
    }
  }, [isInView]);

  return (
    <Wrapper className={className}>
      <span className="sr-only">{text}</span>
      <motion.span
        ref={ref}
        initial="hidden"
        animate={control}
        transition={{ staggerChildren: 0.08 }}
        aria-hidden
      >
        {textArray.map((line, lineIndex) => (
          <span key={lineIndex} className="block">
            {line.split(" ").map((word, wordIndex) => (
              <span key={wordIndex} className="inline-block">
                {word.split("").map((char, charIndex) => (
                  <motion.span
                    key={charIndex}
                    className="inline-block"
                    variants={defaultAnimations}
                  >
                    {char}
                  </motion.span>
                ))}
                <span className="inline-block">&nbsp;</span>
              </span>
            ))}
          </span>
        ))}
      </motion.span>
    </Wrapper>
  );
};
