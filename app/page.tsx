"use client";
import { useEffect, useRef, useState } from "react";
import { LoginButton } from "@/components/auth/login-button";
import { Button } from "@/components/ui/button";
import Image from "next/image";
import { motion, useAnimation, useInView } from "framer-motion";

export default function Home() {
  const [firstAnimationComplete, setFirstAnimationComplete] = useState(false);
  const [disableButton, setDisableButton] = useState(false);

  const onClick = () => {
    setDisableButton(true);
  };

  return (
    <main className="flex flex-col justify-center items-center mt-48">
      <AnimatedText
        once
        text="Take a Peek of the Future"
        className="text-8xl font-bold leading-normal text-[#23314C] text-wrap text-center"
        onAnimationComplete={() => setFirstAnimationComplete(true)}
      />
      {firstAnimationComplete && (
        <DelayedAnimatedText
          once
          text="with MolarSupport."
          className="text-8xl font-bold text-[#6D58C6] text-wrap w-3/5 mx-auto text-center"
        />
      )}
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 3, duration: 1 }}
        className="flex flex-col items-center justify-center"
      >
        <p className="text-[#878787] font-medium text-3xl mt-6">
          Leverage the power of Artificial Intelligence
        </p>
        <LoginButton>
          <Button
            variant="solidPurple"
            size="getStarted"
            className="mt-10"
            disabled={disableButton}
            onClick={onClick}
          >
            Get Started
          </Button>
        </LoginButton>
      </motion.div>
    </main>
  );
}

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

type DelayedAnimatedTextProps = AnimatedTextProps & {
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
