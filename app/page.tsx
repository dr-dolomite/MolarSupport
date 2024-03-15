"use client";

import {useState} from "react";

import { motion } from "framer-motion";

import { LoginButton } from "@/components/auth/login-button";
import { Button } from "@/components/ui/button";

import { AnimatedText } from "@/components/animation/animated-text";
import { DelayedAnimatedText } from "@/components/animation/delayed-animated";

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