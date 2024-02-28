"use client";

import React from "react";
import { Card } from "@/components/ui/card";
import { TiWarningOutline } from "react-icons/ti";
import { Button } from "@/components/ui/button";

interface ErrorModalProps {
    onClose: () => void;
    error: string;
}

const ErrorModal = ({onClose, error}:ErrorModalProps) => {
  
  return (
    // If show is true display the div below
    // Else, display nothing
    <div className="fixed top-0 left-0 right-0 bottom-0 backdrop-blur-sm backdrop-brightness-[30%] z-50 flex items-center justify-center">
      <Card className="size-fit flex flex-col items-center p-16 gap-y-10 drop-shadow-lg shadow-md">
        <div className="rounded-[48px] border-[#3F33751A] bg-[#3F337540] flex items-center justify-center border-8 size-48 p-4">
          <TiWarningOutline className="w-full h-full text-[#3F3375]" />
        </div>

        <div className="space-y-2 flex flex-col items-center justify-center">
        <h1 className="text-[#3F3375] text-center text-[4rem] font-extrabold leading-none">
          Oops!
        </h1>
        <p className="text-[#667085] font-semibold text-[1.6rem] text-center w-3/5 mt-4">
          {error}
        </p>
        </div>
        
        <Button variant="errorButton" size="errorButton" onClick={onClose}>
          Try again
        </Button>
      </Card>
    </div>
  );
};

export default ErrorModal;
