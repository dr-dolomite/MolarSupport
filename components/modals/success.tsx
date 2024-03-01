"use client";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { IoMdCheckmarkCircleOutline } from "react-icons/io";
import { Rings } from "react-loading-icons";
import LoadingPage from "@/components/loading";

interface SuccessModalProps {
  onBack: () => void;
  onSubmit: () => void;
  showLoadingIcon: boolean;
}

const SuccessModal = ({
  onBack,
  onSubmit,
  showLoadingIcon,
}: SuccessModalProps) => {
  const [submitClicked, setSubmitClicked] = useState(false);

  const handleSubmit = () => {
    setSubmitClicked(true);
    onSubmit();
  };

  return (
    <>
      {!submitClicked && (
        <div className="fixed top-0 left-0 right-0 bottom-0 backdrop-blur-sm backdrop-brightness-[30%] z-50 flex items-center justify-center">
          <Card className="size-fit max-w-3xl flex flex-col items-center p-16 drop-shadow-lg shadow-md">
            <div className="rounded-[48px] border-[#ECFDF3] bg-[#D1FADF] flex items-center justify-center border-8 size-48 p-4">
              <IoMdCheckmarkCircleOutline className="w-full h-full text-[#039855]" />
            </div>

            <h1 className="text-[#039855] text-center text-[4rem] font-extrabold leading-none mt-12">
              Nice!
            </h1>
            <p className="text-[#667085] font-semibold text-[1.6rem] text-center mt-16">
              Are you certain that both the M3 and MC images correspond to
              identical slices?
            </p>

            <div className="flex flex-row gap-x-8 items-center mt-8">
              <Button
                variant="successBack"
                size="successButton"
                onClick={onBack}
              >
                Back
              </Button>
              <Button
                variant="successButton"
                size="successButton"
                onClick={handleSubmit}
                disabled={showLoadingIcon}
              >
                {showLoadingIcon && <Rings className="mr-2" />} Let’s start
                assessment
              </Button>
            </div>
          </Card>
        </div>
      )}
      {submitClicked && <LoadingPage />}
    </>
  );
};

export default SuccessModal;
