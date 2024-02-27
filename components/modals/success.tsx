import React from "react";
import { Card } from "@/components/ui/card";
import { TiWarningOutline } from "react-icons/ti";

const SuccessModal = () => {
  return (
    <div className="fixed top-0 left-0 right-0 bottom-0 backdrop-brightness-[35%] z-50 flex items-center justify-center">
      <Card className="h-4/6 w-3/6 flex flex-col items-center p-8">
        <div className="rounded-[48px] border-[#3F33751A] bg-[#3F337540] flex items-center justify-center border-8 size-60 p-4">
          <TiWarningOutline className="w-full h-full text-[#3F3375]" />
        </div>

        <h1 className="text-[#3F3375] text-center text-[4rem] font-extrabold pt-12">
          Oops!
        </h1>
        <p className="text-[#667085] font-semibold text-[2rem] text-center w-3/6">Image upload was not CBCT M3 slice Image</p>
      </Card>
    </div>
  );
};

export default SuccessModal;
