import React from "react";

import {
  Card,
  CardHeader,
  CardFooter,
  CardContent,
  CardDescription,
  CardTitle,
} from "@/components/ui/card";

const AboutCard = () => {
  return (
    <Card className="w-full p-6 py-10 shadow-md drop-shadow-md">
      <CardHeader>
        <CardTitle className="text-[#23314C] text-6xl font-extrabold leading-tight">
          Mandibular Third Molar (M3) Nerve Injury <span className="text-[#6D58C6]">Risk Evaluator</span>
        </CardTitle>
        <CardContent className="p-0 m-0">
            <p className="font-semibold text-[24px] text-[#878787] py-8">
            Molar Support is an advanced application designed for precise preoperative nerve injury risk assessment in mandibular third molar extraction.
            </p>
        </CardContent>
      </CardHeader>
    </Card>
  );
};

export default AboutCard;
