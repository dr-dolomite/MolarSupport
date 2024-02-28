"use client";

import { RiSquareFill } from "react-icons/ri";

import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FaArrowsRotate } from "react-icons/fa6";
import { FaRegFolderOpen } from "react-icons/fa";
import { useEffect, useState } from "react";

import { useRouter } from "next/navigation";

export default function ResultPage({
  params,
}: {
  params: { resultId: string };
}) {
  const router = useRouter();
  const [data, setData] = useState(null);

  // Call getSessionValues() when the component mounts
  useEffect(() => {
    async function getSessionValues() {
      try {
        const response = await fetch(
          `http://127.0.0.1:8000/api/molarcase/${params.resultId}`,
          {
            method: "GET",
          }
        );
        const data = await response.json();
        setData(data);
        console.log(data);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    }

    getSessionValues();
  }, []);

  if (!data) {
    return <h1>Loading</h1>;
  }

  const {
    relation,
    risk,
    distance,
    position,
    corticalization,
    session_folder,
    final_img_filename,
  } = data;

  const imagePath = `${session_folder}/${final_img_filename}`;
  console.log("Image Path: ", imagePath);

  return (
    <div className="flex flex-col items-center justify-center gap-y-16">
      <h1 className="text-6xl capitilized text-[#9641E7] text-center font-bold leading-normal">
        Results
      </h1>
      <div className="flex flex-row gap-x-16">
        <div className="flex flex-col gap-y-8">
          <Card className="px-20 py-7">
            <div className="flex flex-row gap-x-8 items-center">
              <img
                src="/logo/molar-logo-2.svg"
                alt="molar-logo"
                className="w-24"
              />

              <div className="flex flex-col gap-y-1">
                <p className="text-[24px] font-bold">Hi, I'm Molar Support</p>
                <p className="text-[20px] font-medium">Your assesment buddy</p>
              </div>
            </div>
          </Card>

          <Card className="p-8 flex items-center justify-center">
            <img
              src="/temp-result/temp.jpg"
              alt="result-image"
              className="size-54 object-cover rounded-md shadow-md drop-shadow-lg"
            />
          </Card>
        </div>

        <div className="flex flex-col gap-y-8">
          <Card className="px-20 py-6">
            <div className="flex flex-row gap-x-2 items-center">
              <RiSquareFill className="rounded-[14px] text-[50px] text-[#419e59] mr-4" />
              <p className="text-[#23314C] font-bold text-[1.2rem]">
                Mandibular Third Molar
              </p>
            </div>

            <div className="flex flex-row gap-x-2 items-center">
              <RiSquareFill className="rounded-[14px] text-[50px] text-[#c94dc9] mr-4" />
              <p className="text-[#23314C] font-bold text-[1.2rem]">
                Mandibular Canal
              </p>
            </div>
          </Card>

          <Card className="py-12 px-24 flex flex-col">
            <div className="flex flex-row flex-wrap items-center gap-x-32">
              {/* FIRST COLUMN */}
              <div className="flex flex-col gap-y-12">
                {/* For M3-MC RELATION */}
                <div className="space-y-2">
                  <p className="text-[#5A6579] font-semibold text-[1.2rem]">
                    M3-MC Relation:
                  </p>
                  <p className="text-[#23314C] font-bold text-[2rem]">
                    {relation}
                  </p>
                </div>

                {/* POSITION */}
                <div className="space-y-2">
                  <p className="text-[#5A6579] font-semibold text-[1.2rem]">
                    Position:
                  </p>
                  <div className="flex flex-row flex-auto flex-wrap items-center">
                    <img
                      src="/icons/posIcon.png"
                      alt="corti-icon"
                      className="size-8 mr-4"
                    />
                    <p className="text-[#23314C] font-bold text-[2rem]">
                      {position}
                    </p>
                  </div>
                </div>
              </div>
              {/* SECOND COLUMN */}
              <div className="flex flex-col gap-y-12">
                {/* INTERRUPTION */}
                <div className="space-y-2">
                  <p className="text-[#5A6579] font-semibold text-[1.2rem]">
                    Interruption:
                  </p>
                  <div className="flex flex-row flex-auto flex-wrap items-center">
                    <img
                      src="/icons/cortiIcon.png"
                      alt="corti-icon"
                      className="size-8 mr-4"
                    />
                    <p className="text-[#23314C] font-bold text-[2rem]">
                      {corticalization}
                    </p>
                  </div>
                </div>
                {/* DISTANCE */}
                <div className="space-y-2">
                  <p className="text-[#5A6579] font-semibold text-[1.2rem]">
                    Distance:
                  </p>
                  <div className="flex flex-row flex-auto flex-wrap items-center">
                    <img
                      src="/icons/distIcon.png"
                      alt="corti-icon"
                      className="size-8 mr-4"
                    />
                    <p className="text-[#23314C] font-bold text-[2rem]">
                      {distance}
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="border-b-2 border-[#0000001A] mt-8" />
            <div className="rounded-full px-6 py-4 mt-8 bg-[#e1f3dc]">
              <p className="text-[#6ABC55] font-extrabold text-center text-3xl">
                {risk}
              </p>
            </div>

            <div className="flex flex-row gap-x-4">
              <Button variant="successBack">
                View Folder Results <FaRegFolderOpen className="ml-2" />
              </Button>
            </div>

            <Button
              variant="submitButton"
              size="submitButton"
              className="mt-12"
              onClick={() => router.push("/home")}
            >
              <FaArrowsRotate className="mr-4 text-extrabold" />
              Evaluate another case
            </Button>
          </Card>
        </div>
      </div>
    </div>
  );
}

// export default ResultPage;
