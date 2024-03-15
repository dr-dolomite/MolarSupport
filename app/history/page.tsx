"use client";

import { DeleteIcon } from "@/components/delete-icon";
import { DeleteModal } from "@/components/modals/delete-modal";

import Link from "next/link";
import React from "react";
import { useEffect, useState } from "react";


import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import { Button } from "@/components/ui/button";

import { useRouter } from "next/navigation";

interface MolarCase {
  session_id: string;
  session_folder: string;
  final_img_filename: string;
  corticalization: string;
  position: string;
  distance: string;
  relation: string;
  risk: string;
  date: string;
}

const HistoryPage = () => {
  const [allSessionData, setAllSessionData] = useState<MolarCase[]>([]);
  // const [data, setData] = useState(null);
  // const [copySuccess, setCopySuccess] = useState(false);
  // const linkInputRef = useRef<HTMLInputElement>(null);
  const [disableViewSession, setDisableViewSession] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [disableButton, setDisableButton] = useState(false);

  const router = useRouter();

  // Call getSessionValues() when the component mounts
  useEffect(() => {
    getAllSessionValues();
  }, []);

  useEffect(() => {
    if (allSessionData && allSessionData.length === 0) {
      setDisableButton(true);
    } else {
      setDisableButton(false);
    }
  }, [allSessionData]);

  // if (!data) {
  //   return <LoadingPage />;
  // }

  async function getAllSessionValues() {
    try {
      const response = await fetch("http://127.0.0.1:8000/api/molarcases", {
        method: "GET",
      });
      // Add artificial delay
      // await new Promise((resolve) => setTimeout(resolve, 1000));

      if (!response.ok) {
        console.error("Error fetching data:", response);
      }

      const data = await response.json();

      if (data.error) {
        console.error("Error fetching data:", data.error);
        return;
      }

      setAllSessionData(Array.isArray(data) ? data : []);
      // setData(data);
      // console.log(data);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  }

  const onDeleteClick = () => {
    setShowDeleteModal(true);
  };

  function closeDeleteModal() {
    setShowDeleteModal(false);
  }

  function deleteDatabase() {
    setDisableButton(true);
    async function deleteAllSessions() {
      try {
        const response = await fetch(
          "http://127.0.0.1:8000/api/molarcases/delete",
          {
            method: "DELETE",
          }
        );

        console.log(response);
        setShowDeleteModal(false);
        setDisableButton(false);

        
        router.push("/home");
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    }

    deleteAllSessions();
    console.log("Database deleted");
  }

  function RedirectToSessionFolder({ session_id }: { session_id: string }) {
    setDisableViewSession(true);
    router.push(`/result/${session_id}`);
  }

  const getRiskStyles = (risk: string) => {
    let backgroundColor = "";
    let textColor = "";

    switch (risk) {
      case "N.0 (Non-determinant)":
        backgroundColor = "#58C5C080";
        textColor = "#30BCB5";
        break;
      case "N.1 (Low)":
        backgroundColor = "#87CF7580";
        textColor = "#6ABC55";
        break;
      case "N.2 (Medium)":
        backgroundColor = "#F0993F80";
        textColor = "#F0871A";
        break;
      case "N.3 (High)":
        backgroundColor = "#E9503980";
        textColor = "#EC432A";
        break;
      default:
        backgroundColor = "#58C5C080";
        textColor = "#30BCB5";
    }

    return { backgroundColor, textColor };
  };

  return (
    <div className="bg-[#f5f5f7] flex flex-row justify-center w-full">
      {showDeleteModal && (
        <DeleteModal
          onClose={closeDeleteModal}
          deleteDatabase={deleteDatabase}
          disableButton={disableButton}
        />
      )}
      <div className="bg-[#f5f5f7] w-[1440px] h-[1024px] relative">
        <div className="absolute top-[60px] left-[652px] font-bold text-[#9641e7] text-[40px] text-center tracking-[0] leading-[normal]">
          History
        </div>
        <div className="absolute top-[70px] right-[52px]">
          <DeleteIcon onClick={onDeleteClick} disabled={disableButton} />
        </div>
        <div className="inline-flex items-center gap-[16px] px-[24px] py-0 absolute top-[160px] left-[46px] bg-white rounded-[16px] shadow-[0px_3px_6px_#00000014,0px_10px_10px_#00000012,0px_24px_14px_#0000000a,0px_42px_17px_#00000003,0px_65px_18px_transparent]">
          <div className="flex w-[172px] items-center gap-[8px] px-[8px] py-[16px] relative">
            <div className="relative w-fit mt-[-1.00px] font-normal text-[#5a6579] text-[20px] text-center tracking-[0] leading-[normal]">
              Date
            </div>
          </div>
          <div className="flex w-[172px] items-center gap-[8px] px-[8px] py-[16px] relative">
            <div className="relative w-fit mt-[-1.00px] font-normal text-[#5a6579] text-[20px] text-center tracking-[0] leading-[normal]">
              M3 - MC relation
            </div>
          </div>
          <div className="flex w-[172px] items-center justify-center gap-[8px] px-[8px] py-[9px] relative">
            <img src="/icons/posIcon.png" alt="corti-icon" className="size-5" />
            <div className="relative w-fit mt-[-1.00px] font-normal text-[#5a6579] text-[20px] text-center tracking-[0] leading-[normal]">
              Position
            </div>
          </div>
          <div className="flex w-[172px] items-center justify-center gap-[8px] px-[8px] py-[16px] relative">
            <div className="relative w-[24px] h-[24px] bg-[100%_100%]">
              <div className="relative w-[30px] h-[10px] top-[3px] left-[12px] bg-white rounded-[5px]">
                <img
                  src="/icons/cortiIcon.png"
                  alt="corti-icon"
                  className="size-4"
                />
              </div>
            </div>
            <div className="relative w-fit mt-[-1.00px] font-normal text-[#5a6579] text-[20px] text-center tracking-[0] leading-[normal]">
              Interruption
            </div>
          </div>
          <div className="flex w-[172px] items-center justify-center gap-[8px] px-[8px] py-[16px] relative">
            <img
              src="/icons/distIcon.png"
              alt="corti-icon"
              className="size-5"
            />
            <div className="relative w-fit mt-[-1.00px] font-normal text-[#5a6579] text-[20px] text-center tracking-[0] leading-[normal]">
              Distance
            </div>
          </div>
          <div className="flex w-[172px] items-center justify-center gap-[8px] px-[8px] py-[16px] relative">
            <div className="relative w-fit mt-[-1.00px] font-normal text-[#5a6579] text-[20px] text-center tracking-[0] leading-[normal]">
              Risk
            </div>
          </div>
          <div className="flex w-[172px] items-center justify-center gap-[8px] px-[8px] py-[16px] relative">
            <div className="relative w-fit mt-[-1.00px] font-normal text-[#5a6579] text-[20px] text-center tracking-[0] leading-[normal]">
              Details
            </div>
          </div>
        </div>
        <div className="overflow-y-scroll h-[575px] w-[1348px] overflow-x-hidden inline-flex flex-col items-start px-[24px] py-0 absolute top-[250px] left-[46px] bg-white rounded-[16px] shadow-[0px_3px_6px_#00000014,0px_10px_10px_#00000012,0px_24px_14px_#0000000a,0px_42px_17px_#00000003,0px_65px_18px_transparent]">
          {allSessionData && allSessionData.length === 0 ? (
            <div className="text-center text-gray-500 mt-12 ml-12 font-bold text-2xl">
              Database is empty.
            </div>
          ) : (
            allSessionData.map((sessionData, key) => {
              return (
                <div
                  key={key}
                  className="inline-flex items-center gap-[16px] px-0 py-[24px] relative flex-[0_0_auto]"
                >
                  <div className="flex w-[172px] items-center gap-[16px] px-[8px] py-[16px] relative">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <div className="relative w-fit mt-[-1.00px] font-bold text-[#23314c] text-[20px] text-center tracking-[0] leading-[normal]">
                            {sessionData.date}
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          {sessionData.session_id}
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <div className="flex w-[172px] items-center justify-center gap-[16px] px-[8px] py-[16px] relative">
                    <div className="relative w-fit mt-[-1.00px] font-bold text-[#23314c] text-[20px] text-center tracking-[0] leading-[normal]">
                      {sessionData.relation}
                    </div>
                  </div>
                  <div className="flex w-[172px] items-center justify-center gap-[16px] px-[8px] py-[16px] relative">
                    <div className="relative w-fit mt-[-1.00px] font-bold text-[#23314c] text-[20px] text-center tracking-[0] leading-[normal]">
                      {sessionData.position}
                    </div>
                  </div>
                  <div className="flex w-[172px] items-center justify-center gap-[16px] px-[8px] py-[16px] relative">
                    <div className="relative w-fit mt-[-1.00px] font-bold text-[#23314c] text-[20px] text-center tracking-[0] leading-[normal]">
                      {sessionData.corticalization}
                    </div>
                  </div>
                  <div className="flex w-[172px] items-center justify-center gap-[16px] px-[8px] py-[16px] relative">
                    <div className="relative w-fit mt-[-1.00px] font-bold text-[#23314c] text-[20px] text-center tracking-[0] leading-[normal]">
                      {sessionData.distance}
                    </div>
                  </div>
                  <div className="flex w-[172px] items-center justify-center gap-[16px] px-[8px] py-[16px] relative">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <div
                            className="inline-flex items-center justify-center gap-[10px] px-[32px] py-[4px] relative flex-[0_0_auto] bg-[#87cf7540] rounded-[100px]"
                            style={{
                              backgroundColor: getRiskStyles(sessionData.risk)
                                .backgroundColor,
                            }}
                          >
                            <div
                              className="relative w-fit mt-[-1.00px] font-extrabold text-[#69bc54] text-[20px] text-center tracking-[0] leading-[normal]"
                              style={{
                                color: getRiskStyles(sessionData.risk)
                                  .textColor,
                              }}
                            >
                              {(() => {
                                if (
                                  sessionData.risk === "N.0 (Non-determinant)"
                                ) {
                                  return "N.0";
                                } else if (sessionData.risk === "N.1 (Low)") {
                                  return "N.1";
                                } else if (
                                  sessionData.risk === "N.2 (Medium)"
                                ) {
                                  return "N.2";
                                } else if (sessionData.risk === "N.3 (High)") {
                                  return "N.3";
                                } else {
                                  return "Unknown";
                                }
                              })()}
                            </div>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>{sessionData.risk}</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>

                  <Button
                    variant="purpleButton"
                    className="rounded-[16px]"
                    onClick={() =>
                      RedirectToSessionFolder({
                        session_id: sessionData.session_id,
                      })
                    }
                    disabled={disableViewSession}
                  >
                    View Session
                  </Button>
                </div>
              );
            })
          )}
        </div>
        <Link href="/home">
          <div className="flex w-[192px] items-center gap-[16px] px-[24px] py-[16px] absolute top-[60px] left-[47px] rounded-[16px] overflow-hidden border-2 border-solid border-[#6d58c6]">
            <svg
              width="32"
              height="33"
              viewBox="0 0 32 33"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M20.0001 25.8327L10.6667 16.4993L20.0001 7.16602"
                stroke="#6D58C6"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>

            <div className="relative w-fit mt-[-2.00px] font-bold text-[#6d58c6] text-[24px] text-center tracking-[0] leading-[normal]">
              Back
            </div>
          </div>
        </Link>
      </div>
    </div>
  );
};

export default HistoryPage;