"use client";

import Link from 'next/link'
import React from 'react'
import { useEffect, useState, useRef } from "react";

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

import { Button } from "@/components/ui/button"

import { CopyIcon } from "@radix-ui/react-icons"

import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface MolarCase {
  session_id: string;
  session_folder: string;
  final_img_filename: string;
  corticalization: string;
  position: string;
  distance: string;
  relation: string;
  risk: string;
}

const HistoryPage = () => {
  const [allSessionData, setAllSessionData] = useState<MolarCase[]>([]);;
  const [copySuccess, setCopySuccess] = useState(false);
  const linkInputRef = useRef<HTMLInputElement>(null);

  // Call getSessionValues() when the component mounts
  useEffect(() => {
    async function getAllSessionValues() {
      try {
        const response = await fetch(
          "http://127.0.0.1:8000/api/molarcases",
          {
            method: "GET",
          }
        );
        const data = await response.json();
        setAllSessionData(data);
        // console.log(data);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    }

    getAllSessionValues();
  }, []);

  const copyToClipboard = () => {
    if (linkInputRef.current) {
      linkInputRef.current.select();
      document.execCommand('copy');
      setCopySuccess(true);
      setTimeout(() => {
        setCopySuccess(false);
      }, 2000); // Reset copy success indicator after 2 seconds
    }
  };

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
      <div className="bg-[#f5f5f7] w-[1440px] h-[1024px] relative">
        <div className="absolute top-[60px] left-[652px] font-bold text-[#9641e7] text-[40px] text-center tracking-[0] leading-[normal]">
          History
        </div>
        <div className="inline-flex items-center gap-[16px] px-[24px] py-0 absolute top-[160px] left-[46px] bg-white rounded-[16px] shadow-[0px_3px_6px_#00000014,0px_10px_10px_#00000012,0px_24px_14px_#0000000a,0px_42px_17px_#00000003,0px_65px_18px_transparent]">
          <div className="flex w-[172px] items-center gap-[8px] px-[8px] py-[16px] relative">
            <div className="relative w-fit mt-[-1.00px] font-normal text-[#5a6579] text-[20px] text-center tracking-[0] leading-[normal]">
              Session ID
            </div>
          </div>
          <div className="flex w-[172px] items-center gap-[8px] px-[8px] py-[16px] relative">
            <div className="relative w-fit mt-[-1.00px] font-normal text-[#5a6579] text-[20px] text-center tracking-[0] leading-[normal]">
              M3 - MC relation
            </div>
          </div>
          <div className="flex w-[172px] items-center justify-center gap-[8px] px-[8px] py-[9px] relative">
            <img
              src="/icons/posIcon.png"
              alt="corti-icon"
              className="size-5"
            />
            <div className="relative w-fit mt-[-1.00px] font-normal text-[#5a6579] text-[20px] text-center tracking-[0] leading-[normal]">
              Position
            </div>
          </div>
          <div className="flex w-[172px] items-center justify-center gap-[8px] px-[8px] py-[16px] relative">
            <div className="relative w-[24px] h-[24px] bg-[url(/vector.svg)] bg-[100%_100%]">
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
          {!allSessionData ? (
            <h1>Loading</h1>
          ) :
          (
          allSessionData.map((sessionData, key) => {
              return (
            <div key={key} className="inline-flex items-center gap-[16px] px-0 py-[24px] relative flex-[0_0_auto]">
              <div className="flex w-[172px] items-center gap-[16px] px-[8px] py-[16px] relative">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="relative w-fit mt-[-1.00px] font-bold text-[#23314c] text-[20px] text-center tracking-[0] leading-[normal]">
                          {sessionData.session_id.substring(0, 9) + "..." }
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
                        <div className="inline-flex items-center justify-center gap-[10px] px-[32px] py-[4px] relative flex-[0_0_auto] bg-[#87cf7540] rounded-[100px]" style={{ backgroundColor: getRiskStyles(sessionData.risk).backgroundColor }} >
                          <div className="relative w-fit mt-[-1.00px] font-extrabold text-[#69bc54] text-[20px] text-center tracking-[0] leading-[normal]" style={{ color: getRiskStyles(sessionData.risk).textColor }}>
                            {(
                              () => {
                                if (sessionData.risk === "N.0 (Non-determinant)") {
                                  return "N.0";
                                } else if (sessionData.risk === "N.1 (Low)") {
                                  return "N.1";
                                } else if (sessionData.risk === "N.2 (Medium)") {
                                  return "N.2";
                                } else if (sessionData.risk === "N.3 (High)") {
                                  return "N.3";
                                } else {
                                  return "Unknown";
                                }
                              }
                            )()}
                          </div>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        {sessionData.risk}
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
              </div>
              {/* <div className="cursor-pointer flex w-[172px] items-center justify-center gap-[16px] px-[8px] py-[16px] relative">
                <div className="inline-flex items-center justify-center gap-[10px] px-[16px] py-[4px] relative flex-[0_0_auto] bg-[#e2def4] rounded-[100px] border border-solid border-[#6d58c6]">
                  <div className="relative w-fit mt-[-1.00px] font-extrabold text-[#6d58c6] text-[20px] text-center tracking-[0] leading-[normal]">
                    View Folder
                  </div>
                </div>
              </div> */}
              
              
              <Dialog>
                <DialogTrigger asChild>
                  <div className="cursor-pointer flex w-[172px] items-center justify-center gap-[16px] px-[8px] py-[16px] relative">
                    <div className="inline-flex items-center justify-center gap-[10px] px-[16px] py-[4px] relative flex-[0_0_auto] bg-[#e2def4] rounded-[100px] border border-solid border-[#6d58c6]">
                      <div className="relative w-fit mt-[-1.00px] font-extrabold text-[#6d58c6] text-[20px] text-center tracking-[0] leading-[normal]">
                        View Folder
                      </div>
                    </div>
                  </div>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle>Share link</DialogTitle>
                    <DialogDescription>
                      Anyone who has this link will be able to view this.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="flex items-center space-x-2">
                    <div className="grid flex-1 gap-2">
                      <Label htmlFor="link" className="sr-only">
                        Link
                      </Label>
                      <Input
                        ref={linkInputRef}
                        id="link"
                        defaultValue="https://ui.shadcn.com/docs/installation"
                        readOnly
                      />
                    </div>
                    <Button type="button" size="sm" className="px-3" onClick={copyToClipboard}>
                      <span className="sr-only">Copy</span>
                      <CopyIcon className="h-4 w-4" />
                    </Button>
                    {copySuccess && <div className="text-[#69bc54]">Copied to clipboard!</div>}
                  </div>
                  <DialogFooter className="sm:justify-start">
                    <DialogClose asChild>
                      <Button type="button" variant="secondary">
                        Close
                      </Button>
                    </DialogClose>
                  </DialogFooter>
                </DialogContent>
              </Dialog>

            </div>
            )
          })
        )}
        </div>
        <Link href="/">
          <div className="flex w-[192px] items-center gap-[16px] px-[24px] py-[16px] absolute top-[60px] left-[47px] rounded-[16px] overflow-hidden border-2 border-solid border-[#6d58c6]">
            <svg width="32" height="33" viewBox="0 0 32 33" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M20.0001 25.8327L10.6667 16.4993L20.0001 7.16602" stroke="#6D58C6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>

            
              <div className="relative w-fit mt-[-2.00px] font-bold text-[#6d58c6] text-[24px] text-center tracking-[0] leading-[normal]">
                Back
              </div>
            
          </div>
        </Link>
      </div>
    </div>
  )
}

export default HistoryPage