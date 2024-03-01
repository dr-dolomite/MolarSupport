"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";

import {
  Card,
  CardHeader,
  CardFooter,
  CardContent,
  CardDescription,
  CardTitle,
} from "@/components/ui/card";

const ExampleCard = () => {
  const [clicked, setClicked] = useState(false);

  const router = useRouter();

  return (
    <Card className="px-6 py-2 shadow-md drop-shadow-md flex flex-col justify-center">
      <CardHeader>
        <CardTitle>
          <p className="text-[#878787] text-xl font-normal">
            You can try these sample cases:
          </p>
        </CardTitle>
        <CardContent className="flex flex-row items-center justify-center py-2 gap-x-6">
          <Button
            variant="ghost"
            onClick={() => {
              if (!clicked) {
                setClicked(true);
                router.push("/sample/1");
              }
            }}
            disabled={clicked}
            className="size-44 "
          >
            <img
              src="/sample/m3sample-1.jpg"
              alt="sample-1"
              className="object-cover"
            />
          </Button>

          <Button
            variant="ghost"
            onClick={() => {
              if (!clicked) {
                setClicked(true);
                router.push("/sample/2");
              }
            }}
            disabled={clicked}
            className="size-44 "
          >
            <img
              src="/sample/m3sample-2.jpg"
              alt="sample-2"
              className="object-cover cursor-pointer"
            />
          </Button>

          <Button
            variant="ghost"
            onClick={() => {
              if (!clicked) {
                setClicked(true);
                router.push("/sample/3");
              }
            }}
            disabled={clicked}
            className="size-44 "
          >
            <img
              src="/sample/m3sample-3.jpg"
              alt="sample-3"
              className=" object-cover cursor-pointer"
            />
          </Button>

          <Button
            variant="ghost"
            onClick={() => {
              if (!clicked) {
                setClicked(true);
                router.push("/sample/4");
              }
            }}
            disabled={clicked}
            className="size-44 "
          >
            <img
              src="/sample/m3sample-4.jpg"
              alt="sample-4"
              className="object-cover cursor-pointer"
            />
          </Button>
        </CardContent>
      </CardHeader>
    </Card>
  );
};

export default ExampleCard;
