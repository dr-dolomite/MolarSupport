"use client";

import { useRouter } from "next/navigation";
import { userInfo } from "os";

interface SubmitButtonProps {
  children: React.ReactNode;
  mode?: "modal" | "redirect";
  asChild?: boolean;
}

export const SubmitButton = ({
  children,
  mode = "redirect",
  asChild,
}: SubmitButtonProps) => {
  const router = useRouter();
  const onClick = () => {
    router.push("/result");
  };

  if (mode == "modal") {
    return <span>TODO: modal</span>;
  }

  return (
    <span onClick={onClick} className="cursor-pointer">
      {children}
    </span>
  );
};
