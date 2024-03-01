import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground shadow hover:bg-primary/90",
        destructive:
          "bg-destructive text-destructive-foreground shadow-sm hover:bg-destructive/90",
        outline:
          "border border-input bg-background shadow-sm hover:bg-accent hover:text-accent-foreground",
        secondary:
          "bg-secondary text-secondary-foreground shadow-sm hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
        submitButton: "bg-[#6D58C6] hover:bg-[#6D58C6]/90 text-white lg:text-2xl text-lg font-bold shadow-sm drop-shadow-xl",
        errorButton: "bg-[#595080] hover:bg-[#595080]/90 text-white text-center lg:text-2xl text-lg font-bold cursor-pointer",
        successButton: "border-3 border-[#039855] bg-[#039855] hover:bg-[#039855]/80 text-white text-center lg:text-2xl text-lg font-bold cursor-pointer",
        successBack: "border-3 border-[#039855] hover:border-[#039855]/80 outline text-[#039855] hover:text-[#039855]/80 text-center lg:text-2xl text-lg font-bold cursor-pointer",
        purpleButton: "border-1 border-[#6D58C6] bg-[#A99AEB40] hover:border-[#6D58C6]/80 outline text-[#6D58C6] hover:text-[#6D58C6]/80 text-center text-lg font-bold cursor-pointer",
        solidPurple: "bg-[#b66fdc] hover:bg-[#b66fdc]/90 text-white text-center lg:text-xl text-lg font-bold cursor-pointer",
        destructiveButton: "bg-[#FF4D4F] hover:bg-[#FF4D4F]/90 text-white text-center lg:text-2xl text-lg font-bold cursor-pointer",
      },
      size: {
        default: "h-9 px-4 py-2",
        ghost: "p-2 rounded-full",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 rounded-md px-8",
        submitButton: "rounded-[16px] lg:py-5 px-3 py-3",
        errorButton: "rounded-[16px] lg:px-6 lg:py-4 px-3 py-3 w-full",
        successButton: "rounded-[16px] lg:px-6 lg:py-4 px-3 py-3 w-full",
        purpleButton: "rounded-[16px] px-4 py-2",
        getStarted: "rounded-[16px] px-8 py-4",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
