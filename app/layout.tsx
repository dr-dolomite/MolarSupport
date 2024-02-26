import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Molar Support",
  description: "An intelligent web application for evaluating M3-MC risk and relation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
