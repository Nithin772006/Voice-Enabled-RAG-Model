import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HHGOA-2026 Multilingual RAG Voice Assistant",
  description: "A premium voice-enabled question-answering assistant running stateful LangGraph workflows and Sarvam TTS audio generations.",
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
