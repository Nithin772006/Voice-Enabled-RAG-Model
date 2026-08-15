import React, { useState, useEffect } from 'react';

interface LoadingStateProps {
  isLoading: boolean;
}

const CYCLING_MESSAGES = [
  'Searching multilingual knowledge base...',
  'Grounding evidence and matching context...',
  'Formulating structured answer via Sarvam-105B...',
  'Synthesizing voice response using Sarvam Bulbul v3...',
  'Finalizing audio response files...'
];

export const LoadingState: React.FC<LoadingStateProps> = ({ isLoading }) => {
  const [prevIsLoading, setPrevIsLoading] = useState(false);
  const [messageIndex, setMessageIndex] = useState(0);

  if (isLoading !== prevIsLoading) {
    setPrevIsLoading(isLoading);
    setMessageIndex(0);
  }

  useEffect(() => {
    if (!isLoading) return;

    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % CYCLING_MESSAGES.length);
    }, 2000);

    return () => clearInterval(interval);
  }, [isLoading]);

  if (!isLoading) return null;

  return (
    <div className="loading-state-container">
      <div className="spinner-glow">
        <div className="double-bounce1"></div>
        <div className="double-bounce2"></div>
      </div>
      <p className="loading-text">{CYCLING_MESSAGES[messageIndex]}</p>
    </div>
  );
};
