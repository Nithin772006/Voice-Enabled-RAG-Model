import React, { useState, useRef } from 'react';
import { getAudioSourceUrl } from '../lib/api';

interface AudioPlayerProps {
  audioUrl: string | null;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({ audioUrl }) => {
  const [error, setError] = useState<string | null>(null);
  const [prevUrl, setPrevUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  if (audioUrl !== prevUrl) {
    setPrevUrl(audioUrl);
    setError(null);
  }

  if (!audioUrl) {
    return null;
  }

  const absoluteUrl = getAudioSourceUrl(audioUrl);

  const handleAudioError = () => {
    setError('Synthesized voice speech could not be loaded or retrieved.');
  };

  return (
    <div className="audio-player-wrapper">
      <div className="audio-player-header">
        <span className="audio-icon">🔊</span>
        <span className="audio-title">Voice Output</span>
      </div>
      
      {error ? (
        <div className="audio-error-banner">
          <span>{error}</span>
        </div>
      ) : (
        <div className="audio-element-container">
          <audio
            ref={audioRef}
            src={absoluteUrl}
            controls
            onError={handleAudioError}
            className="native-audio-player"
          />
        </div>
      )}
    </div>
  );
};
