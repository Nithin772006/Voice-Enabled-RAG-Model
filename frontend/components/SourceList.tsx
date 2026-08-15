import React from 'react';
import { SourceMetadata } from '../types/api';

interface SourceListProps {
  sources: SourceMetadata[];
}

export const SourceList: React.FC<SourceListProps> = ({ sources }) => {
  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="sources-container">
      <h3 className="sources-title">Retrieved Sources ({sources.length})</h3>
      <div className="sources-grid">
        {sources.map((src, index) => {
          // Display similarity score dynamically if available
          const scoreDisplay = src.score !== undefined ? `${(src.score * 100).toFixed(1)}%` : 'N/A';
          
          return (
            <div key={src.chunk_id || index} className="source-item-card">
              <div className="source-item-header">
                <span className="source-rank">#{index + 1}</span>
                <span className="source-id">ID: {src.chunk_id}</span>
                {src.language && (
                  <span className="source-lang-badge">{src.language}</span>
                )}
              </div>
              <div className="source-score-badge">
                <span className="score-label">Similarity Relevance:</span>
                <span className="score-value">{scoreDisplay}</span>
              </div>
              {src.text && (
                <div className="source-snippet">
                  <p className="snippet-text">&quot;{src.text}&quot;</p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
