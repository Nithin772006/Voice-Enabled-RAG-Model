import React from 'react';

interface AnswerCardProps {
  answer: string;
  language: string;
  status: 'success' | 'fallback' | 'error';
}

export const AnswerCard: React.FC<AnswerCardProps> = ({
  answer,
  language,
  status
}) => {
  const isFallback = status === 'fallback';

  return (
    <div className={`answer-card ${isFallback ? 'fallback-mode' : 'success-mode'}`}>
      <div className="answer-header">
        <span className="badge-title">Answer</span>
        <span className="badge-lang">Language: {language.toUpperCase()}</span>
      </div>
      
      {isFallback && (
        <div className="fallback-banner">
          <span className="fallback-icon">⚠️</span>
          <span className="fallback-text">
            <strong>System Notice:</strong> The question could not be answered using the available knowledge base. Showing safe fallback response below.
          </span>
        </div>
      )}

      <div className="answer-body">
        <p className="answer-text">{answer}</p>
      </div>
    </div>
  );
};
