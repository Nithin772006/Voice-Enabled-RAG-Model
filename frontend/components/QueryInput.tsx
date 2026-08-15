import React, { useRef, KeyboardEvent } from 'react';

interface QueryInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  placeholder?: string;
}

export const QueryInput: React.FC<QueryInputProps> = ({
  value,
  onChange,
  onSubmit,
  isLoading,
  placeholder = 'Ask a question about the database...'
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const isQueryEmpty = !value || !value.trim();

  // Allow Ctrl+Enter or Cmd+Enter keyboard shortcut for submission
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (!isQueryEmpty && !isLoading) {
        onSubmit();
      }
    }
  };

  return (
    <div className="query-input-container">
      <label htmlFor="query-textarea" className="input-label">
        Enter Query:
      </label>
      <div className="input-textarea-wrapper">
        <textarea
          id="query-textarea"
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          placeholder={placeholder}
          rows={3}
          className="form-textarea"
        />
        <div className="input-footer">
          <span className="input-tip">Tip: Press Ctrl + Enter to search</span>
          <button
            type="button"
            onClick={onSubmit}
            disabled={isQueryEmpty || isLoading}
            className="submit-button"
          >
            {isLoading ? 'Processing...' : 'Search'}
          </button>
        </div>
      </div>
    </div>
  );
};
