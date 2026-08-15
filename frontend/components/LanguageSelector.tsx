import React from 'react';

interface LanguageSelectorProps {
  selectedLanguage: string;
  onChange: (language: string) => void;
  disabled?: boolean;
}

const SUPPORTED_LANGUAGES = [
  { code: 'tamil', label: 'தமிழ் (Tamil)' },
  { code: 'english', label: 'English' },
  { code: 'telugu', label: 'తెలుగు (Telugu)' },
  { code: 'hindi', label: 'हिन्दी (Hindi)' }
];

export const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  selectedLanguage,
  onChange,
  disabled = false
}) => {
  return (
    <div className="language-selector-container">
      <label htmlFor="language-select" className="input-label">
        Target Language:
      </label>
      <select
        id="language-select"
        value={selectedLanguage}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="form-select"
      >
        {SUPPORTED_LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.label}
          </option>
        ))}
      </select>
    </div>
  );
};
