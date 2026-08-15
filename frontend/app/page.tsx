'use client';

import React, { useState, useEffect, useRef } from 'react';
import { QueryInput } from '../components/QueryInput';
import { LanguageSelector } from '../components/LanguageSelector';
import { AnswerCard } from '../components/AnswerCard';
import { SourceList } from '../components/SourceList';
import { AudioPlayer } from '../components/AudioPlayer';
import { LoadingState } from '../components/LoadingState';
import { getHealth, getReadiness, queryRAG } from '../lib/api';
import { RAGResponse } from '../types/api';

const EXAMPLE_QUESTIONS = [
  { text: 'What is machine learning?', lang: 'english' },
  { text: 'ஒரு நிறுவனம் என்பது என்ன?', lang: 'tamil' },
  { text: 'మెషిన్ లెర్నింగ్ అంటే ఏమిటి?', lang: 'telugu' },
  { text: 'मशीन लर्निंग क्या है?', lang: 'hindi' }
];

export default function Home() {
  // Query and settings state
  const [query, setQuery] = useState('');
  const [language, setLanguage] = useState('tamil'); // Default target
  
  // Pipeline status and result states
  const [status, setStatus] = useState<'initial' | 'loading' | 'success' | 'fallback' | 'error'>('initial');
  const [result, setResult] = useState<RAGResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Backend status monitoring
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [dbReady, setDbReady] = useState<boolean>(false);

  // References for request cancellation
  const abortControllerRef = useRef<AbortController | null>(null);

  // 1. Monitor backend health on load and poll periodically
  useEffect(() => {
    async function verifyStatus() {
      try {
        await getHealth();
        setBackendOnline(true);
        
        // Check database collection readiness
        try {
          const readyRes = await getReadiness();
          setDbReady(readyRes.status === 'ready' && readyRes.qdrant === 'ok');
        } catch {
          setDbReady(false);
        }
      } catch (err) {
        setBackendOnline(false);
        setDbReady(false);
        console.error('Backend connection check failed:', err);
      }
    }

    verifyStatus();
    const interval = setInterval(verifyStatus, 15000); // Check every 15s

    return () => {
      clearInterval(interval);
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // 2. Submit handler with AbortController
  const handleSubmit = async () => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    // Cancel any active query running in the background to prevent overlaps
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Spawn a new AbortController
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setStatus('loading');
    setErrorMsg(null);

    try {
      const response = await queryRAG(trimmedQuery, language, controller.signal);
      
      setResult(response);
      if (response.status === 'fallback') {
        setStatus('fallback');
      } else {
        setStatus('success');
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;

      console.error('RAG Query submission failed:', err);
      setStatus('error');
      const msg = err instanceof Error ? err.message : 'Unable to connect to the RAG backend. Please verify that the server is online.';
      setErrorMsg(msg);
    }
  };

  // 3. Example click loader
  const handleSelectExample = (text: string, lang: string) => {
    setQuery(text);
    setLanguage(lang);
  };

  return (
    <main className="app-container">
      {/* 1. Header and description */}
      <header className="app-header">
        <h1 className="app-title">HHGOA-2026 Voice RAG</h1>
        <p className="app-description">
          Multilingual Indian Language Question Answering & Speech Synthesizer
        </p>

        {/* Backend health status indicator */}
        <div className="health-indicator">
          <span className={`status-dot ${backendOnline === true ? 'online' : 'offline'}`}></span>
          <span>
            {backendOnline === null && 'Checking backend status...'}
            {backendOnline === true && dbReady && 'Backend Online (Database Ready)'}
            {backendOnline === true && !dbReady && 'Backend Online (Database Locking/Unavailable)'}
            {backendOnline === false && 'RAG backend is unavailable. Please start the FastAPI server.'}
          </span>
        </div>
      </header>

      {/* 2. Main query interface glass panel */}
      <section className="glass-panel">
        <LanguageSelector
          selectedLanguage={language}
          onChange={setLanguage}
          disabled={status === 'loading'}
        />

        <QueryInput
          value={query}
          onChange={setQuery}
          onSubmit={handleSubmit}
          isLoading={status === 'loading'}
        />

        {/* Suggested examples (Empty state helper) */}
        <div className="examples-row">
          <span className="examples-title">Try asking:</span>
          <div className="examples-chips">
            {EXAMPLE_QUESTIONS.map((ex, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleSelectExample(ex.text, ex.lang)}
                disabled={status === 'loading'}
                className="example-chip"
              >
                {ex.text}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* 3. Loading animation log messages */}
      <LoadingState isLoading={status === 'loading'} />

      {/* 4. Error state block */}
      {status === 'error' && (
        <div className="error-banner">
          <span className="error-icon">❌</span>
          <div className="error-body">
            <h4 className="error-title">Search Failed</h4>
            <p className="error-text">{errorMsg}</p>
          </div>
        </div>
      )}

      {/* 5. Response sections (Answers, citations, and Audio playback) */}
      {(status === 'success' || status === 'fallback') && result && (
        <section className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <AnswerCard
            answer={result.answer}
            language={result.language}
            status={result.status}
          />

          <AudioPlayer audioUrl={result.audio_url} />

          <SourceList sources={result.sources} />
        </section>
      )}
    </main>
  );
}
