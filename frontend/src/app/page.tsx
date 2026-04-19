'use client';

import { useState, useEffect } from 'react';
import { ChatMessage, DocumentChunk, UploadResponse, ErrorResponse } from '@/types';
import { sendQuery, QueryMode } from '@/lib/api/query';

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [thinkingDots, setThinkingDots] = useState('');
  const [queryMode, setQueryMode] = useState<QueryMode>('stream');

  // Animate thinking dots when loading
  useEffect(() => {
    if (!isLoading) {
      setThinkingDots('');
      return;
    }

    const interval = setInterval(() => {
      setThinkingDots(prev => {
        if (prev === '') return '.';
        if (prev === '.') return '..';
        if (prev === '..') return '...';
        return '';
      });
    }, 500);

    return () => clearInterval(interval);
  }, [isLoading]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadStatus('Uploading...');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';
      const response = await fetch(`${backendUrl}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result: UploadResponse = await response.json();
        setUploadStatus(`✅ ${file.name} uploaded successfully! (${result.document.chunksCount} chunks)`);
      } else {
        const error: ErrorResponse = await response.json();
        setUploadStatus(`❌ Failed to upload: ${error.error}`);
      }
    } catch (error) {
      setUploadStatus(`❌ Upload error: ${error}`);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const question = input;
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
      timestamp: new Date(),
    };

    const assistantId = (Date.now() + 1).toString();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage, assistantMessage]);
    setInput('');
    setIsLoading(true);

    const appendToAssistant = (append: string) => {
      setMessages(prev =>
        prev.map(m => (m.id === assistantId ? { ...m, content: m.content + append } : m))
      );
    };

    const setAssistantContent = (content: string) => {
      setMessages(prev => prev.map(m => (m.id === assistantId ? { ...m, content } : m)));
    };

    const setAssistantSources = (sources: DocumentChunk[]) => {
      setMessages(prev =>
        prev.map(m => (m.id === assistantId ? { ...m, sources } : m))
      );
    };

    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';
    await sendQuery(queryMode, question, backendUrl, {
      onSources: setAssistantSources,
      onToken: (chunk) => {
        setIsLoading(false);
        appendToAssistant(chunk);
      },
      onError: (message) => {
        setAssistantContent(`Sorry, there was an error: ${message}`);
      },
      onDone: () => {
        setIsLoading(false);
      },
    });
  };

  return (
    <div className="container mx-auto max-w-4xl p-4">
      <h1 className="text-3xl font-bold mb-8 text-center">Document Q&A Bot</h1>
      
      {/* File Upload */}
      <div className="mb-8 p-6 border-2 border-dashed border-gray-300 rounded-lg">
        <h2 className="text-xl font-semibold mb-4">Upload Document</h2>
        <input
          type="file"
          accept=".pdf,.txt"
          onChange={handleFileUpload}
          className="mb-2"
        />
        {uploadStatus && (
          <p className="text-sm mt-2">{uploadStatus}</p>
        )}
      </div>

      {/* Response mode toggle */}
      <div className="flex items-center gap-4 mb-2 text-sm">
        <span className="font-medium">Response mode:</span>
        <label className="flex items-center gap-1">
          <input
            type="radio"
            value="stream"
            checked={queryMode === 'stream'}
            onChange={() => setQueryMode('stream')}
            disabled={isLoading}
          />
          Streaming
        </label>
        <label className="flex items-center gap-1">
          <input
            type="radio"
            value="single"
            checked={queryMode === 'single'}
            onChange={() => setQueryMode('single')}
            disabled={isLoading}
          />
          Single response
        </label>
      </div>

      {/* Chat Interface */}
      <div className="border rounded-lg h-96 mb-4 overflow-y-auto p-4 bg-gray-50">
        {messages.length === 0 ? (
          <p className="text-gray-500 text-center">Upload a document and start asking questions!</p>
        ) : (
          messages
            .filter(message => message.role === 'user' || message.content.length > 0)
            .map((message) => (
              <div
                key={message.id}
                className={`mb-4 ${message.role === 'user' ? 'text-right' : 'text-left'
                  }`}
              >
                <div
                  className={`inline-block p-3 rounded-lg max-w-xs lg:max-w-md ${message.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-black'
                    }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
              </div>
            ))
        )}
        {isLoading && (
          <div className="text-left">
            <div className="inline-block p-3 rounded-lg bg-gray-200 text-black">
              <p>
                Thinking
                <span className="inline-block w-6 text-left">
                  {thinkingDots}
                </span>
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about your document..."
          className="flex-1 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400"
        >
          Send
        </button>
      </form>
    </div>
  );
}
