'use client';

import { useState, useEffect } from 'react';
import { ChatMessage, StreamEvent, UploadResponse, ErrorResponse } from '@/types';

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [thinkingDots, setThinkingDots] = useState('');

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

    const setAssistantSources = (sources: StreamEvent & { type: 'sources' }) => {
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantId ? { ...m, sources: sources.sources } : m
        )
      );
    };

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';
      const response = await fetch(`${backendUrl}/api/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        const error: ErrorResponse = await response.json();
        setAssistantContent(`Sorry, there was an error: ${error.error}`);
        return;
      }

      if (!response.body) {
        setAssistantContent('Sorry, the streaming response was empty.');
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let done = false;

      while (!done) {
        const { value, done: streamDone } = await reader.read();
        if (streamDone) break;

        buffer += decoder.decode(value, { stream: true });

        let delimiterIndex = buffer.indexOf('\n\n');
        while (delimiterIndex !== -1) {
          const frame = buffer.slice(0, delimiterIndex);
          buffer = buffer.slice(delimiterIndex + 2);
          delimiterIndex = buffer.indexOf('\n\n');

          const dataLine = frame
            .split('\n')
            .find(line => line.startsWith('data:'));
          if (!dataLine) continue;

          const payload = dataLine.slice('data:'.length).trim();
          if (!payload) continue;

          let event: StreamEvent;
          try {
            event = JSON.parse(payload) as StreamEvent;
          } catch {
            continue;
          }

          switch (event.type) {
            case 'sources':
              setAssistantSources(event);
              break;
            case 'token':
              setIsLoading(false);
              appendToAssistant(event.content);
              break;
            case 'error':
              setAssistantContent(`Sorry, there was an error: ${event.error}`);
              done = true;
              break;
            case 'done':
              done = true;
              break;
          }
        }
      }
    } catch {
      setAssistantContent('Sorry, there was an error processing your question.');
    } finally {
      setIsLoading(false);
    }
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
