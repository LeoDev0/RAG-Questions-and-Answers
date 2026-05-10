import type { ChatMessage, ApiMessage } from '@/types';

// Transport cap: avoid unbounded request payloads on very long sessions.
// Independent of the backend's history cap — the backend will trim further.
const MAX_TRANSPORTED_TURNS = 50;

export function buildHistory(messages: ChatMessage[]): ApiMessage[] {
  return messages
      .filter(m => m.content.trim().length > 0)
      .map<ApiMessage>(m => ({ role: m.role, content: m.content }))
      .slice(-MAX_TRANSPORTED_TURNS);
}
