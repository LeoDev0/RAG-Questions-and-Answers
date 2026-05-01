import type { ChatMessage, ApiMessage } from '@/types';

export const MAX_HISTORY_TURNS = 10;

export function buildHistory(messages: ChatMessage[]): ApiMessage[] {
  const cleaned = messages
    .filter(m => m.content.trim().length > 0)
    .map<ApiMessage>(m => ({ role: m.role, content: m.content }));
  return cleaned.slice(-MAX_HISTORY_TURNS);
}
