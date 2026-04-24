import { ErrorResponse, RAGResponse } from '@/types';
import { QueryCallbacks } from './query';

export async function sendQuerySingle(
  question: string,
  backendUrl: string,
  callbacks: QueryCallbacks,
): Promise<void> {
  try {
    const response = await fetch(`${backendUrl}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      const error = (await response.json()) as ErrorResponse;
      callbacks.onError(error.error, error.code);
      return;
    }

    const result = (await response.json()) as RAGResponse;
    callbacks.onSources?.(result.sources, result.confidence);
    callbacks.onToken(result.answer);
  } catch {
    callbacks.onError('There was an error processing your question.');
  } finally {
    callbacks.onDone();
  }
}
