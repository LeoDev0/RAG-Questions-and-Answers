import { DocumentChunk } from '@/types';
import { sendQueryStream } from './queryStream';
import { sendQuerySingle } from './queryOnce';

export type QueryMode = 'stream' | 'single';

export interface QueryCallbacks {
  onSources?: (sources: DocumentChunk[], confidence: number) => void;
  onToken: (chunk: string) => void;
  onError: (message: string, code?: string) => void;
  onDone: () => void;
}

export async function sendQuery(
  mode: QueryMode,
  question: string,
  backendUrl: string,
  callbacks: QueryCallbacks,
): Promise<void> {
  if (mode === 'stream') {
    return sendQueryStream(question, backendUrl, callbacks);
  }
  return sendQuerySingle(question, backendUrl, callbacks);
}
