import { ErrorResponse, StreamEvent } from '@/types';
import { QueryCallbacks } from './query';

export async function sendQueryStream(
  question: string,
  backendUrl: string,
  callbacks: QueryCallbacks,
): Promise<void> {
  try {
    const response = await fetch(`${backendUrl}/api/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      const error = (await response.json()) as ErrorResponse;
      callbacks.onError(error.error, error.code);
      return;
    }

    if (!response.body) {
      callbacks.onError('Streaming response was empty.');
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
            callbacks.onSources?.(event.sources, event.confidence);
            break;
          case 'token':
            callbacks.onToken(event.content);
            break;
          case 'error':
            callbacks.onError(event.error, event.code);
            done = true;
            break;
          case 'done':
            done = true;
            break;
        }
      }
    }
  } catch {
    callbacks.onError('There was an error processing your question.');
  } finally {
    callbacks.onDone();
  }
}
