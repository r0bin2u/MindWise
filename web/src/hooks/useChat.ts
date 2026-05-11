import { useCallback, useRef, useState } from 'react';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import {
  API_BASE,
  isMockMode,
  parseMeta,
  toWireChatPayload,
  type ChatPayload,
} from '@/lib/api';
import { streamMockChat } from '@/lib/mockChat';
import type { ChatMeta } from '@/types/chat';

export interface ChatCallbacks {
  onMeta?: (meta: ChatMeta) => void;
  onToken?: (text: string) => void;
  onDone?: () => void;
  onError?: (err: unknown) => void;
}

class FatalSSEError extends Error {}

async function streamRealChat(
  payload: ChatPayload,
  cb: ChatCallbacks,
  signal: AbortSignal,
): Promise<void> {
  await fetchEventSource(`${API_BASE}/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(toWireChatPayload(payload)),
    signal,
    openWhenHidden: true,
    async onopen(res) {
      const ct = res.headers.get('content-type') ?? '';
      if (res.ok && ct.includes('text/event-stream')) return;
      throw new FatalSSEError(`Bad response: ${res.status} ${ct}`);
    },
    onmessage(ev) {
      if (ev.event === 'meta') {
        cb.onMeta?.(parseMeta(ev.data));
      } else if (ev.event === 'done') {
        cb.onDone?.();
      } else {
        cb.onToken?.(ev.data);
      }
    },
    onerror(err) {
      throw err;
    },
  });
}

export function useChat() {
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(async (payload: ChatPayload, cb: ChatCallbacks = {}) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setStreaming(true);

    const transport = isMockMode() ? streamMockChat : streamRealChat;

    try {
      await transport(payload, {
        ...cb,
        onDone: () => {
          cb.onDone?.();
          ctrl.abort();
        },
      }, ctrl.signal);
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        cb.onError?.(err);
      }
    } finally {
      setStreaming(false);
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setStreaming(false);
  }, []);

  return { send, cancel, streaming };
}
