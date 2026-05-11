import { useState } from 'react';
import { AppHeader } from '@/components/AppHeader';
import { ChatInput } from '@/components/ChatInput';
import { ChatMessage } from '@/components/ChatMessage';
import { MetaBar } from '@/components/MetaBar';
import { SessionList } from '@/components/SessionList';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useChat } from '@/hooks/useChat';
import { useT } from '@/lib/i18n';
import { MOCK_SESSIONS } from '@/lib/mock';
import type { ChatMessage as MessageT, ChatSession } from '@/types/chat';

const DEMO_USER_ID = 'demo-user';

export function ChatPage() {
  const t = useT();
  const [sessions, setSessions] = useState<ChatSession[]>(MOCK_SESSIONS);
  const [currentId, setCurrentId] = useState(sessions[0]?.id ?? '');
  const current = sessions.find((s) => s.id === currentId);

  const { send, streaming } = useChat();

  function patchCurrent(updater: (s: ChatSession) => ChatSession) {
    setSessions((prev) => prev.map((s) => (s.id === currentId ? updater(s) : s)));
  }

  function handleSend(text: string) {
    if (!current || streaming) return;

    const userMsg: MessageT = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };
    const asstId = crypto.randomUUID();
    const asstMsg: MessageT = {
      id: asstId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    };

    patchCurrent((s) => ({ ...s, messages: [...s.messages, userMsg, asstMsg] }));

    send(
      { userId: DEMO_USER_ID, message: text, sessionId: current.id },
      {
        onMeta: (meta) => patchCurrent((s) => ({ ...s, lastMeta: meta })),
        onToken: (tok) =>
          patchCurrent((s) => ({
            ...s,
            messages: s.messages.map((m) =>
              m.id === asstId ? { ...m, content: m.content + tok } : m,
            ),
          })),
        onError: (err) => {
          const reason = err instanceof Error ? err.message : String(err);
          patchCurrent((s) => ({
            ...s,
            messages: s.messages.map((m) =>
              m.id === asstId
                ? { ...m, content: m.content || `⚠️ ${reason}` }
                : m,
            ),
          }));
        },
      },
    );
  }

  function handleNew() {
    const id = crypto.randomUUID();
    const fresh: ChatSession = {
      id,
      title: t('newSession'),
      messages: [],
      createdAt: Date.now(),
    };
    setSessions((prev) => [fresh, ...prev]);
    setCurrentId(id);
  }

  return (
    <div className="h-screen flex flex-col">
      <AppHeader />
      <div className="flex-1 flex overflow-hidden">
        <SessionList
          sessions={sessions}
          currentId={currentId}
          onSelect={setCurrentId}
          onNew={handleNew}
        />
        <main className="flex-1 flex flex-col min-w-0">
          <MetaBar meta={current?.lastMeta ?? null} />
          <ScrollArea className="flex-1">
            <div className="max-w-3xl mx-auto p-6 space-y-4">
              {!current || current.messages.length === 0 ? (
                <div className="text-center text-muted-foreground py-12 text-sm">
                  {t('emptyState')}
                </div>
              ) : (
                current.messages.map((m) => <ChatMessage key={m.id} message={m} />)
              )}
            </div>
          </ScrollArea>
          <ChatInput onSend={handleSend} disabled={streaming} />
        </main>
      </div>
    </div>
  );
}
