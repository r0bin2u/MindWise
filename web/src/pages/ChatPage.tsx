import { useEffect, useRef, useState } from 'react';
import { AppHeader } from '@/components/AppHeader';
import { ChatInput } from '@/components/ChatInput';
import { ChatMessage } from '@/components/ChatMessage';
import { CrisisDialog } from '@/components/CrisisDialog';
import { MetaBar } from '@/components/MetaBar';
import { SessionList } from '@/components/SessionList';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useChat } from '@/hooks/useChat';
import { useT } from '@/lib/i18n';
import { useChatStore, useCurrentSession, useSessions } from '@/store/chat';
import type { ChatMessage as MessageT } from '@/types/chat';

const DEMO_USER_ID = 'demo-user';

export function ChatPage() {
  const t = useT();
  const sessions = useSessions();
  const current = useCurrentSession();
  const newSession = useChatStore((s) => s.newSession);
  const switchSession = useChatStore((s) => s.switchSession);
  const addMessages = useChatStore((s) => s.addMessages);
  const appendToMessage = useChatStore((s) => s.appendToMessage);
  const setMessageContent = useChatStore((s) => s.setMessageContent);
  const setLastMeta = useChatStore((s) => s.setLastMeta);
  const markCrisisSeen = useChatStore((s) => s.markCrisisSeen);

  const { send, cancel, streaming } = useChat();
  const [crisisOpen, setCrisisOpen] = useState(false);

  const messages = current?.messages ?? [];
  const lastIdx = messages.length - 1;
  const lastMsg = lastIdx >= 0 ? messages[lastIdx] : null;

  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [lastMsg?.id, lastMsg?.content]);

  function handleSend(text: string) {
    if (!current || streaming) return;

    const userMsg: MessageT = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };
    const asstMsg: MessageT = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    };
    const sessionId = current.id;
    addMessages(sessionId, [userMsg, asstMsg]);

    send(
      { userId: DEMO_USER_ID, message: text, sessionId },
      {
        onMeta: (meta) => {
          setLastMeta(sessionId, meta);
          const isCrisis = meta.intent === 'RISK' || meta.risk === '高风险';
          if (isCrisis && !useChatStore.getState().seenCrisis[sessionId]) {
            markCrisisSeen(sessionId);
            setCrisisOpen(true);
          }
        },
        onToken: (tok) => appendToMessage(sessionId, asstMsg.id, tok),
        onError: (err) => {
          const reason = err instanceof Error ? err.message : String(err);
          const peek = useChatStore
            .getState()
            .sessions[sessionId]?.messages.find((m) => m.id === asstMsg.id);
          if (peek && !peek.content) {
            setMessageContent(sessionId, asstMsg.id, `⚠️ ${reason}`);
          }
        },
      },
    );
  }

  return (
    <div className="h-screen flex flex-col">
      <AppHeader />
      <div className="flex-1 flex overflow-hidden">
        <SessionList
          sessions={sessions}
          currentId={current?.id ?? ''}
          onSelect={switchSession}
          onNew={newSession}
        />
        <main className="flex-1 flex flex-col min-w-0">
          <MetaBar meta={current?.lastMeta ?? null} />
          <ScrollArea className="flex-1">
            <div className="max-w-3xl mx-auto p-6 space-y-4">
              {messages.length === 0 ? (
                <div className="text-center text-muted-foreground py-12 text-sm">
                  {t('emptyState')}
                </div>
              ) : (
                messages.map((m, i) => (
                  <ChatMessage
                    key={m.id}
                    message={m}
                    streaming={streaming && i === lastIdx && m.role === 'assistant'}
                  />
                ))
              )}
              <div ref={endRef} />
            </div>
          </ScrollArea>
          <ChatInput
            onSend={handleSend}
            onCancel={cancel}
            disabled={streaming}
            streaming={streaming}
          />
        </main>
      </div>
      <CrisisDialog open={crisisOpen} onClose={() => setCrisisOpen(false)} />
    </div>
  );
}
