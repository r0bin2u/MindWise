import { Brain, User } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ChatMessage as MessageT } from '@/types/chat';

interface Props {
  message: MessageT;
  streaming?: boolean;
}

export function ChatMessage({ message, streaming = false }: Props) {
  const isUser = message.role === 'user';
  const showDots = streaming && !message.content;
  const showCursor = streaming && message.content;

  return (
    <div className={cn('flex gap-3', isUser && 'flex-row-reverse')}>
      <div className="size-8 rounded-full bg-muted flex items-center justify-center shrink-0">
        {isUser ? <User className="size-4" /> : <Brain className="size-4" />}
      </div>
      <div
        className={cn(
          'max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap',
          isUser
            ? 'bg-primary text-primary-foreground rounded-br-sm'
            : 'bg-muted text-foreground rounded-bl-sm',
        )}
      >
        {showDots ? (
          <ThinkingDots />
        ) : (
          <>
            {message.content || <span className="italic opacity-60">...</span>}
            {showCursor && <Cursor />}
          </>
        )}
      </div>
    </div>
  );
}

function ThinkingDots() {
  return (
    <span className="inline-flex items-center gap-1 py-1.5">
      <Dot delay={0} />
      <Dot delay={150} />
      <Dot delay={300} />
    </span>
  );
}

function Dot({ delay }: { delay: number }) {
  return (
    <span
      className="inline-block size-1.5 rounded-full bg-current opacity-60 animate-bounce"
      style={{ animationDelay: `${delay}ms` }}
    />
  );
}

function Cursor() {
  return (
    <span className="inline-block w-[2px] h-4 bg-current align-middle ml-0.5 animate-pulse" />
  );
}
