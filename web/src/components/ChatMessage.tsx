import { Brain, User } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ChatMessage as MessageT } from '@/types/chat';

export function ChatMessage({ message }: { message: MessageT }) {
  const isUser = message.role === 'user';
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
        {message.content || <span className="italic opacity-60">...</span>}
      </div>
    </div>
  );
}
