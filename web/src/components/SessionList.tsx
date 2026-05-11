import { Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { useT } from '@/lib/i18n';
import type { ChatSession } from '@/types/chat';

interface Props {
  sessions: ChatSession[];
  currentId: string;
  onSelect: (id: string) => void;
  onNew: () => void;
}

export function SessionList({ sessions, currentId, onSelect, onNew }: Props) {
  const t = useT();
  return (
    <aside className="w-64 border-r flex flex-col bg-muted/30 shrink-0">
      <div className="p-3 border-b">
        <Button onClick={onNew} variant="outline" className="w-full justify-start gap-2">
          <Plus className="size-4" />
          {t('newSession')}
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <ul className="p-2 space-y-0.5">
          {sessions.map((s) => (
            <li key={s.id}>
              <button
                type="button"
                onClick={() => onSelect(s.id)}
                className={cn(
                  'w-full text-left px-3 py-2 rounded-md text-sm truncate transition-colors',
                  s.id === currentId
                    ? 'bg-accent text-accent-foreground'
                    : 'hover:bg-accent/50',
                )}
              >
                {s.title || t('newSession')}
              </button>
            </li>
          ))}
        </ul>
      </ScrollArea>
    </aside>
  );
}
