import { useState, type KeyboardEvent } from 'react';
import { Paperclip, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { useT } from '@/lib/i18n';

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: Props) {
  const t = useT();
  const [value, setValue] = useState('');

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className="border-t p-3 bg-background">
      <div className="flex gap-2 items-end max-w-3xl mx-auto">
        <Button variant="ghost" size="icon" disabled title={t('attachAudio')}>
          <Paperclip className="size-4" />
        </Button>
        <Textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={t('inputPlaceholder')}
          rows={1}
          className="min-h-10 resize-none"
          disabled={disabled}
        />
        <Button onClick={submit} disabled={disabled || !value.trim()} size="icon">
          <Send className="size-4" />
        </Button>
      </div>
    </div>
  );
}
