import { Brain } from 'lucide-react';
import { LanguageToggle } from './LanguageToggle';
import { useT } from '@/lib/i18n';

export function AppHeader() {
  const t = useT();
  return (
    <header className="border-b h-14 flex items-center justify-between px-4 shrink-0">
      <div className="flex items-center gap-2 font-semibold">
        <Brain className="size-5" />
        <span>{t('appName')}</span>
        <span className="text-muted-foreground text-sm font-normal hidden sm:inline">
          · {t('appTagline')}
        </span>
      </div>
      <LanguageToggle />
    </header>
  );
}
