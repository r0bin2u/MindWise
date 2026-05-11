import { Brain } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { LanguageToggle } from './LanguageToggle';
import { isMockMode } from '@/lib/api';
import { useT } from '@/lib/i18n';

export function AppHeader() {
  const t = useT();
  const mock = isMockMode();
  return (
    <header className="border-b h-14 flex items-center justify-between px-4 shrink-0">
      <div className="flex items-center gap-2 font-semibold">
        <Brain className="size-5" />
        <span>{t('appName')}</span>
        <span className="text-muted-foreground text-sm font-normal hidden sm:inline">
          · {t('appTagline')}
        </span>
        {mock && (
          <Badge
            variant="secondary"
            className="ml-2 text-[10px] font-normal uppercase tracking-wide"
            title={t('demoModeTooltip')}
          >
            {t('demoMode')}
          </Badge>
        )}
      </div>
      <LanguageToggle />
    </header>
  );
}
