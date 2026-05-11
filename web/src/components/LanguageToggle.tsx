import { Button } from '@/components/ui/button';
import { useLangStore } from '@/lib/i18n';
import { Languages } from 'lucide-react';

export function LanguageToggle() {
  const lang = useLangStore((s) => s.lang);
  const toggle = useLangStore((s) => s.toggle);
  return (
    <Button variant="ghost" size="sm" onClick={toggle} className="gap-1.5">
      <Languages className="size-4" />
      <span className="text-xs font-medium">{lang === 'en' ? '中文' : 'EN'}</span>
    </Button>
  );
}
