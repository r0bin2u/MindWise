import { LifeBuoy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useLangStore, useT, type Lang } from '@/lib/i18n';

interface Hotline {
  label: string;
  value: string;
}

const HOTLINES: Record<Lang, Hotline[]> = {
  en: [
    { label: 'Campus counseling center', value: '24h' },
    { label: '988 Suicide & Crisis Lifeline (US)', value: '988' },
    { label: 'Find a Helpline (international)', value: 'findahelpline.com' },
  ],
  zh: [
    { label: '学校心理咨询中心', value: '24 小时' },
    { label: '北京心理危机研究与干预中心', value: '010-82951332' },
    { label: '全国心理援助热线', value: '400-161-9995' },
  ],
};

interface Props {
  open: boolean;
  onClose: () => void;
}

export function CrisisDialog({ open, onClose }: Props) {
  const t = useT();
  const lang = useLangStore((s) => s.lang);
  const lines = HOTLINES[lang];

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <LifeBuoy className="size-5 text-destructive" />
            {t('crisisTitle')}
          </DialogTitle>
          <DialogDescription>{t('crisisBody')}</DialogDescription>
        </DialogHeader>
        <ul className="space-y-2 text-sm">
          {lines.map((h) => (
            <li
              key={h.label}
              className="flex items-baseline justify-between gap-3 rounded-md bg-muted px-3 py-2"
            >
              <span className="text-muted-foreground">{h.label}</span>
              <span className="font-mono font-medium">{h.value}</span>
            </li>
          ))}
        </ul>
        <DialogFooter>
          <Button onClick={onClose} className="w-full">
            {t('crisisAck')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
