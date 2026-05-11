import { Badge } from '@/components/ui/badge';
import { RiskBadge } from './RiskBadge';
import { useT } from '@/lib/i18n';
import type { ChatMeta, Emotion, Intent } from '@/types/chat';

const intentLabelKey: Record<Intent, 'intentChat' | 'intentConsult' | 'intentRisk'> = {
  CHAT: 'intentChat',
  CONSULT: 'intentConsult',
  RISK: 'intentRisk',
};

const emotionLabelKey: Record<
  Emotion,
  'emotionNormal' | 'emotionAnxiety' | 'emotionLow' | 'emotionHighRisk'
> = {
  正常: 'emotionNormal',
  焦虑: 'emotionAnxiety',
  低落: 'emotionLow',
  高风险: 'emotionHighRisk',
};

export function MetaBar({ meta }: { meta: ChatMeta | null }) {
  const t = useT();
  if (!meta) {
    return (
      <div className="border-b h-12 flex items-center px-4 text-xs text-muted-foreground">
        {t('emptyMeta')}
      </div>
    );
  }
  return (
    <div className="border-b h-12 flex items-center gap-x-2 gap-y-1 px-4 text-xs flex-wrap">
      <span className="text-muted-foreground">{t('intent')}</span>
      <Badge variant="outline">{t(intentLabelKey[meta.intent])}</Badge>
      <span className="text-muted-foreground ml-2">{t('risk')}</span>
      <RiskBadge risk={meta.risk} />
      <span className="text-muted-foreground ml-2">{t('score')}</span>
      <span className="font-mono">{meta.fusedScore.toFixed(2)}</span>
      <span className="ml-auto flex items-center gap-1.5 text-muted-foreground">
        <span>{t('emotion')}</span>
        <span className="font-medium text-foreground">
          {t(emotionLabelKey[meta.emotionLabel])}
        </span>
      </span>
    </div>
  );
}
