import { Badge } from '@/components/ui/badge';
import { useT } from '@/lib/i18n';
import type { Risk } from '@/types/chat';

type Variant = 'outline' | 'secondary' | 'destructive';

const variantByRisk: Record<Risk, Variant> = {
  正常: 'outline',
  需关注: 'secondary',
  高风险: 'destructive',
};

const labelKeyByRisk = {
  正常: 'riskNormal',
  需关注: 'riskAttention',
  高风险: 'riskHigh',
} as const;

export function RiskBadge({ risk }: { risk: Risk }) {
  const t = useT();
  return <Badge variant={variantByRisk[risk]}>{t(labelKeyByRisk[risk])}</Badge>;
}
