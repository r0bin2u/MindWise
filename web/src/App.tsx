import { Brain } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LanguageToggle } from '@/components/LanguageToggle';
import { useT } from '@/lib/i18n';

export default function App() {
  const t = useT();
  return (
    <div className="min-h-screen bg-background flex flex-col">
      <header className="border-b">
        <div className="mx-auto w-full max-w-5xl flex items-center justify-between px-4 h-14">
          <div className="flex items-center gap-2 font-semibold">
            <Brain className="size-5" />
            <span>{t('appName')}</span>
            <span className="text-muted-foreground text-sm font-normal hidden sm:inline">
              · {t('appTagline')}
            </span>
          </div>
          <LanguageToggle />
        </div>
      </header>

      <main className="flex-1 flex items-center justify-center p-6">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="size-5" />
              {t('appName')}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground text-sm">{t('scaffoldReady')}</p>
            <div className="flex flex-wrap items-center gap-2">
              <Badge>{t('intentChat')}</Badge>
              <Badge variant="secondary">{t('riskAttention')}</Badge>
              <Badge variant="destructive">{t('riskHigh')}</Badge>
            </div>
            <Button className="w-full">{t('testButton')}</Button>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
