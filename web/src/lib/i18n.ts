import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Lang = 'en' | 'zh';

const en = {
  appName: 'MindWise',
  appTagline: 'Multimodal mental-health AI assistant',
  scaffoldReady: 'Frontend scaffold is ready. Tailwind v4 + shadcn/ui wired end-to-end.',
  testButton: 'Test button',
  riskNormal: 'Normal',
  riskAttention: 'Needs attention',
  riskHigh: 'High risk',
  intentChat: 'Chat',
  intentConsult: 'Consult',
  intentRisk: 'Risk',
} as const;

type Key = keyof typeof en;

const zh: Record<Key, string> = {
  appName: 'MindWise',
  appTagline: '多模态心理健康 AI 助手',
  scaffoldReady: '前端骨架已就绪。Tailwind v4 + shadcn/ui 全链路通。',
  testButton: '测试按钮',
  riskNormal: '正常',
  riskAttention: '需关注',
  riskHigh: '高风险',
  intentChat: '闲聊',
  intentConsult: '咨询',
  intentRisk: '危机',
};

const dict = { en, zh } as const;

function detectInitialLang(): Lang {
  if (typeof navigator !== 'undefined' && navigator.language?.toLowerCase().startsWith('zh')) {
    return 'zh';
  }
  return 'en';
}

interface LangState {
  lang: Lang;
  setLang: (l: Lang) => void;
  toggle: () => void;
}

export const useLangStore = create<LangState>()(
  persist(
    (set, get) => ({
      lang: detectInitialLang(),
      setLang: (l) => set({ lang: l }),
      toggle: () => set({ lang: get().lang === 'en' ? 'zh' : 'en' }),
    }),
    { name: 'mindwise.lang' },
  ),
);

export function useT(): (k: Key) => string {
  const lang = useLangStore((s) => s.lang);
  return (k) => dict[lang][k];
}
