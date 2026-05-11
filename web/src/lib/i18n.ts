import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Lang = 'en' | 'zh';

const en = {
  appName: 'MindWise',
  appTagline: 'Multimodal mental-health AI assistant',

  newSession: 'New session',
  inputPlaceholder: 'Type your message...',
  attachAudio: 'Attach audio',
  attachVideo: 'Attach video',
  emptyState: 'Say hi to start the conversation.',
  emptyMeta: 'No conversation yet.',

  intent: 'Intent',
  risk: 'Risk',
  score: 'Score',
  emotion: 'Emotion',

  riskNormal: 'Normal',
  riskAttention: 'Needs attention',
  riskHigh: 'High risk',

  intentChat: 'Chat',
  intentConsult: 'Consult',
  intentRisk: 'Crisis',

  emotionNormal: 'Normal',
  emotionAnxiety: 'Anxiety',
  emotionLow: 'Low mood',
  emotionHighRisk: 'High risk',

  stop: 'Stop',
  crisisTitle: 'We are here with you.',
  crisisBody:
    'Reaching out takes courage. If you are struggling right now, please connect with a real person who can be present with you.',
  crisisAck: 'I understand',

  demoMode: 'Demo mode',
  demoModeTooltip:
    'Replies are scripted for public demo. To use the real backend, run uvicorn locally or set VITE_API_BASE.',
} as const;

type Key = keyof typeof en;

const zh: Record<Key, string> = {
  appName: 'MindWise',
  appTagline: '多模态心理健康 AI 助手',

  newSession: '新建会话',
  inputPlaceholder: '输入消息...',
  attachAudio: '上传语音',
  attachVideo: '上传视频',
  emptyState: '和我说说你今天的感受。',
  emptyMeta: '还没有对话。',

  intent: '意图',
  risk: '风险',
  score: '得分',
  emotion: '情绪',

  riskNormal: '正常',
  riskAttention: '需关注',
  riskHigh: '高风险',

  intentChat: '闲聊',
  intentConsult: '咨询',
  intentRisk: '危机',

  emotionNormal: '正常',
  emotionAnxiety: '焦虑',
  emotionLow: '低落',
  emotionHighRisk: '高风险',

  stop: '停止',
  crisisTitle: '我们一直在。',
  crisisBody: '你愿意说出来已经需要很多勇气。如果当下感觉撑不住，请联系一个真实的人来陪伴你。',
  crisisAck: '我会去联系',

  demoMode: '演示模式',
  demoModeTooltip:
    '当前回复是脚本化的，用于公开演示。要连真实后端，请本地运行 uvicorn 或设置 VITE_API_BASE。',
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
