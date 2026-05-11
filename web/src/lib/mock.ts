import type { ChatSession } from '@/types/chat';

const now = Date.now();
const MIN = 60_000;
const HOUR = 60 * MIN;
const DAY = 24 * HOUR;

export const MOCK_SESSIONS: ChatSession[] = [
  {
    id: 's1',
    title: '最近论文压力很大',
    createdAt: now - 30 * MIN,
    lastMeta: {
      sessionId: 's1',
      intent: 'CONSULT',
      emotionLabel: '焦虑',
      risk: '需关注',
      fusedScore: 1.3,
    },
    messages: [
      {
        id: 'm1',
        role: 'user',
        content: '最近因为论文 deadline 睡不太着，每天醒来都觉得心慌。',
        timestamp: now - 30 * MIN,
      },
      {
        id: 'm2',
        role: 'assistant',
        content:
          '听到你说这些，我心里有点心疼。论文 deadline 压在身上、又睡不好、每天醒来还心慌，这种状态持续下去，身体和心情都会被慢慢掏空。可以告诉我最近一次心慌大概是什么时候发生的吗？',
        timestamp: now - 29 * MIN,
      },
    ],
  },
  {
    id: 's2',
    title: '想聊聊和室友的关系',
    createdAt: now - DAY,
    lastMeta: {
      sessionId: 's2',
      intent: 'CHAT',
      emotionLabel: '正常',
      risk: '正常',
      fusedScore: 0.2,
    },
    messages: [
      {
        id: 'm3',
        role: 'user',
        content: 'Hi，想聊聊和室友相处的一些小摩擦。',
        timestamp: now - DAY,
      },
      {
        id: 'm4',
        role: 'assistant',
        content: '当然，欢迎你来聊。是什么样的摩擦呢？哪一件最近让你比较介意？',
        timestamp: now - DAY + 30_000,
      },
    ],
  },
  {
    id: 's3',
    title: '深夜情绪记录',
    createdAt: now - 3 * DAY,
    lastMeta: {
      sessionId: 's3',
      intent: 'RISK',
      emotionLabel: '低落',
      risk: '高风险',
      fusedScore: 2.4,
    },
    messages: [
      {
        id: 'm5',
        role: 'user',
        content: '感觉撑不下去了，想消失一段时间。',
        timestamp: now - 3 * DAY,
      },
      {
        id: 'm6',
        role: 'assistant',
        content:
          '谢谢你愿意把这些告诉我。"想消失"这四个字背后，压着很多说不出口的疲惫。我想先和你确认一下，你现在是安全的吗？身边有可以联系到的人吗？学校心理中心 24 小时热线是 xxx-xxxx-xxxx，可以现在就拨过去。',
        timestamp: now - 3 * DAY + 60_000,
      },
    ],
  },
];
