import type { ChatPayload } from './api';
import type { ChatCallbacks } from '@/hooks/useChat';
import type { ChatMeta, Emotion, Intent, Risk } from '@/types/chat';

const TOKEN_DELAY_MS = 30;
const META_DELAY_MS = 150;
const FIRST_TOKEN_DELAY_MS = 400;

type Category = 'RISK' | 'CONSULT' | 'CHAT';

interface ReplyTemplate {
  intent: Intent;
  emotionLabel: Emotion;
  risk: Risk;
  fusedScore: number;
  reply: string;
}

const REPLIES: Record<Category, ReplyTemplate[]> = {
  RISK: [
    {
      intent: 'RISK',
      emotionLabel: '高风险',
      risk: '正常',
      fusedScore: 0.4,
      reply:
        '谢谢你愿意把这些告诉我。"想消失"这四个字背后，压着很多说不出口的疲惫。我想先和你确认一下，你现在是安全的吗？身边有可以联系到的人吗？学校心理中心 24 小时热线是 xxx-xxxx-xxxx，可以现在就拨过去。',
    },
    {
      intent: 'RISK',
      emotionLabel: '高风险',
      risk: '正常',
      fusedScore: 0.4,
      reply:
        '我听到了，撑到现在已经不容易了。请先做一件事：把这个号码拨出去——北京心理危机研究与干预中心 010-82951332，他们 24 小时有人接听。你不需要解释那么多，只要让对方陪你说会儿话。',
    },
  ],
  CONSULT: [
    {
      intent: 'CONSULT',
      emotionLabel: '焦虑',
      risk: '需关注',
      fusedScore: 1.3,
      reply:
        '听起来你最近背的东西很重。睡不好、心慌、白天又没法停下来——这种状态身体迟早会发出信号。能告诉我，最近一次让你觉得"撑不住"的时刻，是在做什么吗？',
    },
    {
      intent: 'CONSULT',
      emotionLabel: '焦虑',
      risk: '需关注',
      fusedScore: 1.3,
      reply:
        '压力这件事不是非要"扛过去"才算赢。我们可以一起拆一拆——是哪一块在消耗你最多？是 deadline 本身、是对结果的担心、还是没有人能商量？',
    },
    {
      intent: 'CONSULT',
      emotionLabel: '低落',
      risk: '需关注',
      fusedScore: 1.5,
      reply:
        '谢谢你说出来。这种低落感像在水底走路，每一步都需要更多力气。你愿意告诉我，是从什么时候开始有这种感觉的吗？',
    },
  ],
  CHAT: [
    {
      intent: 'CHAT',
      emotionLabel: '正常',
      risk: '正常',
      fusedScore: 0.2,
      reply: '挺好的呀，今天怎么样？有什么想和我聊的吗？',
    },
    {
      intent: 'CHAT',
      emotionLabel: '正常',
      risk: '正常',
      fusedScore: 0.1,
      reply: '听到啦，欢迎你来。是发生了什么事，还是只是想找人随便聊聊？两种都可以。',
    },
    {
      intent: 'CHAT',
      emotionLabel: '正常',
      risk: '正常',
      fusedScore: 0.2,
      reply: '好啊，慢慢说。我在听。',
    },
  ],
};

const RISK_PATTERNS = /想消失|想自杀|撑不下去|不想活|活不下去|self.?harm|suicide|kill myself/i;
const CONSULT_PATTERNS = /焦虑|压力|睡不着|压抑|难过|低落|抑郁|stress|anxiety|depressed|sad|exhausted|burned out/i;

function categorize(text: string): Category {
  if (RISK_PATTERNS.test(text)) return 'RISK';
  if (CONSULT_PATTERNS.test(text)) return 'CONSULT';
  return 'CHAT';
}

function pickReply(category: Category): ReplyTemplate {
  const pool = REPLIES[category];
  return pool[Math.floor(Math.random() * pool.length)];
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    function onAbort() {
      clearTimeout(timer);
      reject(new DOMException('aborted', 'AbortError'));
    }
    if (signal?.aborted) {
      clearTimeout(timer);
      reject(new DOMException('aborted', 'AbortError'));
      return;
    }
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

export async function streamMockChat(
  payload: ChatPayload,
  cb: ChatCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  try {
    const pick = pickReply(categorize(payload.message));

    await sleep(META_DELAY_MS, signal);
    const meta: ChatMeta = {
      sessionId: payload.sessionId ?? crypto.randomUUID(),
      intent: pick.intent,
      emotionLabel: pick.emotionLabel,
      risk: pick.risk,
      fusedScore: pick.fusedScore,
    };
    cb.onMeta?.(meta);

    await sleep(FIRST_TOKEN_DELAY_MS, signal);

    for (const ch of pick.reply) {
      if (signal?.aborted) return;
      cb.onToken?.(ch);
      await sleep(TOKEN_DELAY_MS, signal);
    }

    cb.onDone?.();
  } catch (err) {
    if ((err as Error).name === 'AbortError') return;
    cb.onError?.(err);
  }
}
