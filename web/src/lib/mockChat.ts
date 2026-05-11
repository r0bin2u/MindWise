// Mock picks a reply by:
//   (1) category — RISK / CONSULT / CHAT — keyword routing on the user text
//   (2) language — zh / en — CJK-ratio heuristic on the same text
//
// The bilingual pool exists only here in the mock. The real Qwen backend is
// fine-tuned on a Chinese psychology corpus and replies in Chinese regardless
// of input language; mock pretends Qwen is bilingual so the public Vercel
// demo feels coherent for english-speaking reviewers. The meta values
// (intent / emotionLabel / risk / fusedScore) stay identical across languages
// because those reflect the (fictional) classifier output, not the reply text.
//
// All fusedScore values mirror the real text-only fusion math:
//   score = text_score × 0.1, where text_score ∈ {正常:0, 焦虑:2, 低落:3, 高风险:4}
// In text-only mode the maximum reachable score is 0.4, so risk stays "正常";
// crisis surfaces via intent === 'RISK', not via the risk band.

import type { ChatPayload } from './api';
import type { ChatCallbacks } from '@/hooks/useChat';
import type { ChatMeta, Emotion, Intent, Risk } from '@/types/chat';

const TOKEN_DELAY_MS = 30;
const META_DELAY_MS = 150;
const FIRST_TOKEN_DELAY_MS = 400;

type Category = 'RISK' | 'CONSULT' | 'CHAT';
type Lang = 'zh' | 'en';

interface ReplyTemplate {
  intent: Intent;
  emotionLabel: Emotion;
  risk: Risk;
  fusedScore: number;
  reply: Record<Lang, string>;
}

const REPLIES: Record<Category, ReplyTemplate[]> = {
  RISK: [
    {
      intent: 'RISK',
      emotionLabel: '高风险',
      risk: '正常',
      fusedScore: 0.4,
      reply: {
        zh: '谢谢你愿意把这些告诉我。"想消失"这四个字背后，压着很多说不出口的疲惫。我想先和你确认一下，你现在是安全的吗？身边有可以联系到的人吗？学校心理中心 24 小时热线是 xxx-xxxx-xxxx，可以现在就拨过去。',
        en: 'Thank you for telling me this. Words like "I want to disappear" carry a lot of unspoken exhaustion. First, can I check with you — are you safe right now? Is there someone you can reach out to? Please call 988 (US Suicide & Crisis Lifeline) or your campus counseling center; you do not have to explain everything, you just need someone with you.',
      },
    },
    {
      intent: 'RISK',
      emotionLabel: '高风险',
      risk: '正常',
      fusedScore: 0.4,
      reply: {
        zh: '我听到了，撑到现在已经不容易了。请先做一件事：把这个号码拨出去 —— 北京心理危机研究与干预中心 010-82951332，他们 24 小时有人接听。你不需要解释那么多，只要让对方陪你说会儿话。',
        en: 'I hear you. Holding on this far has already taken so much. Please do one thing before anything else: dial 988 (US) or 116 123 (Samaritans, UK) — both lines are 24/7. You do not have to explain everything; you just need a real person there while you breathe.',
      },
    },
  ],
  CONSULT: [
    {
      intent: 'CONSULT',
      emotionLabel: '焦虑',
      risk: '正常',
      fusedScore: 0.2,
      reply: {
        zh: '听起来你最近背的东西很重。睡不好、心慌、白天又没法停下来 —— 这种状态身体迟早会发出信号。能告诉我，最近一次让你觉得"撑不住"的时刻，是在做什么吗？',
        en: 'It sounds like you have been carrying a lot. The sleep trouble, the racing thoughts, not being able to slow down during the day — your body will eventually start signaling. Can you tell me about the most recent moment when it felt "too much"? What were you doing then?',
      },
    },
    {
      intent: 'CONSULT',
      emotionLabel: '焦虑',
      risk: '正常',
      fusedScore: 0.2,
      reply: {
        zh: '压力这件事不是非要"扛过去"才算赢。我们可以一起拆一拆 —— 是哪一块在消耗你最多？是 deadline 本身、是对结果的担心、还是没有人能商量？',
        en: 'Stress is not something you have to muscle through to count as "winning". Let us break it apart — what is draining you most right now? The deadlines themselves, the worry about how things turn out, or not having anyone to talk it through with?',
      },
    },
    {
      intent: 'CONSULT',
      emotionLabel: '低落',
      risk: '正常',
      fusedScore: 0.3,
      reply: {
        zh: '谢谢你说出来。这种低落感像在水底走路，每一步都需要更多力气。你愿意告诉我，是从什么时候开始有这种感觉的吗？',
        en: 'Thank you for naming this. That low-mood feeling can be like walking through water — every step takes more energy than it should. When did this start, do you remember? Did something shift recently, or has it been building for a while?',
      },
    },
  ],
  CHAT: [
    {
      intent: 'CHAT',
      emotionLabel: '正常',
      risk: '正常',
      fusedScore: 0.0,
      reply: {
        zh: '挺好的呀，今天怎么样？有什么想和我聊的吗？',
        en: 'Hey, glad you are here. How is your day going? Anything you feel like talking about?',
      },
    },
    {
      intent: 'CHAT',
      emotionLabel: '正常',
      risk: '正常',
      fusedScore: 0.0,
      reply: {
        zh: '听到啦，欢迎你来。是发生了什么事，还是只是想找人随便聊聊？两种都可以。',
        en: 'I am here, welcome. Did something happen, or did you just feel like having someone to chat with? Either is fine.',
      },
    },
    {
      intent: 'CHAT',
      emotionLabel: '正常',
      risk: '正常',
      fusedScore: 0.0,
      reply: {
        zh: '好啊，慢慢说。我在听。',
        en: 'Sure thing. Take your time, I am listening.',
      },
    },
  ],
};

const RISK_PATTERNS =
  /想消失|想自杀|撑不下去|不想活|活不下去|self.?harm|suicide|suicidal|kill myself|want to disappear|can'?t go on|end it all/i;
const CONSULT_PATTERNS =
  /焦虑|压力|睡不着|压抑|难过|低落|抑郁|stress|stressed|anxiety|anxious|depress(?:ed|ion)|sad(?!\w)|exhausted|burn(?:ed|t) out|overwhelmed/i;

function categorize(text: string): Category {
  if (RISK_PATTERNS.test(text)) return 'RISK';
  if (CONSULT_PATTERNS.test(text)) return 'CONSULT';
  return 'CHAT';
}

// cjkCount * 3 > text.length is the integer-only form of "CJK ratio > 1/3",
// which avoids dividing when text is empty.
function detectLang(text: string): Lang {
  const cjkCount = (text.match(/[一-龥]/g) ?? []).length;
  return cjkCount * 3 > text.length ? 'zh' : 'en';
}

function pickReply(category: Category): ReplyTemplate {
  const pool = REPLIES[category];
  return pool[Math.floor(Math.random() * pool.length)];
}

// zh: one CJK char per emit (matches Qwen's per-character emit on Chinese).
// en: chunk by "word + trailing whitespace" so the stream feels word-level
// the way a BPE model emits English.
function tokenize(text: string, lang: Lang): string[] {
  if (lang === 'zh') return Array.from(text);
  return text.match(/\S+\s*|\s+/g) ?? [text];
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
    const lang = detectLang(payload.message);
    const pick = pickReply(categorize(payload.message));
    const replyText = pick.reply[lang];

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

    for (const tok of tokenize(replyText, lang)) {
      if (signal?.aborted) return;
      cb.onToken?.(tok);
      await sleep(TOKEN_DELAY_MS, signal);
    }

    cb.onDone?.();
  } catch (err) {
    if ((err as Error).name === 'AbortError') return;
    cb.onError?.(err);
  }
}
