export type Intent = 'CHAT' | 'CONSULT' | 'RISK';
export type Risk = '正常' | '需关注' | '高风险';
export type Emotion = '正常' | '焦虑' | '低落' | '高风险';

export const EMOTION_SCORE: Record<Emotion, number> = {
  正常: 0,
  焦虑: 2,
  低落: 3,
  高风险: 4,
};

export interface ChatMeta {
  sessionId: string;
  intent: Intent;
  emotionLabel: Emotion;
  risk: Risk;
  fusedScore: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  lastMeta?: ChatMeta;
}
