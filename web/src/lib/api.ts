import type { ChatMeta, Emotion, Intent, Risk } from '@/types/chat';

interface WireChatMeta {
  session_id: string;
  intent: Intent;
  emotion_label: Emotion;
  risk: Risk;
  fused_score: number;
}

export function parseMeta(json: string): ChatMeta {
  const raw = JSON.parse(json) as WireChatMeta;
  return {
    sessionId: raw.session_id,
    intent: raw.intent,
    emotionLabel: raw.emotion_label,
    risk: raw.risk,
    fusedScore: raw.fused_score,
  };
}

export interface ChatPayload {
  userId: string;
  message: string;
  sessionId?: string;
  audioEmotion?: Emotion;
  videoEmotion?: Emotion;
  videoScore?: number;
}

export function toWireChatPayload(p: ChatPayload) {
  return {
    user_id: p.userId,
    message: p.message,
    session_id: p.sessionId,
    audio_emotion: p.audioEmotion,
    video_emotion: p.videoEmotion,
    video_score: p.videoScore,
  };
}
