import type { ChatMeta, Emotion, Intent, Risk } from '@/types/chat';

export const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '');

export function isMockMode(): boolean {
  if (typeof window === 'undefined') return false;
  const q = new URLSearchParams(window.location.search).get('mock');
  if (q === '1') return true;
  if (q === '0') return false;
  return import.meta.env.PROD && !API_BASE;
}

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
