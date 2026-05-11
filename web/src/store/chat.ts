import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { useShallow } from 'zustand/react/shallow';
import { MOCK_SESSIONS } from '@/lib/mock';
import type { ChatMessage, ChatMeta, ChatSession } from '@/types/chat';

const TITLE_MAX = 24;
const STORAGE_KEY = 'mindwise.chat';

interface ChatState {
  sessions: Record<string, ChatSession>;
  order: string[];
  currentId: string | null;
  seenCrisis: Record<string, boolean>;

  newSession: () => string;
  switchSession: (id: string) => void;
  addMessages: (sessionId: string, messages: ChatMessage[]) => void;
  appendToMessage: (sessionId: string, messageId: string, text: string) => void;
  setMessageContent: (sessionId: string, messageId: string, content: string) => void;
  setLastMeta: (sessionId: string, meta: ChatMeta) => void;
  markCrisisSeen: (sessionId: string) => void;
}

function seedFromMocks(): Pick<
  ChatState,
  'sessions' | 'order' | 'currentId' | 'seenCrisis'
> {
  const sessions: Record<string, ChatSession> = {};
  const order: string[] = [];
  for (const s of MOCK_SESSIONS) {
    sessions[s.id] = s;
    order.push(s.id);
  }
  return { sessions, order, currentId: order[0] ?? null, seenCrisis: {} };
}

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      ...seedFromMocks(),

      newSession: () => {
        const id = crypto.randomUUID();
        const session: ChatSession = {
          id,
          title: '',
          messages: [],
          createdAt: Date.now(),
        };
        set((state) => ({
          sessions: { ...state.sessions, [id]: session },
          order: [id, ...state.order],
          currentId: id,
        }));
        return id;
      },

      switchSession: (id) => set({ currentId: id }),

      addMessages: (sessionId, messages) =>
        set((state) => {
          const s = state.sessions[sessionId];
          if (!s) return state;
          const merged: ChatSession = { ...s, messages: [...s.messages, ...messages] };
          if (!s.title) {
            const firstUser = messages.find((m) => m.role === 'user');
            if (firstUser) merged.title = firstUser.content.slice(0, TITLE_MAX);
          }
          return { sessions: { ...state.sessions, [sessionId]: merged } };
        }),

      appendToMessage: (sessionId, messageId, text) =>
        set((state) => {
          const s = state.sessions[sessionId];
          if (!s) return state;
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === messageId ? { ...m, content: m.content + text } : m,
                ),
              },
            },
          };
        }),

      setMessageContent: (sessionId, messageId, content) =>
        set((state) => {
          const s = state.sessions[sessionId];
          if (!s) return state;
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === messageId ? { ...m, content } : m,
                ),
              },
            },
          };
        }),

      setLastMeta: (sessionId, meta) =>
        set((state) => {
          const s = state.sessions[sessionId];
          if (!s) return state;
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: { ...s, lastMeta: meta },
            },
          };
        }),

      markCrisisSeen: (sessionId) =>
        set((state) => ({
          seenCrisis: { ...state.seenCrisis, [sessionId]: true },
        })),
    }),
    {
      name: STORAGE_KEY,
      version: 1,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        sessions: state.sessions,
        order: state.order,
        currentId: state.currentId,
        seenCrisis: state.seenCrisis,
      }),
    },
  ),
);

export const useSessions = () =>
  useChatStore(
    useShallow((s) =>
      s.order
        .map((id) => s.sessions[id])
        .filter((v): v is ChatSession => v !== undefined),
    ),
  );

export const useCurrentSession = () =>
  useChatStore((s) => (s.currentId ? s.sessions[s.currentId] : undefined));
