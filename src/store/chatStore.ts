import { create } from 'zustand';
import { Chat, ChatMessage, ChatSettings, defaultSettings } from '../types/chat';

interface ChatStore {
  chats: Chat[];
  currentChatId: string | null;
  settings: ChatSettings;
  isDarkMode: boolean;
  addChat: () => void;
  setCurrentChat: (id: string) => void;
  addMessage: (chatId: string, message: ChatMessage) => void;
  updateSettings: (settings: Partial<ChatSettings>) => void;
  toggleDarkMode: () => void;
  retryLastMessage: (chatId: string) => void;
  undoLastMessage: (chatId: string) => void;
  clearChat: (chatId: string) => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  chats: [],
  currentChatId: null,
  settings: defaultSettings,
  isDarkMode: false,

  addChat: () => {
    set((state) => {
      const newChat: Chat = {
        id: crypto.randomUUID(),
        messages: [],
        createdAt: new Date(),
      };
      return {
        chats: [...state.chats, newChat],
        currentChatId: newChat.id,
      };
    });
  },

  setCurrentChat: (id) => set({ currentChatId: id }),

  addMessage: (chatId, message) => {
    set((state) => ({
      chats: state.chats.map((chat) =>
        chat.id === chatId
          ? { ...chat, messages: [...chat.messages, message] }
          : chat
      ),
    }));
  },

  updateSettings: (newSettings) =>
    set((state) => ({
      settings: { ...state.settings, ...newSettings },
    })),

  toggleDarkMode: () =>
    set((state) => ({ isDarkMode: !state.isDarkMode })),

  retryLastMessage: (chatId) => {
    set((state) => {
      const chat = state.chats.find((c) => c.id === chatId);
      if (!chat) return state;

      const lastUserMessage = [...chat.messages]
        .reverse()
        .find((m) => m.role === 'user');
      if (!lastUserMessage) return state;

      return {
        chats: state.chats.map((c) =>
          c.id === chatId
            ? {
                ...c,
                messages: c.messages.filter(
                  (m) => m !== c.messages[c.messages.length - 1]
                ),
              }
            : c
        ),
      };
    });
  },

  undoLastMessage: (chatId) => {
    set((state) => ({
      chats: state.chats.map((chat) =>
        chat.id === chatId
          ? {
              ...chat,
              messages: chat.messages.slice(0, -2),
            }
          : chat
      ),
    }));
  },

  clearChat: (chatId) => {
    set((state) => ({
      chats: state.chats.map((chat) =>
        chat.id === chatId ? { ...chat, messages: [] } : chat
      ),
    }));
  },
}));