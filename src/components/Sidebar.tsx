import React from 'react';
import { PlusCircle } from 'lucide-react';
import { useChatStore } from '../store/chatStore';

export const Sidebar: React.FC = () => {
  const { chats, currentChatId, addChat, setCurrentChat } = useChatStore();

  return (
    <div className="w-64 border-r dark:border-gray-700 p-4">
      <button
        onClick={addChat}
        className="w-full flex items-center gap-2 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 mb-4"
      >
        <PlusCircle className="w-5 h-5" />
        New Chat
      </button>

      <div className="space-y-2">
        {chats.map((chat) => (
          chat.messages.length > 0 && (
            <button
              key={chat.id}
              onClick={() => setCurrentChat(chat.id)}
              className={`w-full p-2 rounded-lg text-left truncate ${
                chat.id === currentChatId
                  ? 'bg-gray-100 dark:bg-gray-800'
                  : 'hover:bg-gray-50 dark:hover:bg-gray-900'
              }`}
            >
              {chat.messages[0]?.content.slice(0, 30)}...
            </button>
          )
        ))}
      </div>
    </div>
  );
};