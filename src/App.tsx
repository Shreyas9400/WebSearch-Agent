import React, { useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatWindow } from './components/ChatWindow';
import { Settings } from './components/Settings';
import { useChatStore } from './store/chatStore';


function App() {
  const { isDarkMode, addChat, currentChatId } = useChatStore();

  useEffect(() => {
    if (!currentChatId) {
      addChat();
    }
    // Apply dark mode to document
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [currentChatId, addChat, isDarkMode]);

  return (
    <div className="h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <div className="flex h-full">
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <div className="p-4 border-b dark:border-gray-700">
            <Settings />
          </div>
          <div className="flex-1 overflow-hidden">
            <ChatWindow />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;