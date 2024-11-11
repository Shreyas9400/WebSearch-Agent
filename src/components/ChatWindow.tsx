import React, { useRef, useEffect } from 'react';
import { Send, RotateCcw, Undo, Trash2 } from 'lucide-react';
import { useChatStore } from '../store/chatStore';
import { ChatMessage } from '../types/chat';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export const ChatWindow: React.FC = () => {
  const {
    currentChatId,
    chats,
    settings,
    addMessage,
    retryLastMessage,
    undoLastMessage,
    clearChat,
  } = useChatStore();
  const [input, setInput] = React.useState('');
  const [isLoading, setIsLoading] = React.useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const currentChat = chats.find((chat) => chat.id === currentChatId);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentChat?.messages]);

  const sendMessageToAPI = async (message: string, history: ChatMessage[]) => {
    setIsLoading(true);
    try {
      const apiSettings = {
        num_results: settings.numResults,
        max_chars: settings.maxChars,
        score_threshold: settings.scoreThreshold,
        temperature: settings.temperature,
        scoring_method: settings.scoringMethod,
        engines: settings.engines,
        safe_search: settings.safeSearch,
        language: settings.language,
        search_mode: settings.searchMode,
      };

      const formattedHistory = history.map((msg, index, arr) => {
        if (msg.role === 'user') {
          const nextMsg = arr[index + 1];
          return [msg.content, nextMsg && nextMsg.role === 'assistant' ? nextMsg.content : ''];
        }
        return null;
      }).filter(Boolean);

      const response = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          history: formattedHistory,
          ...apiSettings,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.response;
    } catch (error) {
      console.error('Error:', error);
      return 'Sorry, there was an error processing your request. Please check if the API server is running at http://127.0.0.1:8000/api';
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentChatId || !input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input.trim(),
    };

    addMessage(currentChatId, userMessage);
    setInput('');
    
    const response = await sendMessageToAPI(userMessage.content, currentChat?.messages || []);
    
    addMessage(currentChatId, {
      role: 'assistant',
      content: response,
    });
  };

  const handleRetry = async () => {
    if (!currentChatId || !currentChat || isLoading) return;

    const messages = [...currentChat.messages];
    let lastUserMessageIndex = messages.length - 1;

    while (lastUserMessageIndex >= 0 && messages[lastUserMessageIndex].role !== 'user') {
      lastUserMessageIndex--;
    }

    if (lastUserMessageIndex < 0) return;

    const lastUserMessage = messages[lastUserMessageIndex];
    const historyForApi = messages.slice(0, lastUserMessageIndex);

    retryLastMessage(currentChatId);

    const response = await sendMessageToAPI(lastUserMessage.content, historyForApi);
    
    addMessage(currentChatId, {
      role: 'assistant',
      content: response,
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  if (!currentChat) return null;

  return (
    <div className="flex flex-col h-full bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {currentChat.messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.role === 'user' ? 'justify-end' : 'justify-start'
            } animate-fade-in`}
          >
            <div
              className={`max-w-[80%] rounded-2xl p-4 shadow-sm ${
                message.role === 'user'
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white'
                  : 'bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700'
              } transform transition-all duration-200 hover:shadow-md`}
            >
              <ReactMarkdown 
                remarkPlugins={[remarkGfm]}
                className={`prose ${
                  message.role === 'user' 
                    ? 'prose-invert' 
                    : 'prose-gray dark:prose-invert'
                } max-w-none`}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="border-t dark:border-gray-700 bg-white dark:bg-gray-800 p-4 shadow-lg">
        <div className="flex gap-3 mb-4 justify-end">
          <button
            onClick={handleRetry}
            disabled={isLoading || !currentChat.messages.some(m => m.role === 'user')}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Retry last message"
          >
            <RotateCcw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={() => undoLastMessage(currentChatId)}
            disabled={isLoading || currentChat.messages.length === 0}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Undo last message"
          >
            <Undo className="w-5 h-5" />
          </button>
          <button
            onClick={() => clearChat(currentChatId)}
            disabled={isLoading || currentChat.messages.length === 0}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Clear chat"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="flex gap-3 items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            className="flex-1 p-4 bg-gray-50 dark:bg-gray-700 border-0 rounded-full focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:outline-none transition-shadow duration-200"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="p-4 bg-blue-500 text-white rounded-full hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transform transition-all duration-200 hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
};