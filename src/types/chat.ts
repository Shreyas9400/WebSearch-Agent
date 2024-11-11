export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface Chat {
  id: string;
  messages: ChatMessage[];
  createdAt: Date;
}

export interface ChatSettings {
  numResults: number;
  maxChars: number;
  scoreThreshold: number;
  temperature: number;
  scoringMethod: 'BM25' | 'TF-IDF' | 'Combined';
  engines: string[];
  safeSearch: 'Strict (2)' | 'Moderate (1)' | 'None (0)';
  language: string;
  searchMode: 'Auto (Knowledge Base + Web)' | 'Web Search Only';
}

export const defaultSettings: ChatSettings = {
  numResults: 10,
  maxChars: 10000,
  scoreThreshold: 0.8,
  temperature: 0.1,
  scoringMethod: 'Combined',
  engines: ['google', 'bing', 'duckduckgo'],
  safeSearch: 'Moderate (1)',
  language: 'all - All Languages',
  searchMode: 'Auto (Knowledge Base + Web)'
};