import React from 'react';
import { Settings as SettingsIcon, Sun, Moon } from 'lucide-react';
import { useChatStore } from '../store/chatStore';

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
}

interface Settings {
  searchMode: 'Web Search Only' | 'Auto (Knowledge Base + Web)';
  numResults: number;
  maxChars: number;
  scoreThreshold: number;
  temperature: number;
  scoringMethod: 'BM25' | 'TF-IDF' | 'Combined';
  engines: string[];
  safeSearch: 'Strict (2)' | 'Moderate (1)' | 'None (0)';
  language: string;
}

export const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ checked, onChange, label }) => (
  <label className="inline-flex items-center cursor-pointer">
    <div className="relative">
      <input
        type="checkbox"
        className="sr-only"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <div 
        className={`w-11 h-6 rounded-full border transition-colors duration-200 ${
          checked 
            ? 'bg-blue-500 border-blue-600' 
            : 'bg-gray-200 dark:bg-gray-700 dark:border-gray-600'
        }`}
      >
        <div
          className={`absolute w-4 h-4 rounded-full transition-transform duration-200 ease-in-out bg-white transform ${
            checked ? 'translate-x-6' : 'translate-x-1'
          } top-1`}
        />
      </div>
    </div>
    <span className="ml-3 text-sm font-medium">{label}</span>
  </label>
);

const SEARCH_ENGINES = ['google', 'bing', 'duckduckgo', 'brave', 'wikipedia'] as const;
const LANGUAGES = [
  { value: 'all', label: 'All Languages' },
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Spanish' },
  { value: 'fr', label: 'French' },
  { value: 'de', label: 'German' },
  { value: 'it', label: 'Italian' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'ru', label: 'Russian' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ja', label: 'Japanese' },
  { value: 'ko', label: 'Korean' },
] as const;

export const Settings: React.FC = () => {
  const { settings, updateSettings, isDarkMode, toggleDarkMode } = useChatStore();
  const [isOpen, setIsOpen] = React.useState(false);
  const settingsRef = React.useRef<HTMLDivElement>(null);

  const isWebSearchOnly = settings.searchMode === 'Web Search Only';

  const handleClickOutside = React.useCallback((event: MouseEvent) => {
    // Check if the click is outside both the settings panel and the settings button
    if (
      settingsRef.current && 
      !settingsRef.current.contains(event.target as Node) &&
      !(event.target as Element).closest('[aria-label="Open settings"]')
    ) {
      setIsOpen(false);
    }
  }, []);

  // Toggle settings panel
  const toggleSettings = () => {
    setIsOpen(prev => !prev);
  };

  React.useEffect(() => {
    // Only add the event listener if the panel is open
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      // Add escape key listener
      const handleEscape = (event: KeyboardEvent) => {
        if (event.key === 'Escape') {
          setIsOpen(false);
        }
      };
      document.addEventListener('keydown', handleEscape);

      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
        document.removeEventListener('keydown', handleEscape);
      };
    }
  }, [isOpen, handleClickOutside]);

  return (
    <div className="relative" ref={settingsRef}>
      <div className="flex items-center gap-4">
        <button
          onClick={() => toggleDarkMode()}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
          aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDarkMode ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
        </button>

        <ToggleSwitch
          checked={isWebSearchOnly}
          onChange={(checked) =>
            updateSettings({
              searchMode: checked
                ? 'Web Search Only'
                : 'Auto (Knowledge Base + Web)',
            })
          }
          label="Web Search Only"
        />

        <button
          onClick={toggleSettings}
          className={`p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 ${
            isOpen ? 'bg-gray-100 dark:bg-gray-800' : ''
          }`}
          aria-label="Open settings"
          aria-expanded={isOpen}
        >
          <SettingsIcon className="w-5 h-5" />
        </button>
      </div>

      {isOpen && (
        <div 
          className="settings-panel absolute right-0 top-12 w-80 bg-white dark:bg-gray-900 border dark:border-gray-700 rounded-lg shadow-lg p-4 z-50 animate-in fade-in slide-in-from-top-2 duration-200"
        >
          <div className="space-y-4 max-h-[80vh] overflow-y-auto">
            <div>
              <label htmlFor="numResults" className="block text-sm mb-1">Number of Results</label>
              <input
                id="numResults"
                type="range"
                min="5"
                max="30"
                value={settings.numResults}
                onChange={(e) =>
                  updateSettings({ numResults: parseInt(e.target.value) })
                }
                className="w-full"
              />
              <span className="text-sm">{settings.numResults}</span>
            </div>

            <div>
              <label htmlFor="maxChars" className="block text-sm mb-1">Max Characters</label>
              <input
                id="maxChars"
                type="range"
                min="1000"
                max="50000"
                step="1000"
                value={settings.maxChars}
                onChange={(e) =>
                  updateSettings({ maxChars: parseInt(e.target.value) })
                }
                className="w-full"
              />
              <span className="text-sm">{settings.maxChars.toLocaleString()}</span>
            </div>

            <div>
              <label htmlFor="scoreThreshold" className="block text-sm mb-1">Score Threshold</label>
              <input
                id="scoreThreshold"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.scoreThreshold}
                onChange={(e) =>
                  updateSettings({ scoreThreshold: parseFloat(e.target.value) })
                }
                className="w-full"
              />
              <span className="text-sm">{settings.scoreThreshold}</span>
            </div>

            <div>
              <label htmlFor="temperature" className="block text-sm mb-1">Temperature</label>
              <input
                id="temperature"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.temperature}
                onChange={(e) =>
                  updateSettings({ temperature: parseFloat(e.target.value) })
                }
                className="w-full"
              />
              <span className="text-sm">{settings.temperature}</span>
            </div>

            <div>
              <label htmlFor="scoringMethod" className="block text-sm mb-1">Scoring Method</label>
              <select
                id="scoringMethod"
                value={settings.scoringMethod}
                onChange={(e) =>
                  updateSettings({ scoringMethod: e.target.value as Settings['scoringMethod'] })
                }
                className="w-full p-2 rounded border dark:bg-gray-800 dark:border-gray-700"
              >
                <option value="BM25">BM25</option>
                <option value="TF-IDF">TF-IDF</option>
                <option value="Combined">Combined</option>
              </select>
            </div>

            <div>
              <label className="block text-sm mb-1">Search Engines</label>
              <div className="space-y-2">
                {SEARCH_ENGINES.map((engine) => (
                  <label key={engine} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings.engines.includes(engine)}
                      onChange={(e) => {
                        const newEngines = e.target.checked
                          ? [...settings.engines, engine]
                          : settings.engines.filter((e) => e !== engine);
                        updateSettings({ engines: newEngines });
                      }}
                      className="mr-2"
                    />
                    {engine.charAt(0).toUpperCase() + engine.slice(1)}
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label htmlFor="safeSearch" className="block text-sm mb-1">Safe Search</label>
              <select
                id="safeSearch"
                value={settings.safeSearch}
                onChange={(e) =>
                  updateSettings({ safeSearch: e.target.value as Settings['safeSearch'] })
                }
                className="w-full p-2 rounded border dark:bg-gray-800 dark:border-gray-700"
              >
                <option value="Strict (2)">Strict</option>
                <option value="Moderate (1)">Moderate</option>
                <option value="None (0)">None</option>
              </select>
            </div>

            <div>
              <label htmlFor="language" className="block text-sm mb-1">Language</label>
              <select
                id="language"
                value={settings.language}
                onChange={(e) => updateSettings({ language: e.target.value })}
                className="w-full p-2 rounded border dark:bg-gray-800 dark:border-gray-700"
              >
                {LANGUAGES.map(({ value, label }) => (
                  <option key={value} value={`${value} - ${label}`}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};