import React, { useState, useRef, useEffect } from 'react';
import { classifyText, reasonAbout, correctClassification, type ClassifyResponse, type ReasonResponse } from '../api/client';

interface Message {
  id: string;
  from: 'user' | 'bot';
  text: string;
  timestamp: Date;
  classification?: ClassifyResponse;
  reasoning?: ReasonResponse;
}

export default function ChatDock() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (message: Omit<Message, 'id' | 'timestamp'>) => {
    const newMessage: Message = {
      ...message,
      id: Date.now().toString(),
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isProcessing) return;

    const userText = inputText.trim();
    setInputText('');
    setIsProcessing(true);

    // Add user message
    addMessage({
      from: 'user',
      text: userText
    });

    try {
      // Step 1: Classify the text
      const classification = await classifyText(userText);
      
      // Add bot response with classification
      addMessage({
        from: 'bot',
        text: `I see this as: ${classification.categories.map(c => c.label).join(', ')} (${classification.state.label})`,
        classification
      });

      // Step 2: Handle routing based on classification
      if (classification.route === "needs_confirm") {
        addMessage({
          from: 'bot',
          text: `🤔 I'm not entirely sure. Could you confirm the categories and state?`
        });
      } else if (classification.route === "suggest_plan") {
        // Get intelligent planning suggestions
        const reasoning = await reasonAbout({
          text: userText,
          pred: {
            cats: classification.categories.map(c => c.label),
            state: classification.state.label
          }
        });

        // Add reasoning response
        addMessage({
          from: 'bot',
          text: `📋 Here's a plan to help: ${reasoning.subtasks.join(' → ')}`,
          reasoning
        });

        // Add blockers if any
        if (reasoning.blockers.length > 0) {
          addMessage({
            from: 'bot',
            text: `⚠️ Watch out for: ${reasoning.blockers.join(', ')}`
          });
        }

        // Add next step suggestion
        if (reasoning.next_step) {
          addMessage({
            from: 'bot',
            text: `🎯 Next: ${reasoning.next_step.template} ${Object.entries(reasoning.next_step.slots || {}).map(([k, v]) => `${k}: ${v}`).join(', ')}`
          });
        }

      } else if (classification.route === "auto_save_ok") {
        addMessage({
          from: 'bot',
          text: `✅ Got it! This looks clear and actionable.`
        });
      }

    } catch (error) {
      console.error('Error processing message:', error);
      addMessage({
        from: 'bot',
        text: `❌ Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCorrection = async (messageId: string, correction: {
    categories?: string[];
    state?: string;
    linkTo?: string;
    nextStepTemplate?: string;
  }) => {
    const message = messages.find(m => m.id === messageId);
    if (!message || !message.classification) return;

    try {
      console.log('🔄 Sending correction:', { text: message.text, ...correction });
      
      const result = await correctClassification({
        text: message.text,
        ...correction
      });

      console.log('✅ Correction result:', result);

      addMessage({
        from: 'bot',
        text: `✅ Thanks for the correction! I'll learn from this. The model has been updated.`
      });

      // Optionally, re-classify the text to see the improvement
      if (correction.categories || correction.state) {
        addMessage({
          from: 'bot',
          text: `🔄 Let me re-classify that with your feedback...`
        });

        try {
          const newClassification = await classifyText(message.text);
          addMessage({
            from: 'bot',
            text: `📊 New classification: ${newClassification.categories.map(c => c.label).join(', ')} (${newClassification.state.label}) - Route: ${newClassification.route}`,
            classification: newClassification
          });
        } catch (reclassifyError) {
          console.error('Error re-classifying:', reclassifyError);
        }
      }

    } catch (error) {
      console.error('Error sending correction:', error);
      addMessage({
        from: 'bot',
        text: `❌ Failed to save correction: ${error instanceof Error ? error.message : 'Unknown error'}`
      });
    }
  };

  const renderMessage = (message: Message) => {
    const isUser = message.from === 'user';
    
    return (
      <div
        key={message.id}
        className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
      >
        <div
          className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
            isUser
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 text-gray-800'
          }`}
        >
          <div className="text-sm">{message.text}</div>
          
          {/* Show classification details for bot messages */}
          {!isUser && message.classification && (
            <div className="mt-2 text-xs opacity-75">
              <div>Route: {message.classification.route}</div>
              <div>Confidence: {Math.round(message.classification.state.score * 100)}%</div>
            </div>
          )}

          {/* Show reasoning details if available */}
          {!isUser && message.reasoning && (
            <div className="mt-2 text-xs opacity-75">
              <div>Subtasks: {message.reasoning.subtasks.length}</div>
              <div>Blockers: {message.reasoning.blockers.length}</div>
            </div>
          )}

          {/* Correction buttons for user messages */}
          {isUser && (
            <div className="mt-2 flex gap-1">
              <button
                onClick={() => handleCorrection(message.id, { categories: ['Fitness', 'Running'] })}
                className="text-xs bg-blue-600 px-2 py-1 rounded hover:bg-blue-700"
                title="Correct categories to Fitness, Running"
              >
                Fix Cats
              </button>
              <button
                onClick={() => handleCorrection(message.id, { state: 'continue' })}
                className="text-xs bg-green-600 px-2 py-1 rounded hover:bg-green-700"
                title="Correct state to continue"
              >
                Fix State
              </button>
              <button
                onClick={() => handleCorrection(message.id, { 
                  categories: ['Fitness', 'Running'], 
                  state: 'continue' 
                })}
                className="text-xs bg-purple-600 px-2 py-1 rounded hover:bg-purple-700"
                title="Fix both categories and state"
              >
                Fix Both
              </button>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-800">TinyNet Chat</h2>
        <p className="text-sm text-gray-600">AI-powered mind web assistant</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 py-8">
            <p>Start a conversation to see how TinyNet can help!</p>
            <p className="text-sm mt-2">Try: "shin splints after 2mi run"</p>
          </div>
        )}
        
        {messages.map(renderMessage)}
        
        {isProcessing && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-200 text-gray-800 px-4 py-2 rounded-lg">
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                <span className="text-sm">Thinking...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200">
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="What's on your mind?"
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isProcessing}
          />
          <button
            type="submit"
            disabled={isProcessing || !inputText.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
