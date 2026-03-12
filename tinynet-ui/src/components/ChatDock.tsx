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
        className={`chat-row ${isUser ? 'chat-row-user' : 'chat-row-bot'}`}
      >
        <div
          className={`chat-bubble ${isUser ? 'chat-bubble-user' : 'chat-bubble-bot'}`}
        >
          <div className="chat-text">{message.text}</div>
          
          {/* Show classification details for bot messages */}
          {!isUser && message.classification && (
            <div className="chat-meta">
              <div>Route: {message.classification.route}</div>
              <div>Confidence: {Math.round(message.classification.state.score * 100)}%</div>
            </div>
          )}

          {/* Show reasoning details if available */}
          {!isUser && message.reasoning && (
            <div className="chat-meta">
              <div>Subtasks: {message.reasoning.subtasks.length}</div>
              <div>Blockers: {message.reasoning.blockers.length}</div>
            </div>
          )}

          {/* Correction buttons for user messages */}
          {isUser && (
            <div className="correction-actions">
              <button
                onClick={() => handleCorrection(message.id, { categories: ['Fitness', 'Running'] })}
                className="pill-btn"
                title="Correct categories to Fitness, Running"
              >
                Fix Cats
              </button>
              <button
                onClick={() => handleCorrection(message.id, { state: 'continue' })}
                className="pill-btn"
                title="Correct state to continue"
              >
                Fix State
              </button>
              <button
                onClick={() => handleCorrection(message.id, { 
                  categories: ['Fitness', 'Running'], 
                  state: 'continue' 
                })}
                className="pill-btn"
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
    <div className="panel chat-panel">
      {/* Header */}
      <div className="panel-header">
        <h2>TinyNet Chat</h2>
        <p>Party guide for your focus quests</p>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Start a conversation to see how TinyNet can help!</p>
            <p>Try: "shin splints after 2mi run"</p>
          </div>
        )}
        
        {messages.map(renderMessage)}
        
        {isProcessing && (
          <div className="chat-row chat-row-bot">
            <div className="chat-bubble chat-bubble-bot">
              <div className="typing">
                <div className="spinner" />
                <span>Thinking...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="panel-footer">
        <div className="chat-input-row">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="What's on your mind?"
            className="text-input"
            disabled={isProcessing}
          />
          <button
            type="submit"
            disabled={isProcessing || !inputText.trim()}
            className="primary-btn"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
