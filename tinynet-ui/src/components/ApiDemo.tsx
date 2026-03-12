import React, { useState } from 'react';
import { classifyText, reasonAbout, correctClassification } from '../api/client';

export default function ApiDemo() {
  const [demoText, setDemoText] = useState('shin splints after 2mi run');
  const [classification, setClassification] = useState<any>(null);
  const [reasoning, setReasoning] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const runDemo = async () => {
    setIsLoading(true);
    try {
      // Step 1: Classify
      console.log('🔍 Classifying text...');
      const classifyResult = await classifyText(demoText);
      setClassification(classifyResult);
      console.log('✅ Classification:', classifyResult);

      // Step 2: If suggest_plan, get reasoning
      if (classifyResult.route === 'suggest_plan') {
        console.log('🧠 Getting reasoning...');
        const reasonResult = await reasonAbout({
          text: demoText,
          pred: {
            cats: classifyResult.categories.map((c: any) => c.label),
            state: classifyResult.state.label
          }
        });
        setReasoning(reasonResult);
        console.log('✅ Reasoning:', reasonResult);
      }

    } catch (error) {
      console.error('❌ Demo failed:', error);
      alert(`Demo failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  const testCorrection = async () => {
    if (!classification) return;
    
    try {
      const result = await correctClassification({
        text: demoText,
        categories: ['Fitness', 'Running'],
        state: 'blocked'
      });
      console.log('✅ Correction sent:', result);
      alert('Correction sent successfully!');
    } catch (error) {
      console.error('❌ Correction failed:', error);
      alert(`Correction failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  return (
    <div className="panel demo-panel">
      <div className="panel-header">
        <h2>Scout Console</h2>
        <p>Probe classify, reason, and live corrections</p>
      </div>
      
      <div className="demo-stack">
        <div className="field">
          <label>
            Test Text:
          </label>
          <input
            type="text"
            value={demoText}
            onChange={(e) => setDemoText(e.target.value)}
            className="text-input"
            placeholder="Enter text to test..."
          />
        </div>

        <button
          onClick={runDemo}
          disabled={isLoading}
          className="primary-btn full-btn"
        >
          {isLoading ? 'Running Demo...' : '🚀 Run Demo'}
        </button>

        {classification && (
          <div className="result-card result-card-blue">
            <h3>Classification Result</h3>
            <div className="result-body">
              <div><strong>Route:</strong> <span className="route-pill">{classification.route}</span></div>
              <div><strong>Categories:</strong> {classification.categories.map((c: any) => c.label).join(', ')}</div>
              <div><strong>State:</strong> {classification.state.label} ({(classification.state.score * 100).toFixed(1)}%)</div>
              <div><strong>Uncertain:</strong> {classification.uncertain ? 'Yes' : 'No'}</div>
            </div>
          </div>
        )}

        {reasoning && (
          <div className="result-card result-card-green">
            <h3>Reasoning Result</h3>
            <div className="result-body">
              <div>
                <strong>Subtasks:</strong>
                <ul className="bulleted-list">
                  {reasoning.subtasks.map((task: string, i: number) => (
                    <li key={i}>{task}</li>
                  ))}
                </ul>
              </div>
              <div>
                <strong>Blockers:</strong> {reasoning.blockers.join(', ')}
              </div>
              <div>
                <strong>Next Step:</strong> {reasoning.next_step.template}
                {reasoning.next_step.slots && (
                  <span className="slot-text">
                    ({Object.entries(reasoning.next_step.slots).map(([k, v]) => `${k}: ${v}`).join(', ')})
                  </span>
                )}
              </div>
            </div>
          </div>
        )}

        {classification && (
          <div className="result-card result-card-amber">
            <h3>Test Online Learning</h3>
            <p className="result-body">
              Test the correction endpoint to see online learning in action.
            </p>
            <button
              onClick={testCorrection}
              className="pill-btn"
            >
              Send Test Correction
            </button>
          </div>
        )}
      </div>

      <div className="demo-tip">
        <p>💡 Try these examples:</p>
        <ul className="bulleted-list">
          <li>"shin splints after 2mi run" → Should trigger suggest_plan</li>
          <li>"need to send important email" → Should trigger Admin email rules</li>
          <li>"want to learn guitar" → Should trigger generic start state</li>
        </ul>
      </div>
    </div>
  );
}
