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
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">TinyNet API Demo</h2>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Test Text:
          </label>
          <input
            type="text"
            value={demoText}
            onChange={(e) => setDemoText(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter text to test..."
          />
        </div>

        <button
          onClick={runDemo}
          disabled={isLoading}
          className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
        >
          {isLoading ? 'Running Demo...' : '🚀 Run Demo'}
        </button>

        {classification && (
          <div className="border rounded-lg p-4 bg-gray-50">
            <h3 className="font-semibold mb-2">📊 Classification Result</h3>
            <div className="text-sm space-y-1">
              <div><strong>Route:</strong> <span className="px-2 py-1 bg-blue-100 rounded">{classification.route}</span></div>
              <div><strong>Categories:</strong> {classification.categories.map((c: any) => c.label).join(', ')}</div>
              <div><strong>State:</strong> {classification.state.label} ({(classification.state.score * 100).toFixed(1)}%)</div>
              <div><strong>Uncertain:</strong> {classification.uncertain ? 'Yes' : 'No'}</div>
            </div>
          </div>
        )}

        {reasoning && (
          <div className="border rounded-lg p-4 bg-green-50">
            <h3 className="font-semibold mb-2">🧠 Reasoning Result</h3>
            <div className="text-sm space-y-2">
              <div>
                <strong>Subtasks:</strong>
                <ul className="list-disc list-inside ml-2">
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
                  <span className="ml-2 text-gray-600">
                    ({Object.entries(reasoning.next_step.slots).map(([k, v]) => `${k}: ${v}`).join(', ')})
                  </span>
                )}
              </div>
            </div>
          </div>
        )}

        {classification && (
          <div className="border rounded-lg p-4 bg-yellow-50">
            <h3 className="font-semibold mb-2">🔄 Test Online Learning</h3>
            <p className="text-sm text-gray-600 mb-2">
              Test the correction endpoint to see online learning in action.
            </p>
            <button
              onClick={testCorrection}
              className="px-3 py-1 bg-yellow-500 text-white rounded text-sm hover:bg-yellow-600"
            >
              Send Test Correction
            </button>
          </div>
        )}
      </div>

      <div className="mt-6 text-xs text-gray-500">
        <p>💡 Try these examples:</p>
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>"shin splints after 2mi run" → Should trigger suggest_plan</li>
          <li>"need to send important email" → Should trigger Admin email rules</li>
          <li>"want to learn guitar" → Should trigger generic start state</li>
        </ul>
      </div>
    </div>
  );
}
