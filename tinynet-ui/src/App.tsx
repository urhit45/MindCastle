import './styles.css'
import ChatDock from './components/ChatDock'
import ApiDemo from './components/ApiDemo'

function App() {
  return (
    <div className="App">
      <div className="min-h-screen bg-gray-50 p-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-2">TinyNet</h1>
            <p className="text-lg text-gray-600">Your AI-powered mind web assistant</p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="h-96">
              <ChatDock />
            </div>
            <div>
              <ApiDemo />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
