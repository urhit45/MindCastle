import './styles.css'
import ChatDock from './components/ChatDock'
import ApiDemo from './components/ApiDemo'

function App() {
  return (
    <div className="app-shell">
      <div className="app-backdrop" />

      <div className="app-content">
        <header className="hero">
          <p className="hero-kicker">Mind Web Interface</p>
          <h1>TinyNet</h1>
          <p className="hero-subtitle">Chart the quest. Train your thinking. Build momentum daily.</p>
        </header>

        <main className="feature-grid">
          <section className="feature-card">
            <ChatDock />
          </section>
          <section className="feature-card">
            <ApiDemo />
          </section>
        </main>
      </div>
      <div className="app-frame" />
    </div>
  )
}

export default App
