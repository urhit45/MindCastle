import React from 'react'
import './styles.css'
import ChatDock from './components/ChatDock'
import ApiDemo from './components/ApiDemo'
import OneNextTaskView, { type EngineSwitchRequest, type NextTaskCandidate } from './components/OneNextTaskView'
import FocusMode from './components/FocusMode'
import TransitionMode from './components/TransitionMode'

function App() {
  const [focusCandidate, setFocusCandidate] = React.useState<NextTaskCandidate | null>(null)
  const [currentCandidate, setCurrentCandidate] = React.useState<NextTaskCandidate | null>(null)
  const [transition, setTransition] = React.useState<EngineSwitchRequest | null>(null)
  const [transitionModeEnabled, setTransitionModeEnabled] = React.useState<boolean>(() => {
    if (typeof window === 'undefined') return true
    const raw = window.localStorage.getItem('mc_transition_mode_enabled')
    return raw ? raw === '1' : true
  })

  React.useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem('mc_transition_mode_enabled', transitionModeEnabled ? '1' : '0')
  }, [transitionModeEnabled])

  const onExitFocusMode = (reason: 'done' | 'break') => {
    console.log(`Focus Mode exit: ${reason}`)
    if (transitionModeEnabled && focusCandidate) {
      setTransition({
        fromEngine: focusCandidate.engineLabel,
        toEngine: 'Castle Front Door',
        toNextStep: currentCandidate?.task ?? 'Pick the next calm task and continue.',
      })
    }
    setFocusCandidate(null)
  }

  return (
    <div className="app-shell">
      <div className="app-backdrop" />

      <div className="app-content">
        <header className="hero">
          <p className="hero-kicker">Mind Web Interface</p>
          <h1>TinyNet</h1>
          <p className="hero-subtitle">Chart the quest. Train your thinking. Build momentum daily.</p>
        </header>

        <OneNextTaskView
          onStartFocusMode={setFocusCandidate}
          onEngineSwitch={(request) => {
            if (transitionModeEnabled) setTransition(request)
          }}
          onCandidateChange={setCurrentCandidate}
          transitionModeEnabled={transitionModeEnabled}
          onTransitionModeEnabledChange={setTransitionModeEnabled}
        />

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
      {focusCandidate && <FocusMode candidate={focusCandidate} onExit={onExitFocusMode} />}
      {transition && <TransitionMode transition={transition} onComplete={() => setTransition(null)} />}
    </div>
  )
}

export default App
