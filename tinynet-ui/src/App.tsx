import React from 'react'
import './styles.css'
import ChatDock from './components/ChatDock'
import ApiDemo from './components/ApiDemo'
import OneNextTaskView, { type EngineSwitchRequest, type NextTaskCandidate } from './components/OneNextTaskView'
import FocusMode from './components/FocusMode'
import TransitionMode from './components/TransitionMode'
import MindMapView from './components/MindMapView'
import type { MindMapNode } from './api/client'
import { Onboarding } from './Onboarding'
import * as api from './api'

type OnboardingArtifact = {
  id: string
  title: string
  subtitle: string
  status: string
  next: string
  stack: string[]
  progress: { label: string; val: number }[]
  notes: string
  node_id?: string
}

type OnboardingEngine = {
  id: string
  name: string
  subtitle: string
  color: string
  icon: string
  context: string
  contextLabel: string
  description: string
  artifacts: OnboardingArtifact[]
}

function mapArtifactStatusToLogState(status: string): string {
  const normalized = status.toLowerCase()
  if (normalized === 'blocked') return 'blocked'
  if (normalized === 'planned' || normalized === 'planning') return 'start'
  if (normalized === 'live') return 'continue'
  if (normalized === 'active') return 'continue'
  return 'idea'
}

function App() {
  const [onboardingComplete, setOnboardingComplete] = React.useState<boolean>(() => {
    if (typeof window === 'undefined') return true
    const doneFlag = window.localStorage.getItem('mc_onboarding_complete_v1')
    if (doneFlag === '1') return true
    try {
      const raw = window.localStorage.getItem('mindcastle_v1')
      if (!raw) return false
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) && parsed.length > 0
    } catch {
      return false
    }
  })

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

  const [preferredNodeId, setPreferredNodeId] = React.useState<string | null>(null)

  const onNavigateToEngine = (node: MindMapNode) => {
    setPreferredNodeId(node.id)
    window.location.hash = `engine-${node.id}`
    const el = document.getElementById('one-next-task')
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  const persistOnboardingToBackend = async (engines: OnboardingEngine[]): Promise<OnboardingEngine[]> => {
    const syncedEngines: OnboardingEngine[] = []
    for (const engine of engines) {
      const syncedArtifacts: OnboardingArtifact[] = []
      for (const artifact of engine.artifacts) {
        try {
          const node = await api.createNode(
            artifact.title,
            false,
            artifact.status || 'concept',
          )
          const starterNextStep = artifact.next?.trim() || 'OutlineThreeBullets'
          const starterLogText = artifact.notes?.trim()
            ? `Onboarding import for "${artifact.title}". ${artifact.notes.trim()}`
            : `Onboarding import for "${artifact.title}". Starting baseline context and first move.`
          try {
            await api.addLog(
              node.id,
              starterLogText,
              mapArtifactStatusToLogState(artifact.status || 'concept'),
              starterNextStep,
            )
          } catch {
            // Non-fatal: node persistence still succeeded.
          }
          syncedArtifacts.push({ ...artifact, node_id: node.id })
        } catch {
          // Backend offline should not block onboarding completion.
          syncedArtifacts.push(artifact)
        }
      }
      syncedEngines.push({ ...engine, artifacts: syncedArtifacts })
    }
    return syncedEngines
  }

  const handleOnboardingComplete = async (engines: OnboardingEngine[]) => {
    const syncedEngines = await persistOnboardingToBackend(engines)
    window.localStorage.setItem('mindcastle_v1', JSON.stringify(syncedEngines))
    window.localStorage.setItem('mc_onboarding_complete_v1', '1')
    setOnboardingComplete(true)
  }

  return (
    <div className="app-shell">
      <div className="app-backdrop" />
      <header className="app-topbar">
        <div className="app-topbar-brand">MIND · CASTLE</div>
        <div className="app-topbar-mode">{onboardingComplete ? 'FLOW MODE' : 'ONBOARDING'}</div>
      </header>

      <div className="app-content">
        {!onboardingComplete ? (
          <Onboarding
            onComplete={(engines) => {
              void handleOnboardingComplete(engines as OnboardingEngine[])
            }}
            onSkip={() => {
              window.localStorage.setItem('mc_onboarding_complete_v1', '1')
              setOnboardingComplete(true)
            }}
          />
        ) : (
          <>
        <header className="hero">
          <p className="hero-kicker">MindCastle</p>
          <h1>Cognitive Flow Console</h1>
          <p className="hero-subtitle">One next step, visual pattern awareness, and zero-guilt momentum.</p>
        </header>

        <div id="one-next-task">
          <OneNextTaskView
          onStartFocusMode={setFocusCandidate}
          onEngineSwitch={(request) => {
            if (transitionModeEnabled) setTransition(request)
          }}
          onCandidateChange={setCurrentCandidate}
          transitionModeEnabled={transitionModeEnabled}
          onTransitionModeEnabledChange={setTransitionModeEnabled}
          preferredNodeId={preferredNodeId}
          onClearPreferredNode={() => setPreferredNodeId(null)}
          />
        </div>

        <MindMapView onNavigateToEngine={onNavigateToEngine} />

        <main className="feature-grid">
          <section className="feature-card">
            <ChatDock />
          </section>
          <section className="feature-card">
            <ApiDemo />
          </section>
        </main>
          </>
        )}
      </div>
      <div className="app-frame" />
      <footer className="app-footer">
        <span>MindCastle</span>
        <span>{onboardingComplete ? 'Flow surfaces: next task · map · focus' : 'Choose a setup path to continue'}</span>
      </footer>
      {onboardingComplete && focusCandidate && <FocusMode candidate={focusCandidate} onExit={onExitFocusMode} />}
      {onboardingComplete && transition && <TransitionMode transition={transition} onComplete={() => setTransition(null)} />}
    </div>
  )
}

export default App
