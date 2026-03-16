import { useState, useCallback } from 'react'
import { categories, t, type Lang } from './data'
import LanguageScreen from './components/LanguageScreen'
import WelcomeScreen from './components/WelcomeScreen'
import QuestionnaireScreen from './components/QuestionnaireScreen'
import ResultsScreen from './components/ResultsScreen'

export type Screen = 'language' | 'welcome' | 'questionnaire' | 'results'

export interface Ratings {
  [categoryIndex: number]: {
    [itemIndex: number]: number
  }
}

export interface Revealed {
  [categoryIndex: number]: {
    [itemIndex: number]: boolean
  }
}

function App() {
  const [lang, setLang] = useState<Lang>('en')
  const [screen, setScreen] = useState<Screen>('language')
  const [ratings, setRatings] = useState<Ratings>({})
  const [revealed, setRevealed] = useState<Revealed>({})

  const setRating = useCallback((catIdx: number, itemIdx: number, value: number) => {
    setRatings(prev => ({
      ...prev,
      [catIdx]: { ...(prev[catIdx] || {}), [itemIdx]: value },
    }))
  }, [])

  const revealItem = useCallback((catIdx: number, itemIdx: number) => {
    setRevealed(prev => ({
      ...prev,
      [catIdx]: { ...(prev[catIdx] || {}), [itemIdx]: true },
    }))
  }, [])

  const revealAll = useCallback(() => {
    const allRevealed: Revealed = {}
    categories.forEach((cat, ci) => {
      allRevealed[ci] = {}
      cat.items.forEach((_, ii) => {
        allRevealed[ci][ii] = true
      })
    })
    setRevealed(allRevealed)
  }, [])

  const resetApp = useCallback(() => {
    setRatings({})
    setRevealed({})
    setScreen('language')
  }, [])

  const toggleLang = useCallback(() => {
    setLang(l => (l === 'en' ? 'hi' : 'en'))
  }, [])

  const tx = t[lang]

  const allRevealedCheck = categories.every((cat, ci) =>
    cat.items.every((_, ii) => revealed[ci]?.[ii])
  )

  return (
    <div className="app">
      {screen !== 'language' && (
        <button className="lang-toggle" onClick={toggleLang} aria-label="Toggle language">
          {tx.langToggle}
        </button>
      )}

      {screen === 'language' && (
        <LanguageScreen
          onSelect={(l) => { setLang(l); setScreen('welcome') }}
          tx={tx}
          lang={lang}
        />
      )}
      {screen === 'welcome' && (
        <WelcomeScreen
          tx={tx}
          onStart={() => setScreen('questionnaire')}
        />
      )}
      {screen === 'questionnaire' && (
        <QuestionnaireScreen
          lang={lang}
          tx={tx}
          ratings={ratings}
          revealed={revealed}
          setRating={setRating}
          revealItem={revealItem}
          revealAll={revealAll}
          allRevealed={allRevealedCheck}
          onViewResults={() => setScreen('results')}
        />
      )}
      {screen === 'results' && (
        <ResultsScreen
          lang={lang}
          tx={tx}
          ratings={ratings}
          onStartOver={resetApp}
        />
      )}
    </div>
  )
}

export default App
