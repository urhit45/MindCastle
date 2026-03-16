import type { t as TType } from '../data'

interface Props {
  tx: typeof TType['en']
  onStart: () => void
}

export default function WelcomeScreen({ tx, onStart }: Props) {
  return (
    <div className="welcome-screen">
      <div className="welcome-screen__inner">
        <div className="welcome-screen__badge">🗳️</div>
        <h1 className="welcome-screen__title">{tx.welcomeTitle}</h1>
        <p className="welcome-screen__subtitle">{tx.welcomeSubtitle}</p>

        <ol className="welcome-screen__steps">
          {tx.instructions.map((step, i) => (
            <li key={i} className="welcome-screen__step">
              <span className="step-num">{i + 1}</span>
              <span>{step}</span>
            </li>
          ))}
        </ol>

        <button className="btn-primary" onClick={onStart}>
          {tx.startBtn}
        </button>
      </div>
    </div>
  )
}
