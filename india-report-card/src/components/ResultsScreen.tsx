import { useState } from 'react'
import { categories, type Lang, type t as TType } from '../data'
import type { Ratings } from '../App'
import { scoreColor } from './ScoreSlider'

interface Props {
  lang: Lang
  tx: typeof TType['en']
  ratings: Ratings
  onStartOver: () => void
}

function avg(nums: number[]) {
  if (!nums.length) return 0
  return nums.reduce((a, b) => a + b, 0) / nums.length
}

export default function ResultsScreen({ lang, tx, ratings, onStartOver }: Props) {
  const [copied, setCopied] = useState(false)

  const catResults = categories.map((cat, ci) => {
    const userScores = cat.items.map((_, ii) => ratings[ci]?.[ii] ?? 5)
    const dataScores = cat.items.map(item => item.statScore)
    const userAvg = avg(userScores)
    const dataAvg = avg(dataScores)
    const gap = userAvg - dataAvg
    return { cat, userAvg, dataAvg, gap }
  })

  const overallUser = avg(catResults.map(r => r.userAvg))
  const overallData = avg(catResults.map(r => r.dataAvg))
  const overallGap = overallUser - overallData

  function getVerdict(gap: number) {
    if (gap > 1.5) return tx.overconfident
    if (gap < -1.5) return tx.underestimated
    return tx.aligned
  }

  function verdictClass(gap: number) {
    if (gap > 1.5) return 'verdict--red'
    if (gap < -1.5) return 'verdict--green'
    return 'verdict--yellow'
  }

  const maxBar = 10

  async function handleShare() {
    const url = window.location.href
    const lines = catResults.map(r =>
      `${r.cat.tier[lang]}: ${r.userAvg.toFixed(1)}/10 vs ${r.dataAvg.toFixed(1)}/10`
    )

    const text = lang === 'en'
      ? `India Report Card — My Ratings vs Reality\n${lines.join('\n')}\nTry it yourself: ${url}`
      : `भारत रिपोर्ट कार्ड — मेरी रेटिंग बनाम हकीकत\n${lines.join('\n')}\nखुद आज़माएं: ${url}`

    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2500)
    } catch {
      // fallback
      const ta = document.createElement('textarea')
      ta.value = text
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
      setCopied(true)
      setTimeout(() => setCopied(false), 2500)
    }
  }

  return (
    <div className="results">
      <header className="results__header">
        <div className="results__badge">📋</div>
        <h1 className="results__title">{tx.resultsTitle}</h1>
        <p className="results__subtitle">{tx.resultsSubtitle}</p>

        <div className="overall-score">
          <div className="overall-score__pair">
            <span className="overall-score__label">{tx.yourAvg}</span>
            <span
              className="overall-score__value"
              style={{ color: scoreColor(Math.round(overallUser)) }}
            >
              {overallUser.toFixed(1)}
            </span>
          </div>
          <div className="overall-score__sep">vs</div>
          <div className="overall-score__pair">
            <span className="overall-score__label">{tx.dataAvg}</span>
            <span
              className="overall-score__value"
              style={{ color: scoreColor(Math.round(overallData)) }}
            >
              {overallData.toFixed(1)}
            </span>
          </div>
          <div className={`overall-verdict ${verdictClass(overallGap)}`}>
            {getVerdict(overallGap)}
          </div>
        </div>
      </header>

      {/* Bar chart */}
      <section className="chart-section">
        <h2 className="section-heading">
          {lang === 'en' ? 'Category Comparison' : 'श्रेणी तुलना'}
        </h2>
        <div className="bar-chart">
          {catResults.map((r, i) => (
            <div key={i} className="bar-row">
              <div className="bar-row__label">{r.cat.tier[lang]}</div>
              <div className="bar-row__bars">
                <div className="bar-group">
                  <span className="bar-group__tag bar-group__tag--user">{tx.yourAvg}</span>
                  <div className="bar-track">
                    <div
                      className="bar bar--user"
                      style={{
                        width: `${(r.userAvg / maxBar) * 100}%`,
                        background: scoreColor(Math.round(r.userAvg)),
                      }}
                    />
                    <span className="bar-value">{r.userAvg.toFixed(1)}</span>
                  </div>
                </div>
                <div className="bar-group">
                  <span className="bar-group__tag bar-group__tag--data">{tx.dataAvg}</span>
                  <div className="bar-track">
                    <div
                      className="bar bar--data"
                      style={{
                        width: `${(r.dataAvg / maxBar) * 100}%`,
                        background: scoreColor(Math.round(r.dataAvg)),
                        opacity: 0.75,
                      }}
                    />
                    <span className="bar-value">{r.dataAvg.toFixed(1)}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Summary table */}
      <section className="table-section">
        <h2 className="section-heading">
          {lang === 'en' ? 'Full Summary' : 'पूरा सारांश'}
        </h2>
        <div className="summary-table-wrap">
          <table className="summary-table">
            <thead>
              <tr>
                <th>{tx.category}</th>
                <th>{tx.yourAvg}</th>
                <th>{tx.dataAvg}</th>
                <th>{tx.gap}</th>
                <th>{tx.verdict}</th>
              </tr>
            </thead>
            <tbody>
              {catResults.map((r, i) => (
                <tr key={i}>
                  <td className="table-cat">{r.cat.tier[lang]}</td>
                  <td>
                    <span style={{ color: scoreColor(Math.round(r.userAvg)), fontFamily: 'monospace' }}>
                      {r.userAvg.toFixed(1)}
                    </span>
                  </td>
                  <td>
                    <span style={{ color: scoreColor(Math.round(r.dataAvg)), fontFamily: 'monospace' }}>
                      {r.dataAvg.toFixed(1)}
                    </span>
                  </td>
                  <td>
                    <span
                      style={{
                        color: r.gap > 0 ? '#f97316' : r.gap < 0 ? '#22c55e' : '#eab308',
                        fontFamily: 'monospace',
                      }}
                    >
                      {r.gap > 0 ? '+' : ''}{r.gap.toFixed(1)}
                    </span>
                  </td>
                  <td>
                    <span className={`verdict-chip ${verdictClass(r.gap)}`}>
                      {getVerdict(r.gap)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="results__actions">
        <button className="btn-primary" onClick={handleShare}>
          {copied ? tx.copiedMsg : tx.shareBtn}
        </button>
        <button className="btn-outline" onClick={onStartOver}>
          {tx.startOver}
        </button>
      </div>
    </div>
  )
}
