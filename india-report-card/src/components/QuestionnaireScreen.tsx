import { categories, type Lang, type t as TType } from '../data'
import type { Ratings, Revealed } from '../App'
import ScoreSlider, { scoreColor, scoreLabel, scoreLabelHi } from './ScoreSlider'

interface Props {
  lang: Lang
  tx: typeof TType['en']
  ratings: Ratings
  revealed: Revealed
  setRating: (ci: number, ii: number, v: number) => void
  revealItem: (ci: number, ii: number) => void
  revealAll: () => void
  allRevealed: boolean
  onViewResults: () => void
}

function avg(nums: number[]) {
  if (!nums.length) return 0
  return nums.reduce((a, b) => a + b, 0) / nums.length
}

export default function QuestionnaireScreen({
  lang, tx, ratings, revealed, setRating, revealItem, revealAll, allRevealed, onViewResults,
}: Props) {
  return (
    <div className="questionnaire">
      <header className="questionnaire__header">
        <h1 className="questionnaire__title">{tx.appTitle}</h1>
        <p className="questionnaire__subtitle">
          {lang === 'en' ? 'Rate each indicator 1–10' : 'प्रत्येक संकेतक को 1-10 पर रेट करें'}
        </p>
      </header>

      {categories.map((cat, ci) => {
        const catRatings = cat.items.map((_, ii) => ratings[ci]?.[ii] ?? 5)
        const catData = cat.items.map(item => item.statScore)
        const userCatAvg = avg(catRatings)
        const dataCatAvg = avg(catData)
        const revealedCount = cat.items.filter((_, ii) => revealed[ci]?.[ii]).length

        return (
          <section key={ci} className="category">
            <div className="category__header">
              <h2 className="category__title">{cat.tier[lang]}</h2>
              <div className="category__avgs">
                <span className="avg-chip avg-chip--user">
                  {tx.yourAvg}: <strong style={{ color: scoreColor(Math.round(userCatAvg)) }}>{userCatAvg.toFixed(1)}</strong>
                </span>
                {revealedCount > 0 && (
                  <span className="avg-chip avg-chip--data">
                    {tx.dataAvg}: <strong style={{ color: scoreColor(Math.round(dataCatAvg)) }}>{dataCatAvg.toFixed(1)}</strong>
                  </span>
                )}
              </div>
            </div>

            <div className="items">
              {cat.items.map((item, ii) => {
                const rating = ratings[ci]?.[ii] ?? 5
                const isRevealed = revealed[ci]?.[ii] ?? false
                const diff = rating - item.statScore
                const absDiff = Math.abs(diff)

                return (
                  <div
                    key={ii}
                    className={`item ${isRevealed ? 'item--revealed' : ''}`}
                  >
                    <div className="item__top">
                      <h3 className="item__label">{item.label[lang]}</h3>
                      <div className="item__rating-display">
                        <span
                          className="score-badge"
                          style={{ color: scoreColor(rating), borderColor: scoreColor(rating) }}
                        >
                          {rating}/10
                        </span>
                        <span className="score-label" style={{ color: scoreColor(rating) }}>
                          {lang === 'en' ? scoreLabel(rating) : scoreLabelHi(rating)}
                        </span>
                      </div>
                    </div>

                    <ScoreSlider
                      value={rating}
                      onChange={v => setRating(ci, ii, v)}
                      disabled={isRevealed}
                    />

                    {!isRevealed ? (
                      <button
                        className="reveal-btn"
                        onClick={() => revealItem(ci, ii)}
                      >
                        {tx.revealBtn}
                      </button>
                    ) : (
                      <div className="item__reveal">
                        <div className="item__stat-row">
                          <div className="stat-text">
                            <span className="stat-icon">📊</span>
                            <span>{item.stat[lang]}</span>
                          </div>
                          <div className="stat-scores">
                            <div className="stat-score-pair">
                              <span className="stat-score-label">{tx.yourRating}</span>
                              <span
                                className="score-badge"
                                style={{ color: scoreColor(rating), borderColor: scoreColor(rating) }}
                              >
                                {rating}/10
                              </span>
                            </div>
                            <div className="stat-score-sep">→</div>
                            <div className="stat-score-pair">
                              <span className="stat-score-label">{tx.dataScore}</span>
                              <span
                                className="score-badge"
                                style={{ color: scoreColor(item.statScore), borderColor: scoreColor(item.statScore) }}
                              >
                                {item.statScore}/10
                              </span>
                            </div>
                          </div>
                        </div>

                        {absDiff >= 2 && (
                          <div
                            className={`gap-pill ${diff > 0 ? 'gap-pill--over' : 'gap-pill--under'}`}
                          >
                            {diff > 0
                              ? (lang === 'en' ? `↑ Overestimated by ${absDiff}` : `↑ ${absDiff} ज़्यादा आंका`)
                              : (lang === 'en' ? `↓ Underestimated by ${absDiff}` : `↓ ${absDiff} कम आंका`)
                            }
                          </div>
                        )}

                        <div className="source-tag">
                          {tx.source}: <em>{item.source}</em>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </section>
        )
      })}

      <div className="questionnaire__footer">
        {allRevealed ? (
          <button className="btn-primary btn-primary--large" onClick={onViewResults}>
            {tx.viewResults}
          </button>
        ) : (
          <button className="btn-secondary" onClick={revealAll}>
            {tx.revealAll}
          </button>
        )}
      </div>
    </div>
  )
}
