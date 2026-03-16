interface Props {
  value: number
  onChange: (v: number) => void
  disabled?: boolean
}

export function scoreColor(v: number): string {
  if (v <= 2) return '#ef4444'
  if (v <= 4) return '#f97316'
  if (v <= 6) return '#eab308'
  if (v <= 8) return '#22c55e'
  return '#06b6d4'
}

export function scoreLabel(v: number): string {
  if (v <= 2) return 'Very Poor'
  if (v <= 4) return 'Poor'
  if (v <= 6) return 'Moderate'
  if (v <= 8) return 'Good'
  return 'Excellent'
}

export function scoreLabelHi(v: number): string {
  if (v <= 2) return 'बहुत खराब'
  if (v <= 4) return 'खराब'
  if (v <= 6) return 'ठीक-ठाक'
  if (v <= 8) return 'अच्छा'
  return 'उत्कृष्ट'
}

export default function ScoreSlider({ value, onChange, disabled }: Props) {
  const color = scoreColor(value)
  const pct = ((value - 1) / 9) * 100

  return (
    <div className="slider-wrap">
      <input
        type="range"
        min={1}
        max={10}
        step={1}
        value={value}
        disabled={disabled}
        onChange={e => onChange(Number(e.target.value))}
        className="slider"
        style={{
          '--thumb-color': color,
          '--track-fill': `${pct}%`,
          '--fill-color': color,
        } as React.CSSProperties}
      />
      <div className="slider-ticks">
        {Array.from({ length: 10 }, (_, i) => (
          <span
            key={i}
            className="tick"
            style={{ color: i + 1 === value ? color : undefined }}
          >
            {i + 1}
          </span>
        ))}
      </div>
    </div>
  )
}
