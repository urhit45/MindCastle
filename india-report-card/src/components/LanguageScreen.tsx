import type { Lang } from '../data'
import type { t as TType } from '../data'

interface Props {
  onSelect: (lang: Lang) => void
  tx: typeof TType['en']
  lang: Lang
}

export default function LanguageScreen({ onSelect }: Props) {
  return (
    <div className="lang-screen">
      <div className="lang-screen__flag-bar">
        <span className="flag-saffron" />
        <span className="flag-white" />
        <span className="flag-green" />
      </div>

      <div className="lang-screen__content">
        <div className="lang-screen__emblem">🇮🇳</div>
        <h1 className="lang-screen__title">
          <span className="tricolor-text">India Report Card</span>
        </h1>
        <p className="lang-screen__title-hi">भारत रिपोर्ट कार्ड</p>
        <p className="lang-screen__subtitle">
          Choose your language&nbsp;&nbsp;|&nbsp;&nbsp;अपनी भाषा चुनें
        </p>

        <div className="lang-screen__btns">
          <button
            className="lang-btn lang-btn--en"
            onClick={() => onSelect('en')}
          >
            <span className="lang-btn__label">English</span>
            <span className="lang-btn__sub">Continue in English</span>
          </button>
          <button
            className="lang-btn lang-btn--hi"
            onClick={() => onSelect('hi')}
          >
            <span className="lang-btn__label">हिंदी</span>
            <span className="lang-btn__sub">हिंदी में जारी रखें</span>
          </button>
        </div>
      </div>

      <div className="lang-screen__flag-bar lang-screen__flag-bar--bottom">
        <span className="flag-saffron" />
        <span className="flag-white" />
        <span className="flag-green" />
      </div>
    </div>
  )
}
