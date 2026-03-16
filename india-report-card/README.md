# 🇮🇳 India Democracy Report Card | भारत रिपोर्ट कार्ड

An interactive bilingual (English/Hindi) static React web app that lets you rate India's performance on 22 key indicators across 5 categories — then compares your perception with real data from sources like NITI Aayog, WHO, V-Dem, and more.

**Live demo:** https://urhit45.github.io/india-report-card

---

## Features

- **Bilingual** — Full English/Hindi support, toggle at any time
- **5 categories, 22 indicators** with real 2023–24 data
- **Interactive 1–10 sliders** with live color feedback
- **Reveal data** per indicator or all at once
- **Results screen** with bar chart and summary table
- **Share button** copies a formatted text summary to clipboard
- **Fully static** — no backend, no database, deploys to GitHub Pages

## Tech Stack

- React 18 + TypeScript
- Vite
- Plain CSS (dark theme, Playfair Display + Inter fonts)
- gh-pages for deployment

## Local Development

```bash
npm install
npm run dev
```

## Deploy to GitHub Pages

1. Update `homepage` in `package.json` with your GitHub username:
   ```json
   "homepage": "https://YOUR-USERNAME.github.io/india-report-card"
   ```
2. Update `base` in `vite.config.ts` to match:
   ```ts
   base: '/india-report-card/',
   ```
3. Run:
   ```bash
   npm run deploy
   ```

## Categories & Sources

| Category | Key Sources |
|---|---|
| 🧱 Bare Necessities | NITI Aayog, WHO/UNICEF, Global Hunger Index, NFHS-5 |
| 🏫 Rights & Institutions | RSF, World Justice Project, V-Dem, Transparency International, Freedom House |
| 📚 Education & Health | UNDP HDI, MoHFW, WEF Gender Gap, World Bank |
| 💰 Economy & Opportunity | World Inequality Lab, CMIE, IMF, CAG |
| 🌐 Global Standing & Future | Yale EPI, NPCI, ISRO, Access Now |
