# TinyNet UI

A chat-first mind web built with React, TypeScript, and Vite.

## Quickstart

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Development

The development server starts at `http://localhost:5173` (with the optional `tinynet-api` on port 8000 for classify, nodes, and theme preference sync).

## Themes and motion

Design rationale and motion specs live in `THEME ideas/` at this package root. Runtime tokens are in `src/theme/tokens.ts`; `ThemeProvider` wraps the app in `main.tsx` and injects CSS variables consumed by `styles.css`. Changing the theme in the top bar updates `localStorage` and, when the API is reachable, `PATCH /users/me/preferences`.

## Project Structure

```
tinynet-ui/
├── THEME ideas/        # Design docs + reference HTML (source for tokens / motion)
├── src/
│   ├── components/     # React components
│   ├── theme/          # ThemeProvider, palette tokens
│   ├── api.ts          # Backend client (nodes, classify, preferences)
│   ├── App.tsx         # Main app component
│   ├── main.tsx        # App entry (ThemeProvider root)
│   └── styles.css      # Global styles (CSS variables from theme)
├── public/             # Public assets
├── index.html          # HTML template
└── package.json        # Dependencies
```

## Tech Stack

- **Frontend**: React 18 + TypeScript
- **Build Tool**: Vite
- **State Management**: Zustand
- **Styling**: CSS (can switch to Tailwind later)
- **Graph Visualization**: react-force-graph-2d
- **Utilities**: dayjs, classnames
