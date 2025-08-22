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

The development server will start at `http://localhost:5173` and show "Hello TinyNet" in the root div.

## Project Structure

```
tinynet-ui/
├── src/
│   ├── components/     # React components
│   ├── api/           # API integration
│   ├── assets/        # Static assets
│   ├── App.tsx        # Main app component
│   ├── main.tsx       # App entry point
│   └── styles.css     # Global styles
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
