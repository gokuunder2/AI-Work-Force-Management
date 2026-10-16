# Workforce AI — React Implementation

A React + Vite + Tailwind CSS implementation of the "Kinetic Enterprise" Workforce AI
design system (Home, Employee Data, and Tasks screens), converted from the Stitch
design export.

## Getting started

```bash
npm install
npm run dev
```

Then open the printed local URL (usually http://localhost:5173).

## Build for production

```bash
npm run build
npm run preview
```

## Project structure

```
src/
  components/
    TopNav.jsx       Top navigation bar (search, links, icons, avatar)
    SideNav.jsx       Desktop admin console sidebar (Directory / Recruitment active states)
    BottomNav.jsx     Mobile bottom tab bar
  pages/
    Home.jsx          Welcome / bento grid / quick stats
    EmployeeData.jsx   Upload card, directory table, AI Prediction Center, profile panel
    Tasks.jsx          Task list, AI "Top Recommend" matches, Bulk Allocation matrix
  data.js              Mock employees / tasks / recommendations
  App.jsx              Route definitions (react-router-dom)
  main.jsx             Entry point
  index.css            Tailwind directives + Material Symbols + inner-glow helper
tailwind.config.js      Full "Kinetic Enterprise" color/type/spacing tokens from DESIGN.md
```

## Notes

- Colors, type scale, radii, and spacing tokens are wired 1:1 from the design's
  `DESIGN.md` into `tailwind.config.js` (e.g. `bg-surface-container`, `text-primary`,
  `font-headline-lg`, `rounded-xl`, `gap-gutter`, etc.), so class names match the
  original Stitch HTML almost exactly.
- The Tasks page's three-panel flow (Task List / Top Recommend / Bulk Allocation) is
  implemented with React state instead of manual DOM class toggling.
- The Employee Data directory table drives the "Profile Details" panel via state —
  clicking a row updates the panel and briefly highlights it.
- Material Symbols and Google Fonts (Inter, JetBrains Mono) are loaded via CDN links
  in `index.html`.
