import { defineConfig } from '@playwright/test'

// End-to-end test: starts the real server (API + built UI) and plays a game in Chromium.
// Run `npm run build` first (or `make test-web`), inside the `big2` conda env.
export default defineConfig({
  testDir: 'tests',
  timeout: 180_000,
  use: { baseURL: 'http://127.0.0.1:8766', viewport: { width: 1360, height: 900 } },
  webServer: {
    command: 'big2 serve --port 8766',
    url: 'http://127.0.0.1:8766/api/config',
    reuseExistingServer: false,
    env: { POKER_GAME_LOG_DIR: 'test-results/games' },
  },
})
