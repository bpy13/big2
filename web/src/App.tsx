import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, fits, type ActionView, type Analysis, type Card, type Config, type GameView } from './api'
import { cardsText, completions, matchLegal, type SortMode } from './cards'
import { Played } from './components/Card'
import { GameOver } from './components/GameOver'
import { Hand } from './components/Hand'
import { History } from './components/History'
import { NewGameDialog, type NewGameChoice } from './components/NewGameDialog'
import { Seat } from './components/Seat'
import { StudyPanel, type StudyMode, type Verdict } from './components/StudyPanel'

/** Pause before each bot move (ms). Override with ?botDelay=0 in the URL. */
const BOT_DELAY_MS = (() => {
  const v = Number(new URLSearchParams(window.location.search).get('botDelay'))
  return Number.isFinite(v) && v >= 0 && new URLSearchParams(window.location.search).has('botDelay') ? v : 700
})()

/** Small typed wrapper around localStorage that never throws. */
function stored<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return v === null ? fallback : (JSON.parse(v) as T)
  } catch {
    return fallback
  }
}
function store(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    /* private mode etc. */
  }
}

/** Seats relative to you: you sit at the bottom; play passes to your right. */
function seatNames(seat: number): string[] {
  const names = ['', '', '', '']
  names[seat] = 'You'
  for (let k = 1; k < 4; k++) names[(seat + k) % 4] = `Bot ${k}`
  return names
}

/** Each player's latest action in the current trick. */
function lastInTrick(trick: ActionView[]): Map<number, ActionView> {
  const m = new Map<number, ActionView>()
  for (const a of trick) m.set(a.player, a)
  return m
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null)
  const [view, setView] = useState<GameView | null>(null)
  const [selected, setSelected] = useState<Card[]>([])
  const [sort, setSort] = useState<SortMode>(() => stored('sort', 'rank'))
  const [error, setError] = useState<string | null>(null)
  const [studyMode, setStudyMode] = useState<StudyMode>(() => stored('studyMode', 'live'))
  const [objective, setObjective] = useState<'score' | 'win'>('score')
  const [analysis, setAnalysis] = useState<{ key: string; data: Analysis } | null>(null)
  const [verdict, setVerdict] = useState<Verdict | null>(null)
  const [choice, setChoice] = useState<NewGameChoice | null>(() => stored('choice', null))
  const [dialog, setDialog] = useState<'new' | 'result' | null>('new')
  const [busy, setBusy] = useState(false)
  const stepping = useRef(false)

  useEffect(() => {
    api.config().then(setConfig, (e) => setError(`Cannot reach the server: ${e.message}`))
  }, [])
  useEffect(() => store('sort', sort), [sort])
  useEffect(() => store('studyMode', studyMode), [studyMode])

  const startGame = useCallback(async (c: NewGameChoice) => {
    setChoice(c)
    store('choice', c)
    setError(null)
    setSelected([])
    setVerdict(null)
    setAnalysis(null)
    try {
      setView(await api.newGame({ difficulty: c.difficulty, rules: c.rules, model: c.model }))
      api.config().then(setConfig, () => {}) // refresh which networks are loaded
      setDialog(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [])

  // Bots move one at a time, with a pause so you can follow.
  useEffect(() => {
    if (!view || view.over || view.your_turn) return
    const t = setTimeout(async () => {
      if (stepping.current) return
      stepping.current = true
      try {
        const r = await api.step(view.id)
        setView(r.view)
        if (r.view.over) setDialog('result')
      } catch (e) {
        setError((e as Error).message)
      } finally {
        stepping.current = false
      }
    }, BOT_DELAY_MS)
    return () => clearTimeout(t)
  }, [view])

  // Passing is your only legal move: the UI passes for you after a short pause.
  const forcedPass = !!view?.your_turn && !view.over && view.legal.length === 1 && view.legal[0].length === 0

  // Solver analysis for your turn (also computed silently in quiz mode, for the verdict).
  // `turnKey` changes once per human turn, so bot steps do not re-trigger the request.
  const gameId = view?.id
  const turnKey = view && view.your_turn ? `${view.id}:${view.history.length}:${objective}:${view.model}` : null
  useEffect(() => {
    if (!turnKey || !gameId || studyMode === 'off' || forcedPass) return
    let cancelled = false
    api.analyze(gameId, { objective, top: 40 }).then(
      (data) => !cancelled && setAnalysis({ key: turnKey, data }),
      (e) => !cancelled && setError((e as Error).message),
    )
    return () => {
      cancelled = true
    }
  }, [turnKey, gameId, objective, studyMode, forcedPass])

  const currentAnalysis = analysis && analysis.key === turnKey ? analysis.data : null
  const legalMatch = view ? matchLegal(selected, view.legal) : undefined
  const canPass = !!view?.your_turn && !forcedPass && view.legal.some((l) => l.length === 0)
  const playable = useMemo(() => (view?.your_turn ? completions(selected, view.legal) : null), [selected, view])

  const play = useCallback(
    async (cards: Card[], forced = false) => {
      if (!view || busy) return
      setBusy(true)
      setError(null)
      try {
        const next = await api.play(view.id, cards)
        if (forced) {
          // Keep the verdict on your last real decision.
        } else if (currentAnalysis && studyMode !== 'off') {
          const playedStats = currentAnalysis.moves.find((m) => m.cards.length === cards.length && m.cards.every((c) => cards.includes(c)))
          const net = currentAnalysis.network
          setVerdict({
            played: cards,
            best: currentAnalysis.moves[0],
            playedStats,
            network: net && net.moves.length > 0 ? { label: net.model.label, bot: net.model.kind === 'bot', cards: net.moves[0].cards } : null,
          })
        } else {
          setVerdict(null)
        }
        setSelected([])
        setView(next)
        if (next.over) setDialog('result')
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setBusy(false)
      }
    },
    [view, busy, currentAnalysis, studyMode],
  )

  // Loads a network on the server (only when the player confirms) and uses it for the hint.
  const loadModel = useCallback(
    async (id: string) => {
      if (!view) return
      try {
        await api.setModel(view.id, id)
      } catch (e) {
        setError((e as Error).message)
        throw e
      }
      setView((v) => (v && v.id === view.id ? { ...v, model: id } : v))
      setConfig((c) => (c ? { ...c, models: c.models.map((m) => (m.id === id ? { ...m, loaded: true } : m)) } : c))
      setChoice((c) => {
        const next = c ? { ...c, model: id } : c
        if (next) store('choice', next)
        return next
      })
    },
    [view],
  )

  useEffect(() => {
    if (!forcedPass || dialog || busy) return
    const t = setTimeout(() => play([], true), BOT_DELAY_MS)
    return () => clearTimeout(t)
  }, [forcedPass, dialog, busy, play])

  // Keyboard: Enter plays the selection, P passes, Escape clears.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!view?.your_turn || dialog) return
      if ((e.target as HTMLElement).tagName === 'SELECT') return
      if (e.key === 'Enter' && legalMatch && selected.length > 0) play(selected)
      else if ((e.key === 'p' || e.key === 'P') && canPass) play([])
      else if (e.key === 'Escape') setSelected([])
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [view, dialog, legalMatch, selected, canPass, play])

  if (!config) {
    return <div className="app">{error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>}</div>
  }

  const defaults: NewGameChoice = choice ?? { difficulty: 'hard', rules: config.defaults }
  const names = view ? seatNames(view.seat) : seatNames(0)
  const last = view ? lastInTrick(view.trick) : new Map()
  const rel = (k: number) => (view ? (view.seat + k) % 4 : k)

  let status = ''
  if (view && !view.over) {
    if (view.your_turn) {
      if (forcedPass && view.current_play) status = `Nothing beats ${cardsText(view.current_play.cards)} — passing`
      else if (view.is_opening) status = 'Your lead — include 3♦'
      else if (view.current_play) status = `Beat ${cardsText(view.current_play.cards)} or pass`
      else status = 'Your lead — play any combination'
    } else status = `${names[view.current_player ?? 0]} is thinking…`
  }

  const seatEl = (k: number, area: string) => {
    const p = rel(k)
    return (
      <div className={area}>
        {view && (
          <Seat
            name={names[p]}
            detail={difficultyLabel(choice?.difficulty)}
            count={view.hand_sizes[p]}
            isTurn={view.current_player === p}
            lockedOut={view.locked_out[p]}
            last={last.get(p)}
            thinking={!view.your_turn}
          />
        )}
      </div>
    )
  }

  return (
    <div className="app">
      <main className="table" aria-label="Card table">
        <header className="area-title table-title">
          <h1>Big Two</h1>
          {view && (
            <span className="meta">
              Deal #{view.seed} · {difficultyLabel(choice?.difficulty)} bots
            </span>
          )}
        </header>
        <div className="area-menu table-menu">
          {view?.over && dialog === null && (
            <button type="button" onClick={() => setDialog('result')}>
              Results
            </button>
          )}
          <button type="button" onClick={() => setDialog('new')}>
            New game
          </button>
        </div>
        {seatEl(2, 'area-top')}
        {seatEl(3, 'area-left')}
        {seatEl(1, 'area-right')}
        <div className="area-mid mid">
          <div className="label">
            {view?.current_play ? `On the table · ${names[view.current_play.player]}` : 'New trick'}
          </div>
          {view?.current_play ? <Played cards={view.current_play.cards} small={false} /> : <div style={{ height: 'var(--card-h)' }} />}
          <div className="status" role="status">
            {status}
          </div>
        </div>
        <div className="area-me me">
          {view && (
            <>
              <div className="seat-me-last">{last.get(view.seat) && <Played cards={last.get(view.seat)!.cards} />}</div>
              <Hand
                cards={view.hand}
                selected={selected}
                playable={playable}
                sort={sort}
                enabled={view.your_turn && !busy}
                onToggle={(c) => setSelected((s) => (s.includes(c) ? s.filter((x) => x !== c) : [...s, c]))}
              />
              <div className="actions">
                <button
                  type="button"
                  className="primary"
                  title="Play the selected cards (Enter)"
                  disabled={!view.your_turn || busy || selected.length === 0 || !legalMatch}
                  onClick={() => play(selected)}
                >
                  Play
                </button>
                <button type="button" title="Pass (P)" disabled={!canPass || busy} onClick={() => play([])}>
                  Pass
                </button>
                <button type="button" title="Clear the selection (Esc)" disabled={selected.length === 0} onClick={() => setSelected([])}>
                  Clear
                </button>
                <button type="button" onClick={() => setSort(sort === 'rank' ? 'suit' : 'rank')}>
                  Sort by {sort === 'rank' ? 'suit' : 'rank'}
                </button>
                <span className="hint">
                  {view.your_turn && selected.length > 0 && !legalMatch ? 'Not a legal play' : ''}
                </span>
              </div>
              {error && <p className="error" role="alert">{error}</p>}
            </>
          )}
        </div>
      </main>

      <aside className="side">
        <StudyPanel
          mode={studyMode}
          onMode={setStudyMode}
          objective={objective}
          onObjective={setObjective}
          analysis={currentAnalysis ? { ...currentAnalysis, moves: currentAnalysis.moves.slice(0, 3) } : null}
          loading={!currentAnalysis}
          yourTurn={!!view?.your_turn && !forcedPass}
          onPick={(cards) => setSelected(cards)}
          verdict={verdict}
          models={view ? config.models.filter((m) => fits(m, view.rules)) : []}
          model={view?.model ?? null}
          onLoadModel={loadModel}
        />
        <section className="panel" aria-label="History">
          <h2>History</h2>
          {view ? <History view={view} names={names} /> : <span className="muted">No game yet.</span>}
        </section>
      </aside>

      {dialog === 'new' && (
        <NewGameDialog
          config={config}
          initial={defaults}
          onStart={startGame}
          onCancel={view ? () => setDialog(view.over ? 'result' : null) : undefined}
        />
      )}
      {dialog === 'result' && view?.over && (
        <GameOver view={view} names={names} onNewGame={() => setDialog('new')} onClose={() => setDialog(null)} />
      )}
    </div>
  )
}

function difficultyLabel(d: string | undefined): string {
  if (!d) return ''
  if (d === 'rl') return 'RL'
  if (d === 'rl_search') return 'RL + search'
  return d[0].toUpperCase() + d.slice(1)
}
