import { useState, type CSSProperties } from 'react'
import type { Analysis, Card, ModelInfo, MoveStats, NetworkMove, NetworkView } from '../api'
import { cardsText, KIND_LABEL } from '../cards'

export type StudyMode = 'off' | 'live' | 'quiz'

export interface Verdict {
  played: Card[]
  best: MoveStats
  playedStats: MoveStats | undefined
  /** The network's first choice, if a network gave a hint. */
  network: { label: string; bot: boolean; cards: Card[] } | null
}

interface Props {
  mode: StudyMode
  onMode: (m: StudyMode) => void
  objective: 'score' | 'win'
  onObjective: (o: 'score' | 'win') => void
  analysis: Analysis | null
  loading: boolean
  yourTurn: boolean
  onPick: (cards: Card[]) => void
  verdict: Verdict | null
  /** Networks to choose from, and the one this game's hint uses. */
  models: ModelInfo[]
  model: string | null
  onLoadModel: (id: string) => Promise<void>
}

function pct(x: number) {
  return `${(100 * x).toFixed(1)}%`
}

function signed(x: number) {
  return `${x >= 0 ? '+' : ''}${x.toFixed(1)}`
}

function MoveRow({ m, i, onPick }: { m: MoveStats; i: number; onPick?: () => void }) {
  return (
    <button type="button" className={`move${i === 0 ? ' best' : ''}`} onClick={onPick} data-testid="study-move">
      <span className="rank">{i + 1}</span>
      <span className="what">
        {cardsText(m.cards)} {m.kind !== 'pass' && <span className="kind">{KIND_LABEL[m.kind] ?? m.kind}</span>}
      </span>
      <span className="stats">
        <span>Win rate</span>
        <span className="bar" title={`95% interval ±${pct(m.win_ci95)}`}>
          <span style={{ width: pct(m.win_rate) }} />
          <span
            className="ci"
            style={{ left: pct(Math.max(0, m.win_rate - m.win_ci95)), width: pct(2 * m.win_ci95) }}
          />
        </span>
        <span className="v">
          {pct(m.win_rate)} <span className="muted">±{(100 * m.win_ci95).toFixed(1)}</span>
        </span>
        <span>Exp. score</span>
        <span />
        <span className="v">
          {signed(m.mean_score)} <span className="muted">±{m.score_ci95.toFixed(1)}</span>
        </span>
      </span>
    </button>
  )
}

/** Study mode: the solver's top 3 moves for your turn, or a verdict on your last move. */
export function StudyPanel(p: Props) {
  return (
    <section className="panel study" aria-label="Study mode">
      <h2>
        <span className="grow">Study mode</span>
        <span className="toggle" role="group" aria-label="Study mode setting">
          {(['off', 'live', 'quiz'] as StudyMode[]).map((m) => (
            <button key={m} type="button" className={p.mode === m ? 'on' : ''} onClick={() => p.onMode(m)}>
              {m === 'off' ? 'Off' : m === 'live' ? 'Hints' : 'Quiz'}
            </button>
          ))}
        </span>
      </h2>

      {p.mode === 'off' && (
        <p className="muted">
          Turn on <b>Hints</b> to see the solver's top 3 moves on each turn, or <b>Quiz</b> to choose first and
          compare with the solver afterwards.
        </p>
      )}

      {p.mode !== 'off' && (
        <>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 10 }}>
            <span className="muted">Optimize</span>
            <span className="toggle" role="group" aria-label="Objective">
              <button
                type="button"
                className={p.objective === 'score' ? 'on' : ''}
                onClick={() => p.onObjective('score')}
              >
                Score
              </button>
              <button type="button" className={p.objective === 'win' ? 'on' : ''} onClick={() => p.onObjective('win')}>
                Win
              </button>
            </span>
          </div>

          {p.models.length > 0 && (
            <ModelPicker key={p.model ?? ''} models={p.models} current={p.model} onLoad={p.onLoadModel} />
          )}

          {p.yourTurn && p.mode === 'live' && (
            <>
              {p.loading && !p.analysis && <p className="muted">Thinking…</p>}
              {p.analysis && !p.analysis.network && (
                <div className="moves">
                  {p.analysis.moves.map((m, i) => (
                    <MoveRow key={m.cards.join()} m={m} i={i} onPick={() => p.onPick(m.cards)} />
                  ))}
                </div>
              )}
              {p.analysis?.network && (
                <Compare solver={p.analysis.moves} net={p.analysis.network} objective={p.analysis.objective} onPick={p.onPick} />
              )}
            </>
          )}
          {p.yourTurn && p.mode === 'quiz' && !p.verdict && (
            <p className="muted">Make your move — the solver's opinion appears afterwards.</p>
          )}
          {!p.yourTurn && !p.verdict && <p className="muted">Waiting for your turn…</p>}
          {p.verdict && <VerdictBox v={p.verdict} />}
          {p.yourTurn && p.mode === 'live' && p.analysis && (
            <p className="small-print">
              {p.analysis.simulations.toLocaleString()} simulated games in {p.analysis.seconds.toFixed(2)}s.
              Ranked by {p.analysis.objective === 'score' ? 'expected score' : 'win rate'}; click a move to
              select it. ± is a 95% interval from simulation noise only — the solver assumes opponents
              play like its rollout bot.
              {p.analysis.network &&
                ' The network\'s numbers are its own predictions from one forward pass (no interval); the same move has the same colour in both columns.'}
            </p>
          )}
        </>
      )}
    </section>
  )
}

/** Picks what the hint compares the solver with. Choosing a network only stages it;
 * "Load" loads it on the server. */
function ModelPicker(props: { models: ModelInfo[]; current: string | null; onLoad: (id: string) => Promise<void> }) {
  const [chosen, setChosen] = useState(props.current ?? '')
  const [state, setState] = useState<'idle' | 'loading' | 'error'>('idle')
  const staged = props.models.find((m) => m.id === chosen)
  const pending = chosen !== (props.current ?? '')
  return (
    <div className="model-picker">
      <label htmlFor="hint-model" className="muted">
        Network
      </label>
      <select
        id="hint-model"
        value={chosen}
        disabled={state === 'loading'}
        onChange={(e) => {
          setChosen(e.target.value)
          setState('idle')
        }}
      >
        {props.current === null && <option value="">None (no network for these rules)</option>}
        {props.models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.label}
          </option>
        ))}
      </select>
      {pending && (
        <button
          type="button"
          className="primary"
          disabled={state === 'loading'}
          onClick={async () => {
            setState('loading')
            try {
              await props.onLoad(chosen)
              setState('idle')
            } catch {
              setState('error')
            }
          }}
        >
          {state === 'loading' ? 'Loading…' : staged?.loaded ? 'Use' : 'Load'}
        </button>
      )}
      {staged && <span className="help">{staged.note}{state === 'error' && ' — could not load it.'}</span>}
    </div>
  )
}

/** One stable colour per distinct move, so the same move matches across the columns. */
function moveColours(solver: MoveStats[], net: NetworkMove[]): Map<string, number> {
  const colours = new Map<string, number>()
  for (const m of [...solver, ...net]) {
    const k = m.cards.join()
    if (!colours.has(k)) colours.set(k, colours.size)
  }
  return colours
}

function moveLabel(m: { cards: Card[]; kind: string }) {
  return (
    <span className="what">
      {cardsText(m.cards)} {m.kind !== 'pass' && <span className="kind">{KIND_LABEL[m.kind] ?? m.kind}</span>}
    </span>
  )
}

/** The solver's and the network's top moves side by side, with a verdict on whether they agree. */
function Compare(props: {
  solver: MoveStats[]
  net: NetworkView
  objective: 'score' | 'win'
  onPick: (c: Card[]) => void
}) {
  const { solver, net } = props
  const colours = moveColours(solver, net.moves)
  const colour = (cards: Card[]) => ({ '--mc': `var(--m${(colours.get(cards.join()) ?? 0) % 6})` }) as CSSProperties
  const pick = net.pick
  const agrees = pick.solver_rank === 1
  return (
    <div data-testid="compare">
      <div className="compare">
        <div className="col">
          <h3>Solver</h3>
          {solver.map((m, i) => (
            <button
              key={m.cards.join()}
              type="button"
              className="cmp-move"
              style={colour(m.cards)}
              onClick={() => props.onPick(m.cards)}
              data-testid="study-move"
              title={`Win ${pct(m.win_rate)} ±${pct(m.win_ci95)} · score ${signed(m.mean_score)} ±${m.score_ci95.toFixed(1)} · ${pct(m.share)} of simulations`}
            >
              <span className="rank">{i + 1}</span>
              {moveLabel(m)}
              <span className="nums">
                win {pct(m.win_rate)} · {signed(m.mean_score)}
              </span>
            </button>
          ))}
        </div>
        <div className="col">
          <h3 title={net.model.note}>{net.model.label}</h3>
          {net.moves.map((m, i) => (
            <button
              key={m.cards.join()}
              type="button"
              className="cmp-move"
              style={colour(m.cards)}
              onClick={() => props.onPick(m.cards)}
              data-testid="network-move"
            >
              <span className="rank">{i + 1}</span>
              {moveLabel(m)}
              <span className="nums">
                {m.prob !== null
                  ? `probability ${pct(m.prob)}`
                  : m.mean_score !== null && m.win_rate !== null
                    ? `win ${pct(m.win_rate)} · ${signed(m.mean_score)}`
                    : 'its move'}
              </span>
            </button>
          ))}
        </div>
      </div>
      <div className={`verdict${agrees ? '' : ' off'}`} data-testid="network-pick" style={{ marginTop: 10 }}>
        {agrees ? (
          <>
            <b>Agree</b> — both pick {cardsText(pick.cards)}.
          </>
        ) : (
          <>
            <b>Disagree</b> — the solver ranks the {net.model.kind === 'bot' ? "bot's" : "network's"} pick{' '}
            <button type="button" className="linklike" onClick={() => props.onPick(pick.cards)}>
              {cardsText(pick.cards)}
            </button>{' '}
            {pick.solver_rank !== null && pick.loss !== null ? (
              <>
                #{pick.solver_rank},{' '}
                {props.objective === 'score'
                  ? `${pick.loss.toFixed(1)} points per game worse than its best`
                  : `${(100 * pick.loss).toFixed(1)} percentage points less likely to win`}
                {pick.solver && ` (±${props.objective === 'score' ? pick.solver.score_ci95.toFixed(1) : pct(pick.solver.win_ci95)})`}.
              </>
            ) : (
              'outside its list.'
            )}
          </>
        )}
      </div>
    </div>
  )
}

function VerdictBox({ v }: { v: Verdict }) {
  const agrees = v.playedStats && v.playedStats.cards.join() === v.best.cards.join()
  const loss = v.playedStats ? v.best.mean_score - v.playedStats.mean_score : null
  return (
    <div className={`verdict${agrees ? '' : ' off'}`} data-testid="verdict" style={{ marginTop: 10 }}>
      You played <b>{cardsText(v.played)}</b>.{' '}
      {agrees ? (
        <>That was the solver's top move.</>
      ) : (
        <>
          The solver preferred <b>{cardsText(v.best.cards)}</b> ({signed(v.best.mean_score)} expected
          {v.playedStats ? `, yours ${signed(v.playedStats.mean_score)}` : ', yours was outside its top 3'}
          {loss !== null && ` — about ${loss.toFixed(1)} points`}).
        </>
      )}
      {v.network && (
        <>
          {' '}
          {sameMove(v.network.cards, v.played)
            ? `${who(v.network)} would have played the same.`
            : `${who(v.network)} would have played ${cardsText(v.network.cards)}.`}
        </>
      )}
    </div>
  )
}

function sameMove(a: Card[], b: Card[]) {
  return a.length === b.length && a.every((c) => b.includes(c))
}

function who(n: { label: string; bot: boolean }) {
  return n.bot ? `The ${n.label.toLowerCase()}` : `The network (${n.label})`
}
