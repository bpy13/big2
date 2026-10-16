import type { Analysis, Card, MoveStats } from '../api'
import { cardsText, KIND_LABEL } from '../cards'

export type StudyMode = 'off' | 'live' | 'quiz'

export interface Verdict {
  played: Card[]
  best: MoveStats
  playedStats: MoveStats | undefined
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
    <section className="panel" aria-label="Study mode">
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

          {p.verdict && <VerdictBox v={p.verdict} />}

          {p.yourTurn && p.mode === 'live' && (
            <>
              {p.loading && !p.analysis && <p className="muted">Thinking…</p>}
              {p.analysis && (
                <div className="moves">
                  {p.analysis.moves.map((m, i) => (
                    <MoveRow key={m.cards.join()} m={m} i={i} onPick={() => p.onPick(m.cards)} />
                  ))}
                </div>
              )}
              {p.analysis && (
                <p className="small-print">
                  {p.analysis.simulations.toLocaleString()} simulated games in {p.analysis.seconds.toFixed(2)}s.
                  Ranked by {p.analysis.objective === 'score' ? 'expected score' : 'win rate'}; click a move to
                  select it. ± is a 95% interval from simulation noise only — the solver assumes opponents
                  play like its rollout bot.
                </p>
              )}
            </>
          )}
          {p.yourTurn && p.mode === 'quiz' && !p.verdict && (
            <p className="muted">Make your move — the solver's opinion appears afterwards.</p>
          )}
          {!p.yourTurn && !p.verdict && <p className="muted">Waiting for your turn…</p>}
        </>
      )}
    </section>
  )
}

function VerdictBox({ v }: { v: Verdict }) {
  const agrees = v.playedStats && v.playedStats.cards.join() === v.best.cards.join()
  const loss = v.playedStats ? v.best.mean_score - v.playedStats.mean_score : null
  return (
    <div className={`verdict${agrees ? '' : ' off'}`} data-testid="verdict" style={{ marginBottom: 10 }}>
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
    </div>
  )
}
