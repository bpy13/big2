import { useState } from 'react'
import { api, type GameView, type ReviewItem } from '../api'
import { cardsText } from '../cards'
import { useEscape } from '../useEscape'
import { Played } from './Card'

/** Points of expected score lost before a move is flagged as a mistake. */
const MISTAKE_THRESHOLD = 1.0

interface Props {
  view: GameView
  names: string[]
  onNewGame: () => void
  onClose: () => void
}

/** End-of-deal results, every hand revealed, and the post-game review. */
export function GameOver({ view, names, onNewGame, onClose }: Props) {
  const [review, setReview] = useState<ReviewItem[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEscape(onClose)
  const you = view.seat
  const won = view.winner === you

  async function loadReview() {
    setLoading(true)
    setError(null)
    try {
      setReview(await api.review(view.id))
    } catch (e) {
      setError(String((e as Error).message))
    } finally {
      setLoading(false)
    }
  }

  const order = [0, 1, 2, 3].sort((a, b) => (view.scores?.[b] ?? 0) - (view.scores?.[a] ?? 0))
  return (
    <div className="overlay" role="dialog" aria-modal="true" aria-labelledby="result-title">
      <div className="dialog">
        <h2 id="result-title">{won ? 'You went out first!' : `${names[view.winner ?? 0]} went out first`}</h2>
        <table className="scores">
          <thead>
            <tr>
              <th>Player</th>
              <th>Cards left</th>
              <th className="num">Penalty</th>
              <th className="num">Score</th>
            </tr>
          </thead>
          <tbody>
            {order.map((p) => {
              const s = view.scores?.[p] ?? 0
              return (
                <tr key={p} data-testid={`result-${p}`}>
                  <td>
                    <b>{names[p]}</b>
                  </td>
                  <td>{view.final_hands?.[p].length ? <Played cards={view.final_hands[p]} /> : '—'}</td>
                  <td className="num">{view.penalties?.[p]}</td>
                  <td className={`num ${s > 0 ? 'pos' : s < 0 ? 'neg' : ''}`}>
                    {s > 0 ? '+' : ''}
                    {s}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>

        {!review && (
          <p className="muted">
            Review re-examines every decision you made (with full effort for every option) and flags moves that
            cost more than {MISTAKE_THRESHOLD} point of expected score.
          </p>
        )}
        {error && <p className="error">{error}</p>}
        {review && <ReviewList items={review} />}

        <div className="row-end">
          {!review && (
            <button type="button" onClick={loadReview} disabled={loading}>
              {loading ? 'Reviewing…' : 'Review my game'}
            </button>
          )}
          <button type="button" onClick={onClose}>
            Look at the table
          </button>
          <button type="button" className="primary" onClick={onNewGame}>
            New game
          </button>
        </div>
      </div>
    </div>
  )
}

function ReviewList({ items }: { items: ReviewItem[] }) {
  if (items.length === 0) return <p className="muted">All your moves were forced — nothing to review.</p>
  const agree = items.filter((i) => i.agrees).length
  const mistakes = items.filter((i) => i.score_loss > MISTAKE_THRESHOLD)
  const totalLoss = items.reduce((a, i) => a + Math.max(0, i.score_loss), 0)
  return (
    <div data-testid="review">
      <p>
        <b>{agree}</b> of {items.length} decisions matched the solver. <b>{mistakes.length}</b> flagged; estimated
        total cost {totalLoss.toFixed(1)} points.
      </p>
      {items.map((i) => {
        const flagged = i.score_loss > MISTAKE_THRESHOLD
        return (
          <div key={i.turn} className={`review-item ${i.agrees ? 'good' : flagged ? 'mistake' : ''}`}>
            <div>
              {i.to_beat ? (
                <>
                  To beat <b>{cardsText(i.to_beat)}</b>:{' '}
                </>
              ) : (
                <>Leading: </>
              )}
              you played <b>{cardsText(i.played)}</b>
              {i.agrees ? ' ✓' : ''}
            </div>
            {!i.agrees && (
              <div>
                Solver: <b>{cardsText(i.best.cards)}</b> ({i.best.mean_score >= 0 ? '+' : ''}
                {i.best.mean_score.toFixed(1)} vs {i.played_stats.mean_score >= 0 ? '+' : ''}
                {i.played_stats.mean_score.toFixed(1)}, cost {i.score_loss.toFixed(1)} ± 
                {Math.hypot(i.best.score_ci95, i.played_stats.score_ci95).toFixed(1)})
              </div>
            )}
            <div className="muted">Your hand: {cardsText(i.hand)}</div>
          </div>
        )
      })}
    </div>
  )
}
