import { useEffect, useRef } from 'react'
import type { GameView } from '../api'
import { cardsText, KIND_LABEL } from '../cards'

/** Every action so far, grouped by trick, newest at the bottom. */
export function History({ view, names }: { view: GameView; names: string[] }) {
  const box = useRef<HTMLDivElement>(null)
  // Keep the newest move in view by scrolling the list itself (not the page).
  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight
  }, [view.history.length])
  const starts = new Set(view.trick_starts)
  return (
    <div className="history" aria-live="polite" ref={box}>
      {view.history.length === 0 && <span className="muted">No moves yet.</span>}
      {view.history.map((a, i) => (
        <div key={i}>
          {i > 0 && starts.has(i) && <div className="trick-sep" />}
          <div className="row">
            <span className="who">{names[a.player]}</span>
            {a.kind === 'pass' ? (
              <span className="pass">pass</span>
            ) : (
              <span>
                {cardsText(a.cards)} <span className="muted">{KIND_LABEL[a.kind] ?? a.kind}</span>
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}
