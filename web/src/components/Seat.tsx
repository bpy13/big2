import type { ActionView } from '../api'
import { CardBacks, Played } from './Card'

interface Props {
  name: string
  detail: string
  count: number
  isTurn: boolean
  lockedOut: boolean
  /** This player's latest action in the current trick, if any. */
  last?: ActionView
  thinking: boolean
}

/** An opponent's seat: name, card count (as card backs) and their last action this trick. */
export function Seat({ name, detail, count, isTurn, lockedOut, last, thinking }: Props) {
  return (
    <div className={`seat${isTurn ? ' turn' : ''}`} data-testid={`seat-${name}`}>
      <div className="name">
        {name} {count === 1 && <span className="badge warn">last card!</span>}
      </div>
      <div className="sub">
        {detail} · {count} {count === 1 ? 'card' : 'cards'}
        {lockedOut && ' · sitting out'}
      </div>
      <CardBacks count={count} />
      <div className="last">
        {last ? <Played cards={last.cards} /> : isTurn && thinking ? <span className="badge">thinking…</span> : null}
      </div>
    </div>
  )
}
