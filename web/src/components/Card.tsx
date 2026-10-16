import type { Card as CardCode } from '../api'
import { cardName, isRed, rankLabel, suitSymbol } from '../cards'

interface Props {
  card: CardCode
  small?: boolean
  selected?: boolean
  dim?: boolean
  onClick?: () => void
}

/** A face-up card drawn with CSS. Rendered as a button when clickable. */
export function Card({ card, small, selected, dim, onClick }: Props) {
  const cls = ['card', isRed(card) && 'red', small && 'small', selected && 'selected', dim && 'dim']
    .filter(Boolean)
    .join(' ')
  const face = (
    <>
      <span className="corner">
        <span>{rankLabel(card)}</span>
        <span className="s">{suitSymbol(card)}</span>
      </span>
      <span className="pip" aria-hidden>
        {suitSymbol(card)}
      </span>
    </>
  )
  if (onClick) {
    return (
      <button
        type="button"
        className={cls}
        onClick={onClick}
        aria-pressed={selected}
        aria-label={cardName(card)}
        data-card={card}
      >
        {face}
      </button>
    )
  }
  return (
    <div className={cls} role="img" aria-label={cardName(card)} data-card={card}>
      {face}
    </div>
  )
}

export function CardBacks({ count }: { count: number }) {
  const shown = Math.min(count, 13)
  return (
    <div className="backs" aria-label={`${count} cards`}>
      {Array.from({ length: shown }, (_, i) => (
        <div className="card-back" key={i} />
      ))}
    </div>
  )
}

/** A played combination (or "pass"). */
export function Played({ cards, small = true }: { cards: CardCode[]; small?: boolean }) {
  if (cards.length === 0) return <span className="badge">pass</span>
  return (
    <div className="played">
      {cards.map((c) => (
        <Card key={c} card={c} small={small} />
      ))}
    </div>
  )
}
