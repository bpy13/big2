import type { Card as CardCode } from '../api'
import { sortCards, type SortMode } from '../cards'
import { Card } from './Card'

interface Props {
  cards: CardCode[]
  selected: CardCode[]
  /** Cards that can still be part of a legal play with the current selection. */
  playable: Set<CardCode> | null
  sort: SortMode
  enabled: boolean
  onToggle: (card: CardCode) => void
}

/** The human's hand: click cards to select them. */
export function Hand({ cards, selected, playable, sort, enabled, onToggle }: Props) {
  return (
    <div className={`hand${enabled ? '' : ' disabled'}`} role="group" aria-label="Your hand">
      {sortCards(cards, sort).map((c) => (
        <Card
          key={c}
          card={c}
          selected={selected.includes(c)}
          dim={enabled && playable !== null && !playable.has(c) && !selected.includes(c)}
          onClick={enabled ? () => onToggle(c) : undefined}
        />
      ))}
    </div>
  )
}
