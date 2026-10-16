// Card helpers for display and selection. Codes are rank + suit: "3d", "Tc", "2s".
// Big Two order: ranks 3 < 4 < ... < K < A < 2; suits ♦ < ♣ < ♥ < ♠.

import type { Card } from './api'

export const RANKS = '3456789TJQKA2'
export const SUITS = 'dchs'

const SUIT_SYMBOL: Record<string, string> = { d: '♦', c: '♣', h: '♥', s: '♠' }
const SUIT_NAME: Record<string, string> = { d: 'diamonds', c: 'clubs', h: 'hearts', s: 'spades' }

export function rankOf(card: Card): string {
  return card[0].toUpperCase()
}

export function suitOf(card: Card): string {
  return card[1].toLowerCase()
}

/** Big Two strength index: 0 (3♦) … 51 (2♠). */
export function cardIndex(card: Card): number {
  return RANKS.indexOf(rankOf(card)) * 4 + SUITS.indexOf(suitOf(card))
}

export function rankLabel(card: Card): string {
  const r = rankOf(card)
  return r === 'T' ? '10' : r
}

export function suitSymbol(card: Card): string {
  return SUIT_SYMBOL[suitOf(card)]
}

export function isRed(card: Card): boolean {
  return suitOf(card) === 'd' || suitOf(card) === 'h'
}

/** Accessible name, e.g. "10 of clubs". */
export function cardName(card: Card): string {
  return `${rankLabel(card)} of ${SUIT_NAME[suitOf(card)]}`
}

/** Short text with suit symbols, e.g. "3♦ 10♣". Empty list is "pass". */
export function cardsText(cards: Card[]): string {
  return cards.length === 0 ? 'pass' : sortCards(cards).map((c) => rankLabel(c) + suitSymbol(c)).join(' ')
}

export type SortMode = 'rank' | 'suit'

export function sortCards(cards: Card[], mode: SortMode = 'rank'): Card[] {
  const key = (c: Card) =>
    mode === 'rank' ? cardIndex(c) : SUITS.indexOf(suitOf(c)) * 13 + RANKS.indexOf(rankOf(c))
  return [...cards].sort((a, b) => key(a) - key(b))
}

/** Same set of cards, ignoring order. */
export function sameCards(a: Card[], b: Card[]): boolean {
  if (a.length !== b.length) return false
  const s = new Set(a)
  return b.every((c) => s.has(c))
}

/** The legal action equal to `selected`, or undefined. An empty selection is a pass. */
export function matchLegal(selected: Card[], legal: Card[][]): Card[] | undefined {
  return legal.find((l) => sameCards(l, selected))
}

/** Cards that appear in at least one legal action containing every selected card,
 * i.e. cards that could still complete a legal play from the current selection. */
export function completions(selected: Card[], legal: Card[][]): Set<Card> {
  const out = new Set<Card>()
  for (const l of legal) {
    if (l.length === 0) continue
    if (selected.every((c) => l.includes(c))) l.forEach((c) => out.add(c))
  }
  return out
}

export const KIND_LABEL: Record<string, string> = {
  single: 'single',
  pair: 'pair',
  triple: 'triple',
  straight: 'straight',
  flush: 'flush',
  full_house: 'full house',
  four_of_a_kind: 'four of a kind',
  straight_flush: 'straight flush',
  pass: 'pass',
}
