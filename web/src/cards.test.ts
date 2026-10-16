import { describe, expect, it } from 'vitest'
import { cardIndex, cardsText, completions, matchLegal, sameCards, sortCards } from './cards'

describe('card order', () => {
  it('follows Big Two strength', () => {
    expect(cardIndex('3d')).toBe(0)
    expect(cardIndex('3c')).toBe(1)
    expect(cardIndex('As')).toBe(47)
    expect(cardIndex('2s')).toBe(51)
  })

  it('sorts by rank or by suit', () => {
    expect(sortCards(['2s', '3d', 'Tc', '3s'])).toEqual(['3d', '3s', 'Tc', '2s'])
    expect(sortCards(['2s', '3d', 'Tc', '3s'], 'suit')).toEqual(['3d', 'Tc', '3s', '2s'])
  })

  it('formats with symbols', () => {
    expect(cardsText(['Tc', '3d'])).toBe('3♦ 10♣')
    expect(cardsText([])).toBe('pass')
  })
})

describe('selection', () => {
  const legal = [['3d'], ['3d', '3c'], ['3d', '4c', '5h', '6s', '7d'], []]

  it('matches a legal action regardless of order', () => {
    expect(sameCards(['3c', '3d'], ['3d', '3c'])).toBe(true)
    expect(matchLegal(['3c', '3d'], legal)).toEqual(['3d', '3c'])
    expect(matchLegal(['4c'], legal)).toBeUndefined()
    expect(matchLegal([], legal)).toEqual([])
  })

  it('lists cards that can complete a legal play', () => {
    expect([...completions(['3c'], legal)].sort()).toEqual(['3c', '3d'])
    expect(completions(['5h'], legal).has('7d')).toBe(true)
    expect(completions(['Ks'], legal).size).toBe(0)
  })
})
