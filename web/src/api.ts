// Typed client for the FastAPI backend (python/poker/server/app.py).
// Cards are codes like "Td" (rank + suit); see docs/rules/big2.md.

export type Card = string

export interface Rules {
  straights: string
  flush_order: string
  pass_rule: string
  scoring: string
  last_card_duty: boolean
}

export interface ActionView {
  player: number
  cards: Card[]
  kind: string // "single", "pair", ..., or "pass"
}

export interface GameView {
  id: string
  seat: number
  rules: Rules
  bots: string
  seed: number
  hand: Card[]
  hand_sizes: number[]
  current_player: number | null
  your_turn: boolean
  is_opening: boolean
  current_play: { player: number; cards: Card[]; kind: string } | null
  trick: ActionView[]
  history: ActionView[]
  trick_starts: number[]
  legal: Card[][]
  locked_out: boolean[]
  over: boolean
  winner: number | null
  scores: number[] | null
  penalties: number[] | null
  final_hands: Card[][] | null
  initial_hands: Card[][] | null
}

export interface MoveStats {
  cards: Card[]
  kind: string
  win_rate: number
  win_ci95: number
  mean_score: number
  score_ci95: number
  visits: number
  share: number
}

export interface NetworkMove {
  cards: Card[]
  kind: string
  mean_score: number
  win_rate: number
}

export interface Analysis {
  simulations: number
  seconds: number
  objective: 'score' | 'win'
  moves: MoveStats[]
  /** The trained RL network's own top moves (null if no checkpoint or non-default rules). */
  network: NetworkMove[] | null
}

export interface ReviewItem {
  turn: number
  hand: Card[]
  to_beat: Card[] | null
  played: Card[]
  played_stats: MoveStats
  best: MoveStats
  top: MoveStats[]
  score_loss: number
  agrees: boolean
}

export interface Config {
  difficulties: Record<string, string>
  rules: Record<keyof Rules, (string | boolean)[]>
  defaults: Rules
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method,
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const j = await res.json()
      detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  config: () => request<Config>('GET', '/config'),
  newGame: (opts: { difficulty?: string; bots?: string; rules?: Partial<Rules>; seed?: number }) =>
    request<GameView>('POST', '/games', opts),
  game: (id: string) => request<GameView>('GET', `/games/${id}`),
  play: (id: string, cards: Card[]) => request<GameView>('POST', `/games/${id}/play`, { cards }),
  step: (id: string) => request<{ move: ActionView | null; view: GameView }>('POST', `/games/${id}/step`),
  analyze: (id: string, opts: { iterations?: number; objective?: 'score' | 'win'; top?: number } = {}) =>
    request<Analysis>('POST', `/games/${id}/analyze`, opts),
  review: (id: string) => request<ReviewItem[]>('GET', `/games/${id}/review`),
}
