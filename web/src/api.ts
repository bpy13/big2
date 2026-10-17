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
  /** The network used for the hint (a model id), or null if none applies. */
  model: string | null
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

/** A trained network the server can use (see python/poker/server/models.py). */
export interface ModelInfo {
  id: string
  label: string
  note: string
  /** "q": predicts score and win chance per move; "policy" (PPO): a probability per move. */
  kind: 'q' | 'policy'
  loaded: boolean
}

export interface NetworkMove {
  cards: Card[]
  kind: string
  /** Null for a policy network. */
  mean_score: number | null
  win_rate: number | null
  /** Policy networks only. */
  prob: number | null
}

export interface NetworkView {
  model: ModelInfo
  moves: NetworkMove[]
  /** The network's first choice, as the solver rates it. */
  pick: {
    cards: Card[]
    solver_rank: number | null
    solver: MoveStats | null
    /** How much worse than the solver's best: points (objective "score") or win probability ("win"). */
    loss: number | null
  }
}

export interface Analysis {
  simulations: number
  seconds: number
  objective: 'score' | 'win'
  moves: MoveStats[]
  /** The hint network's view (null if there is no network or the rules are not the default). */
  network: NetworkView | null
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
  /** Available networks, default first. */
  models: ModelInfo[]
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
  newGame: (opts: { difficulty?: string; bots?: string; rules?: Partial<Rules>; seed?: number; model?: string }) =>
    request<GameView>('POST', '/games', opts),
  game: (id: string) => request<GameView>('GET', `/games/${id}`),
  play: (id: string, cards: Card[]) => request<GameView>('POST', `/games/${id}/play`, { cards }),
  step: (id: string) => request<{ move: ActionView | null; view: GameView }>('POST', `/games/${id}/step`),
  analyze: (id: string, opts: { iterations?: number; objective?: 'score' | 'win'; top?: number } = {}) =>
    request<Analysis>('POST', `/games/${id}/analyze`, opts),
  /** Loads a network and uses it for this game's hint. */
  setModel: (id: string, model: string) => request<ModelInfo>('POST', `/games/${id}/model`, { model }),
  review: (id: string) => request<ReviewItem[]>('GET', `/games/${id}/review`),
}
