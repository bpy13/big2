import { useState } from 'react'
import type { Config, Rules } from '../api'
import { useEscape } from '../useEscape'

const DIFFICULTY_HELP: Record<string, string> = {
  easy: 'Greedy bot: sheds its weakest cards first.',
  medium: 'Solver with 300 simulations per move.',
  hard: 'Solver with 2,000 simulations per move.',
  expert: 'Solver with 10,000 simulations per move (4 threads).',
  rl: 'The chosen network, one forward pass per move. Default rules only.',
  rl_search: 'RL network + search: plays out sampled deals with the network for its top moves. Default rules only.',
}

const DIFFICULTY_LABEL: Record<string, string> = {
  easy: 'Easy',
  medium: 'Medium',
  hard: 'Hard',
  expert: 'Expert',
  rl: 'RL network',
  rl_search: 'RL network + search',
}

const RULE_LABELS: Record<keyof Rules, string> = {
  straights: 'Straights',
  flush_order: 'Flush comparison',
  pass_rule: 'After passing',
  scoring: 'Scoring',
  last_card_duty: 'Last-card duty',
}

const OPTION_LABELS: Record<string, string> = {
  with_low_twos: 'A-2-3-4-5 and 2-3-4-5-6 allowed',
  no_twos: 'No 2s in straights',
  rank_first: 'Highest card first',
  suit_first: 'Suit first',
  may_reenter: 'May play again in the trick',
  locked_out: 'Sit out the rest of the trick',
  tiered: 'Tiered (×1/×2/×3/×4)',
  original: 'Original (×1/×2/×3)',
  win_loss: 'Win/loss only',
  true: 'On',
  false: 'Off',
}

export interface NewGameChoice {
  difficulty: string
  rules: Rules
  /** Network id for the RL bots and the hint (undefined = the server's default). */
  model?: string
}

interface Props {
  config: Config
  initial: NewGameChoice
  onStart: (c: NewGameChoice) => void
  onCancel?: () => void
}

/** New-game settings: bot difficulty and rule variants (docs/rules/big2.md). */
export function NewGameDialog({ config, initial, onStart, onCancel }: Props) {
  const [difficulty, setDifficulty] = useState(initial.difficulty)
  const [rules, setRules] = useState<Rules>(initial.rules)
  useEscape(onCancel)
  const known = config.models.some((m) => m.id === initial.model)
  const [model, setModel] = useState(known ? initial.model : config.models[0]?.id)
  const chosenModel = config.models.find((m) => m.id === model)
  return (
    <div className="overlay" role="dialog" aria-modal="true" aria-labelledby="new-game-title">
      <div className="dialog">
        <h2 id="new-game-title">New game</h2>
        <div className="form">
          <label htmlFor="difficulty">Opponents</label>
          <select id="difficulty" value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
            {Object.keys(config.difficulties).map((d) => (
              <option key={d} value={d}>
                {DIFFICULTY_LABEL[d] ?? d}
              </option>
            ))}
          </select>
          <span className="help">{DIFFICULTY_HELP[difficulty] ?? config.difficulties[difficulty]}</span>
          {config.models.length > 0 && (
            <>
              <label htmlFor="model">Network</label>
              <select id="model" value={model} onChange={(e) => setModel(e.target.value)}>
                {config.models.map((m, i) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                    {i === 0 ? ' (default)' : ''}
                  </option>
                ))}
              </select>
              <span className="help">
                For the RL bots and the study-mode hint. {chosenModel?.note}
                {chosenModel && !chosenModel.loaded && ' Loaded when you deal.'}
              </span>
            </>
          )}
          {(Object.keys(RULE_LABELS) as (keyof Rules)[]).map((k) => (
            <RuleSelect
              key={k}
              id={k}
              label={RULE_LABELS[k]}
              options={config.rules[k]}
              value={rules[k]}
              onChange={(v) => setRules({ ...rules, [k]: v })}
            />
          ))}
        </div>
        <div className="row-end">
          {onCancel && (
            <button type="button" onClick={onCancel}>
              Cancel
            </button>
          )}
          <button type="button" className="primary" onClick={() => onStart({ difficulty, rules, model })}>
            Deal
          </button>
        </div>
      </div>
    </div>
  )
}

function RuleSelect(props: {
  id: string
  label: string
  options: (string | boolean)[]
  value: string | boolean
  onChange: (v: string | boolean) => void
}) {
  return (
    <>
      <label htmlFor={props.id}>{props.label}</label>
      <select
        id={props.id}
        value={String(props.value)}
        onChange={(e) => {
          const raw = e.target.value
          props.onChange(raw === 'true' ? true : raw === 'false' ? false : raw)
        }}
      >
        {props.options.map((o) => (
          <option key={String(o)} value={String(o)}>
            {OPTION_LABELS[String(o)] ?? String(o)}
          </option>
        ))}
      </select>
    </>
  )
}
