//! Reasons for a move, derived from the position and the solver's numbers.
//!
//! Every key here is a fact the code checks; nothing is guessed. The list of
//! keys and what each one means is in `README.md` ("Reason keys").

use crate::shapes::{round2, Reason};
use big2_big2::card::{format_mask, rank_of, FULL_DECK};
use big2_big2::movegen::legal_combos;
use big2_big2::{Action, State};
use big2_cards::mask::{highest, lowest};
use big2_games::Game;
use big2_search::MoveStats;

/// Big Two rank index of a 2.
const RANK_TWO: u8 = 12;

fn has_two(cards: u64) -> bool {
    (0..4).any(|s| cards & (1u64 << (RANK_TWO * 4 + s)) != 0)
}

fn count_twos(cards: u64) -> u32 {
    (0..4)
        .filter(|s| cards & (1u64 << (RANK_TWO * 4 + s)) != 0)
        .count() as u32
}

/// Reasons that follow from the position alone (before the move is made).
///
/// `legal` is the full legal-action list for the player to act, so the reasons
/// can contrast the move with its alternatives.
pub fn position_reasons(state: &State, action: Action, legal: &[Action]) -> Vec<Reason> {
    let mut out = Vec::new();
    let Some(player) = state.current_player() else {
        return out;
    };
    if legal.len() == 1 {
        out.push(Reason::new("reason.forced"));
        return out;
    }
    let hand = state.hand(player);
    let plays: Vec<_> = legal
        .iter()
        .filter_map(|a| match a {
            Action::Play(c) => Some(*c),
            Action::Pass => None,
        })
        .collect();

    let combo = match action {
        Action::Pass => {
            // Passing while a play exists: the player keeps those cards for later.
            out.push(Reason::new("reason.pass_holds").arg("playable", plays.len() as u64));
            return out;
        }
        Action::Play(c) => c,
    };

    let left = hand.count_ones() - combo.size();
    if left == 0 {
        out.push(Reason::new("reason.goes_out"));
    } else {
        out.push(Reason::new("reason.cards_left").arg("cards_left", left as u64));
    }

    // No combination of the same size among the cards not yet seen beats this
    // play, so no opponent can beat it (whatever they hold) and the player leads again.
    let unseen = FULL_DECK & !state.played() & !hand;
    let mut beating = Vec::new();
    legal_combos(unseen, Some(&combo), 0, state.rules(), &mut beating);
    if beating.is_empty() && left > 0 {
        out.push(Reason::new("reason.keeps_control"));
    }

    if state.current_play().is_some() {
        // The weakest combination that still beats the current play.
        let min_key = plays.iter().map(|c| c.key).min();
        if min_key == Some(combo.key) && plays.len() > 1 {
            out.push(Reason::new("reason.lowest_beating"));
        }
    } else if let Some(low) = lowest(hand) {
        // Leading with the hand's weakest card, which is the hardest to shed later.
        if combo.cards & (1u64 << low) != 0 {
            out.push(Reason::new("reason.sheds_weakest").arg("card", format_mask(1u64 << low)));
        }
    }

    // Keeps every 2 while some other legal play would spend one.
    if has_two(hand)
        && !has_two(combo.cards)
        && plays
            .iter()
            .any(|c| c.cards != combo.cards && has_two(c.cards))
    {
        out.push(Reason::new("reason.saves_two").arg("twos_kept", count_twos(hand) as u64));
    }

    // Keeps the hand's highest card while some other legal play would spend it.
    if let Some(top) = highest(hand) {
        let top_bit = 1u64 << top;
        if combo.cards & top_bit == 0
            && rank_of(top) != RANK_TWO
            && plays.iter().any(|c| c.cards & top_bit != 0)
        {
            out.push(Reason::new("reason.saves_high_card").arg("card", format_mask(top_bit)));
        }
    }

    // An opponent is close to going out.
    let sizes = state.hand_sizes();
    if let Some((p, n)) = (0..4)
        .filter(|&p| p != player && sizes[p] > 0)
        .map(|p| (p, sizes[p]))
        .min_by_key(|&(_, n)| n)
    {
        if n <= 2 {
            out.push(
                Reason::new("reason.opponent_nearly_out")
                    .arg("player", p as u64)
                    .arg("cards_left", n as u64),
            );
        }
    }
    out
}

/// Reasons from the solver's numbers: the margin of the best move over the
/// runner-up, or how much a move loses against the best.
pub fn solver_reasons(
    this: &MoveStats<Action>,
    best: &MoveStats<Action>,
    second: Option<&MoveStats<Action>>,
) -> Vec<Reason> {
    let mut out = Vec::new();
    if this.action == best.action {
        if let Some(s) = second {
            let margin = best.mean_score - s.mean_score;
            if margin > 0.0 {
                out.push(Reason::new("reason.score_margin").arg("points", round2(margin)));
            }
        }
    } else {
        let loss = best.mean_score - this.mean_score;
        if loss > 0.0 {
            out.push(Reason::new("reason.score_loss").arg("points", round2(loss)));
        }
    }
    out
}
