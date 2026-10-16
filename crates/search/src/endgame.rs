//! Exact search of a fully known position.
//!
//! Inside a determinization every card is known, so the rest of the game is a
//! perfect-information game. With few cards left it is small enough to search
//! completely. This module does **max^n** search (Luckhardt & Irani, 1986): at
//! each node the player to move picks the action that maximizes *their own*
//! utility, assuming everyone else does the same. Positions reached by different
//! move orders are solved once, via a transposition table.
//!
//! Max^n is the natural multi-player generalization of minimax, but it is not an
//! equilibrium guarantee: when a player is indifferent between actions that affect
//! others differently, the tie is broken by move order.

use crate::analysis::Objective;
use poker_games::{Outcome, Transposable};
use std::collections::HashMap;

/// A reusable max^n solver. Keep one per thread and call [`Endgame::solve`]
/// repeatedly; the transposition table carries over between calls (values of fully
/// known positions do not depend on how they were reached).
pub struct Endgame<G: Transposable> {
    objective: Objective,
    table: HashMap<G::Key, Outcome>,
    /// Clear the table once it holds this many positions, to bound memory.
    max_table: usize,
    nodes: u64,
}

impl<G: Transposable> Endgame<G> {
    pub fn new(objective: Objective) -> Self {
        Endgame {
            objective,
            table: HashMap::new(),
            max_table: 2_000_000,
            nodes: 0,
        }
    }

    /// Positions currently cached.
    pub fn table_len(&self) -> usize {
        self.table.len()
    }

    /// Solves `state` exactly, expanding at most `max_nodes` new positions.
    /// Returns the outcome under max^n play, or `None` if the budget ran out.
    pub fn solve(&mut self, state: &G, max_nodes: u64) -> Option<Outcome> {
        if self.table.len() > self.max_table {
            self.table.clear();
        }
        self.nodes = 0;
        self.search(state, max_nodes)
    }

    /// Solves every legal action of `state`: the outcome after playing each one.
    pub fn solve_actions(
        &mut self,
        state: &G,
        max_nodes: u64,
    ) -> Option<Vec<(G::Action, Outcome)>> {
        self.nodes = 0;
        let mut out = Vec::new();
        for a in state.legal_actions_vec() {
            let mut child = state.clone();
            child.apply(a);
            out.push((a, self.search(&child, max_nodes)?));
        }
        Some(out)
    }

    fn search(&mut self, state: &G, max_nodes: u64) -> Option<Outcome> {
        if let Some(o) = state.outcome() {
            return Some(o);
        }
        let key = state.position_key();
        if let Some(o) = self.table.get(&key) {
            return Some(o.clone());
        }
        self.nodes += 1;
        if self.nodes > max_nodes {
            return None;
        }
        let player = state.current_player().expect("not terminal");
        let mut best: Option<(f64, Outcome)> = None;
        for a in state.legal_actions_vec() {
            let mut child = state.clone();
            child.apply(a);
            let o = self.search(&child, max_nodes)?;
            let u = self.objective.utility(state, &o, player);
            if best.as_ref().is_none_or(|(bu, _)| u > *bu) {
                best = Some((u, o));
            }
        }
        let (_, o) = best.expect("a non-terminal state has a legal action");
        self.table.insert(key, o.clone());
        Some(o)
    }
}

/// Plain max^n without a transposition table or budget. Exponentially slow; used
/// only to test [`Endgame`] on tiny positions.
pub fn solve_naive<G: Transposable>(state: &G, objective: Objective) -> Outcome {
    if let Some(o) = state.outcome() {
        return o;
    }
    let player = state.current_player().unwrap();
    let mut best: Option<(f64, Outcome)> = None;
    for a in state.legal_actions_vec() {
        let mut child = state.clone();
        child.apply(a);
        let o = solve_naive(&child, objective);
        let u = objective.utility(state, &o, player);
        if best.as_ref().is_none_or(|(bu, _)| u > *bu) {
            best = Some((u, o));
        }
    }
    best.unwrap().1
}
