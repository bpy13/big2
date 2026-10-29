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
use big2_games::{Outcome, Transposable};
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

    /// [`Endgame::solve_actions`] drawing on a node `budget` shared across calls
    /// (the information-set endgame solves many deals under one cap): the
    /// positions expanded are taken off `budget`. `None` once it is used up.
    pub fn solve_actions_within(
        &mut self,
        state: &G,
        budget: &mut u64,
    ) -> Option<Vec<(G::Action, Outcome)>> {
        if self.table.len() > self.max_table {
            self.table.clear();
        }
        let r = self.solve_actions(state, *budget);
        *budget = budget.saturating_sub(self.nodes);
        r
    }

    /// Moves until the deal ends when every player follows the max^n choice
    /// (the first best action, as [`Endgame::solve`] breaks ties), expanding
    /// positions from `budget` when they are not in the table yet.
    pub fn line_length(&mut self, state: &G, budget: &mut u64) -> Option<u32> {
        let mut s = state.clone();
        let mut moves = 0u32;
        while s.outcome().is_none() {
            let player = s.current_player().expect("not terminal");
            let mut best: Option<(f64, G::Action)> = None;
            for a in s.legal_actions_vec() {
                let mut child = s.clone();
                child.apply(a);
                self.nodes = 0;
                let o = self.search(&child, *budget);
                *budget = budget.saturating_sub(self.nodes);
                let u = self.objective.utility(&s, &o?, player);
                if best.as_ref().is_none_or(|(bu, _)| u > *bu) {
                    best = Some((u, a));
                }
            }
            s.apply(best.expect("a non-terminal state has a legal action").1);
            moves += 1;
        }
        Some(moves)
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
