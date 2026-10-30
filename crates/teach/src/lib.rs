//! The teaching layer over the Big Two engine (strategy proposal §2.5, phase 4).
//!
//! Everything here works on one **move table** ([`MoveTable`]): every legal
//! action of the player to act, valued by a solver. The table comes either from
//! the paired evaluator ([`evaluate_paired`]: every action played out on the same
//! sampled deals, so their differences carry little noise) or from any solver's
//! [`big2_search::Analysis`] ([`MoveTable::from_analysis`]).
//!
//! * [`reasons`] — reason keys (`reason.*`) that explain a move. Each key is a fact
//!   about the position or a difference between two rows of the table; nothing is
//!   guessed. Keys and arguments only, never display text (hk-games D42).
//! * [`grade`] — the paired grader: loss against the best move with a 95 %
//!   interval, thresholds as multiples of σ, and an `unclear` grade when the
//!   interval crosses a line.
//! * [`ledger`] — the luck ledger: a seat's result split into deal strength,
//!   the opponents' errors, its own errors and a residual, summing exactly.
//!
//! The crate is ours (bot stream); the app stream calls it from
//! `crates/app-api` without the bot stream editing their crate. The key table
//! and both threshold tables are in `crates/teach/README.md`.
//!
//! Everything runs on the calling thread (no rayon), so it works in WebAssembly.

pub mod grade;
pub mod ledger;
pub mod reasons;
pub mod table;

pub use grade::{
    evaluate_paired, evaluate_paired_with, grade_from_table, grade_paired, grade_paired_table,
    Graded, PairedConfig, Thresholds, SIGMA_TIERED,
};
pub use ledger::{deal_strength, deal_tables, luck_ledger, Ledger, LedgerStep};
pub use reasons::{reasons, reasons_for, Reason};
pub use table::{MoveRow, MoveTable};
