//! Rule variants. See `docs/rules/big2.md` for what each one means.

use std::fmt;
use std::str::FromStr;

/// Which five-card rank sequences count as straights (rules §3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Straights {
    /// `3-4-5-6-7 … T-J-Q-K-A`, plus `A-2-3-4-5` and `2-3-4-5-6` as the two highest.
    #[default]
    WithLowTwos,
    /// Only `3-4-5-6-7 … T-J-Q-K-A`; a 2 can never be in a straight.
    NoTwos,
}

/// How two flushes compare (rules §4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum FlushOrder {
    /// Ranks high to low first; suit only breaks a full tie.
    #[default]
    RankFirst,
    /// Suit first, then ranks high to low.
    SuitFirst,
}

/// What passing means for the rest of the trick (rules §5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PassRule {
    /// A player who passed may play again later in the same trick.
    #[default]
    MayReenter,
    /// A player who passed sits out the rest of the trick.
    LockedOut,
}

/// How a finished deal is scored (rules §7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Scoring {
    /// Penalty multipliers 1–7 ×1, 8–9 ×2, 10–12 ×3, 13 ×4; pairwise settlement.
    #[default]
    Tiered,
    /// Penalty multipliers 1–9 ×1, 10–12 ×2, 13 ×3; pairwise settlement.
    Original,
    /// Winner +3, everyone else −1.
    WinLoss,
}

impl Scoring {
    /// Penalty for finishing with `cards_left` cards. Zero for `WinLoss`.
    pub fn penalty(self, cards_left: u32) -> u32 {
        let multiplier = match (self, cards_left) {
            (Scoring::WinLoss, _) | (_, 0) => return 0,
            (Scoring::Tiered, 1..=7) => 1,
            (Scoring::Tiered, 8..=9) => 2,
            (Scoring::Tiered, 10..=12) => 3,
            (Scoring::Tiered, _) => 4,
            (Scoring::Original, 1..=9) => 1,
            (Scoring::Original, 10..=12) => 2,
            (Scoring::Original, _) => 3,
        };
        cards_left * multiplier
    }
}

/// The full set of rule variants for a game. `RuleSet::default()` is the
/// project's default table rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuleSet {
    pub straights: Straights,
    pub flush_order: FlushOrder,
    pub pass_rule: PassRule,
    pub scoring: Scoring,
    /// When the next player holds one card, any single played must be the
    /// player's highest card (rules §5). On by default.
    pub last_card_duty: bool,
}

impl Default for RuleSet {
    fn default() -> Self {
        RuleSet {
            straights: Straights::default(),
            flush_order: FlushOrder::default(),
            pass_rule: PassRule::default(),
            scoring: Scoring::default(),
            last_card_duty: true,
        }
    }
}

/// Error for an unknown rule-variant name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownVariant(pub String);

impl fmt::Display for UnknownVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown rule variant {:?}", self.0)
    }
}

impl std::error::Error for UnknownVariant {}

macro_rules! names {
    ($ty:ty { $($variant:ident => $name:literal),+ $(,)? }) => {
        impl $ty {
            /// The snake_case name used in config files, the CLI and Python.
            pub fn name(self) -> &'static str {
                match self { $(<$ty>::$variant => $name),+ }
            }
            /// All variants, default first.
            pub const ALL: &'static [$ty] = &[$(<$ty>::$variant),+];
        }
        impl FromStr for $ty {
            type Err = UnknownVariant;
            fn from_str(s: &str) -> Result<Self, UnknownVariant> {
                match s { $($name => Ok(<$ty>::$variant),)+ _ => Err(UnknownVariant(s.to_string())) }
            }
        }
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.name()) }
        }
    };
}

names!(Straights { WithLowTwos => "with_low_twos", NoTwos => "no_twos" });
names!(FlushOrder { RankFirst => "rank_first", SuitFirst => "suit_first" });
names!(PassRule { MayReenter => "may_reenter", LockedOut => "locked_out" });
names!(Scoring { Tiered => "tiered", Original => "original", WinLoss => "win_loss" });

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_rules_doc() {
        let r = RuleSet::default();
        assert_eq!(r.straights, Straights::WithLowTwos);
        assert_eq!(r.flush_order, FlushOrder::RankFirst);
        assert_eq!(r.pass_rule, PassRule::MayReenter);
        assert_eq!(r.scoring, Scoring::Tiered);
        assert!(r.last_card_duty);
    }

    #[test]
    fn penalty_tables_match_rules_doc() {
        let tiered: Vec<u32> = (0..=13).map(|n| Scoring::Tiered.penalty(n)).collect();
        assert_eq!(tiered, [0, 1, 2, 3, 4, 5, 6, 7, 16, 18, 30, 33, 36, 52]);
        let original: Vec<u32> = (0..=13).map(|n| Scoring::Original.penalty(n)).collect();
        assert_eq!(original, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 22, 24, 39]);
    }

    #[test]
    fn names_round_trip() {
        for s in Scoring::ALL {
            assert_eq!(s.name().parse::<Scoring>().unwrap(), *s);
        }
        assert!("bogus".parse::<PassRule>().is_err());
    }
}
