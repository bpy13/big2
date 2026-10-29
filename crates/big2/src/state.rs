//! The game state of one Big Two deal and the turn flow (rules §2, §5–§7).

use crate::card::{format_mask, FULL_DECK, NUM_CARDS, THREE_OF_DIAMONDS};
use crate::combo::{classify, Combo, Kind};
use crate::movegen::{highest_card, legal_combos};
use crate::rules::{PassRule, RuleSet, Scoring};
use big2_cards::mask;
use big2_games::{Game, Outcome, Player, Rewind, Transposable};
use rand::seq::SliceRandom;
use rand::Rng;
use std::fmt;

/// Number of players at a Big Two table.
pub const NUM_PLAYERS: usize = 4;
/// Cards dealt to each player.
pub const HAND_SIZE: u32 = 13;

/// What a player does on their turn.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Pass,
    Play(Combo),
}

impl Action {
    /// The cards played (empty for a pass).
    pub fn cards(&self) -> u64 {
        match self {
            Action::Pass => 0,
            Action::Play(c) => c.cards,
        }
    }
}

impl fmt::Debug for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Pass => f.write_str("pass"),
            Action::Play(c) => write!(f, "{c:?}"),
        }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Pass => f.write_str("pass"),
            Action::Play(c) => write!(f, "{c}"),
        }
    }
}

/// One entry of the public history: who did what.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Turn {
    pub player: u8,
    pub action: Action,
}

/// Why an action was rejected by [`State::try_apply`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IllegalAction {
    GameOver,
    NotInHand(String),
    NotACombination(String),
    MustIncludeThreeOfDiamonds,
    DoesNotBeat { played: String, current: String },
    CannotPassWhenLeading,
    LastCardDuty,
}

impl fmt::Display for IllegalAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IllegalAction::GameOver => write!(f, "the deal is already over"),
            IllegalAction::NotInHand(c) => write!(f, "cards not in hand: {c}"),
            IllegalAction::NotACombination(c) => write!(f, "{c} is not a legal combination"),
            IllegalAction::MustIncludeThreeOfDiamonds => {
                write!(f, "the first play must include 3d")
            }
            IllegalAction::DoesNotBeat { played, current } => {
                write!(f, "{played} does not beat {current}")
            }
            IllegalAction::CannotPassWhenLeading => write!(f, "the leader cannot pass"),
            IllegalAction::LastCardDuty => {
                write!(
                    f,
                    "the next player has one card left: a single must be your highest card"
                )
            }
        }
    }
}

impl std::error::Error for IllegalAction {}

/// A Big Two deal in progress (or finished).
///
/// The state holds every player's hand, so it is the *true* state. A player's
/// view is their own hand plus the public parts: [`State::history`],
/// [`State::hand_sizes`], [`State::current_play`] and [`State::played`].
#[derive(Clone)]
pub struct State {
    rules: RuleSet,
    hands: [u64; NUM_PLAYERS],
    played: u64,
    current: u8,
    /// The current play and who made it; `None` while a trick is being led.
    trick: Option<(u8, Combo)>,
    /// Consecutive passes in this trick (`MayReenter`).
    passes: u8,
    /// Players who passed in this trick, one bit each (`LockedOut`).
    locked: u8,
    opening: bool,
    winner: Option<u8>,
    history: Vec<Turn>,
    /// Started from a full deal ([`State::from_hands`]), so [`Rewind`] can replay it.
    from_deal: bool,
}

impl State {
    /// Shuffles and deals a new game.
    pub fn new_deal<R: Rng + ?Sized>(rules: RuleSet, rng: &mut R) -> State {
        let mut deck: Vec<u8> = (0..NUM_CARDS).collect();
        deck.shuffle(rng);
        let mut hands = [0u64; NUM_PLAYERS];
        for (i, &card) in deck.iter().enumerate() {
            hands[i % NUM_PLAYERS] |= 1 << card;
        }
        State::from_hands(rules, hands).expect("a shuffled deck is a valid deal")
    }

    /// Starts a game from given hands. The four hands must be 13 cards each and
    /// together form the whole deck.
    pub fn from_hands(rules: RuleSet, hands: [u64; NUM_PLAYERS]) -> Result<State, String> {
        let mut seen = 0u64;
        for (p, &h) in hands.iter().enumerate() {
            if h.count_ones() != HAND_SIZE {
                return Err(format!(
                    "player {p} has {} cards, expected {HAND_SIZE}",
                    h.count_ones()
                ));
            }
            if seen & h != 0 {
                return Err(format!(
                    "player {p} shares cards with another player: {}",
                    format_mask(seen & h)
                ));
            }
            seen |= h;
        }
        if seen != FULL_DECK {
            return Err("hands do not cover the full deck".into());
        }
        let opener = hands
            .iter()
            .position(|h| h & (1 << THREE_OF_DIAMONDS) != 0)
            .unwrap() as u8;
        Ok(State {
            rules,
            hands,
            played: 0,
            current: opener,
            trick: None,
            passes: 0,
            locked: 0,
            opening: true,
            winner: None,
            history: Vec::with_capacity(64),
            from_deal: true,
        })
    }

    /// Sets up a mid-game position, for puzzles, study mode and tests.
    ///
    /// * `hands`: each player's remaining cards (disjoint; 0–13 cards each). Cards in
    ///   no hand count as already played.
    /// * `current`: the player to act.
    /// * `current_play`: the play to beat and who made it, or `None` if `current`
    ///   leads. It must be disjoint from every hand.
    ///
    /// The position is never an opening position, and the history starts empty.
    pub fn from_position(
        rules: RuleSet,
        hands: [u64; NUM_PLAYERS],
        current: usize,
        current_play: Option<(usize, u64)>,
    ) -> Result<State, String> {
        let mut seen = 0u64;
        for (p, &h) in hands.iter().enumerate() {
            if h & !FULL_DECK != 0 || h.count_ones() > HAND_SIZE {
                return Err(format!("player {p} has an invalid hand"));
            }
            if seen & h != 0 {
                return Err(format!(
                    "player {p} shares cards with another player: {}",
                    format_mask(seen & h)
                ));
            }
            seen |= h;
        }
        if current >= NUM_PLAYERS {
            return Err(format!("no player {current}"));
        }
        let out: Vec<usize> = (0..NUM_PLAYERS).filter(|&p| hands[p] == 0).collect();
        if out.len() > 1 {
            return Err("at most one player can have an empty hand".into());
        }
        let trick = match current_play {
            None => None,
            Some((holder, cards)) => {
                if holder >= NUM_PLAYERS || holder == current {
                    return Err("the current play must belong to another player".into());
                }
                if cards & seen != 0 {
                    return Err("the current play overlaps a hand".into());
                }
                let combo = classify(cards, &rules)
                    .ok_or_else(|| format!("{} is not a combination", format_mask(cards)))?;
                Some((holder as u8, combo))
            }
        };
        let winner = out.first().map(|&p| p as u8);
        if winner.is_some() && trick.map(|(h, _)| h) != winner {
            return Err("a player with no cards must have made the last play".into());
        }
        Ok(State {
            rules,
            hands,
            played: FULL_DECK & !seen,
            current: current as u8,
            trick,
            passes: 0,
            locked: 0,
            opening: false,
            winner,
            history: Vec::new(),
            from_deal: false,
        })
    }

    /// The rules this game is played under.
    pub fn rules(&self) -> &RuleSet {
        &self.rules
    }

    /// A player's current hand.
    pub fn hand(&self, player: usize) -> u64 {
        self.hands[player]
    }

    /// All four hands (the true state; hidden information included).
    pub fn hands(&self) -> [u64; NUM_PLAYERS] {
        self.hands
    }

    /// Cards left per player (public).
    pub fn hand_sizes(&self) -> [u32; NUM_PLAYERS] {
        self.hands.map(|h| h.count_ones())
    }

    /// All cards played so far (public).
    pub fn played(&self) -> u64 {
        self.played
    }

    /// The play to beat and who made it, or `None` if the current player leads.
    pub fn current_play(&self) -> Option<(usize, Combo)> {
        self.trick.map(|(p, c)| (p as usize, c))
    }

    /// `true` before the first play of the deal (3♦ must be played).
    pub fn is_opening(&self) -> bool {
        self.opening
    }

    /// Every action taken so far, in order (public).
    pub fn history(&self) -> &[Turn] {
        &self.history
    }

    /// The player who went out, once the deal is over.
    pub fn winner(&self) -> Option<usize> {
        self.winner.map(|w| w as usize)
    }

    /// Players who are sitting out the rest of this trick (`LockedOut` only).
    pub fn locked_out(&self) -> [bool; NUM_PLAYERS] {
        std::array::from_fn(|p| self.locked & (1 << p) != 0)
    }

    fn next_seat(&self, from: u8) -> u8 {
        (from + 1) % NUM_PLAYERS as u8
    }

    fn last_card_duty_applies(&self) -> bool {
        self.rules.last_card_duty
            && self.hands[self.next_seat(self.current) as usize].count_ones() == 1
    }

    /// Checks `action` for the current player and explains why it is illegal.
    pub fn check(&self, action: &Action) -> Result<(), IllegalAction> {
        if self.winner.is_some() {
            return Err(IllegalAction::GameOver);
        }
        let hand = self.hands[self.current as usize];
        match action {
            Action::Pass => {
                if self.trick.is_none() {
                    return Err(IllegalAction::CannotPassWhenLeading);
                }
            }
            Action::Play(c) => {
                if c.cards & !hand != 0 {
                    return Err(IllegalAction::NotInHand(format_mask(c.cards & !hand)));
                }
                // Re-classify so a combo built with different rules cannot sneak in.
                if classify(c.cards, &self.rules) != Some(*c) {
                    return Err(IllegalAction::NotACombination(format_mask(c.cards)));
                }
                if self.opening && c.cards & (1 << THREE_OF_DIAMONDS) == 0 {
                    return Err(IllegalAction::MustIncludeThreeOfDiamonds);
                }
                if let Some((_, current)) = &self.trick {
                    if !c.beats(current) {
                        return Err(IllegalAction::DoesNotBeat {
                            played: format_mask(c.cards),
                            current: format_mask(current.cards),
                        });
                    }
                }
                if c.kind == Kind::Single
                    && self.last_card_duty_applies()
                    && c.cards != 1 << highest_card(hand).unwrap()
                {
                    return Err(IllegalAction::LastCardDuty);
                }
            }
        }
        Ok(())
    }

    /// Validates and applies an action (for input from users or Python).
    pub fn try_apply(&mut self, action: Action) -> Result<(), IllegalAction> {
        self.check(&action)?;
        self.apply_unchecked(action);
        Ok(())
    }

    /// Builds the action for playing `cards` (or passing, if `cards == 0`) and
    /// validates it.
    pub fn action_for_cards(&self, cards: u64) -> Result<Action, IllegalAction> {
        let not_held = cards & !self.hands[self.current as usize];
        if not_held != 0 && self.winner.is_none() {
            return Err(IllegalAction::NotInHand(format_mask(not_held)));
        }
        let action = if cards == 0 {
            Action::Pass
        } else {
            Action::Play(
                classify(cards, &self.rules)
                    .ok_or_else(|| IllegalAction::NotACombination(format_mask(cards)))?,
            )
        };
        self.check(&action)?;
        Ok(action)
    }

    fn apply_unchecked(&mut self, action: Action) {
        let p = self.current;
        self.history.push(Turn { player: p, action });
        match action {
            Action::Pass => {
                let (holder, _) = self.trick.expect("cannot pass when leading");
                let trick_over = match self.rules.pass_rule {
                    PassRule::MayReenter => {
                        self.passes += 1;
                        self.current = self.next_seat(p);
                        self.passes as usize == NUM_PLAYERS - 1
                    }
                    PassRule::LockedOut => {
                        self.locked |= 1 << p;
                        self.current = self.next_unlocked(p, holder);
                        self.current == holder
                    }
                };
                if trick_over {
                    self.current = holder;
                    self.trick = None;
                    self.passes = 0;
                    self.locked = 0;
                }
            }
            Action::Play(c) => {
                debug_assert_eq!(c.cards & !self.hands[p as usize], 0);
                self.hands[p as usize] &= !c.cards;
                self.played |= c.cards;
                self.trick = Some((p, c));
                self.passes = 0;
                self.opening = false;
                if self.hands[p as usize] == 0 {
                    self.winner = Some(p);
                } else {
                    self.current = match self.rules.pass_rule {
                        PassRule::MayReenter => self.next_seat(p),
                        PassRule::LockedOut => self.next_unlocked(p, p),
                    };
                }
            }
        }
    }

    /// Next seat after `from` that is not locked out, stopping at `holder`
    /// (the player who made the current play is never locked out).
    fn next_unlocked(&self, from: u8, holder: u8) -> u8 {
        let mut s = self.next_seat(from);
        while s != holder && self.locked & (1 << s) != 0 {
            s = self.next_seat(s);
        }
        s
    }

    /// Each player's penalty if the game ended now (rules §7.1).
    pub fn penalties(&self) -> [u32; NUM_PLAYERS] {
        self.hands
            .map(|h| self.rules.scoring.penalty(h.count_ones()))
    }
}

impl Game for State {
    type Action = Action;

    fn num_players(&self) -> usize {
        NUM_PLAYERS
    }

    fn current_player(&self) -> Option<Player> {
        match self.winner {
            Some(_) => None,
            None => Some(self.current as usize),
        }
    }

    fn legal_actions(&self, out: &mut Vec<Action>) {
        out.clear();
        if self.winner.is_some() {
            return;
        }
        let hand = self.hands[self.current as usize];
        let must_include = if self.opening {
            1 << THREE_OF_DIAMONDS
        } else {
            0
        };
        let mut combos = Vec::with_capacity(64);
        legal_combos(
            hand,
            self.trick.as_ref().map(|(_, c)| c),
            must_include,
            &self.rules,
            &mut combos,
        );
        if self.last_card_duty_applies() {
            let top = 1u64 << highest_card(hand).unwrap();
            combos.retain(|c| c.kind != Kind::Single || c.cards == top);
        }
        out.extend(combos.into_iter().map(Action::Play));
        if self.trick.is_some() {
            out.push(Action::Pass);
        }
    }

    fn apply(&mut self, action: Action) {
        debug_assert_eq!(
            self.check(&action),
            Ok(()),
            "illegal action {action:?} in {self:?}"
        );
        self.apply_unchecked(action);
    }

    fn outcome(&self) -> Option<Outcome> {
        let winner = self.winner? as usize;
        let scores = match self.rules.scoring {
            Scoring::WinLoss => (0..NUM_PLAYERS)
                .map(|p| if p == winner { 3.0 } else { -1.0 })
                .collect(),
            _ => {
                // Pairwise settlement: score_i = Σ_j (p_j − p_i) = total − n·p_i.
                let pen = self.penalties();
                let total: u32 = pen.iter().sum();
                pen.iter()
                    .map(|&p| total as f64 - NUM_PLAYERS as f64 * p as f64)
                    .collect()
            }
        };
        Some(Outcome {
            scores,
            winner: Some(winner),
        })
    }

    fn endgame_size(&self) -> usize {
        self.hands.iter().map(|h| h.count_ones() as usize).sum()
    }

    fn max_abs_score(&self) -> f64 {
        match self.rules.scoring {
            Scoring::WinLoss => 3.0,
            s => (s.penalty(HAND_SIZE) * (NUM_PLAYERS as u32 - 1)) as f64,
        }
    }

    /// Re-deals the cards hidden from `observer` uniformly at random, keeping hand
    /// sizes. The one public constraint is honored: before the opening play, the
    /// player to act is known to hold 3♦.
    fn determinize<R: Rng + ?Sized>(&self, observer: Player, rng: &mut R) -> State {
        let mut s = self.clone();
        let (unknown, fixed) = self.hidden_pool(observer);
        let mut pool: Vec<u8> = mask::bits(unknown).collect();
        pool.shuffle(rng);
        let mut next = pool.into_iter();
        for (p, &fixed_cards) in fixed.iter().enumerate() {
            if p == observer {
                continue;
            }
            let mut h = fixed_cards;
            while h.count_ones() < self.hands[p].count_ones() {
                h |= 1 << next.next().expect("enough hidden cards");
            }
            s.hands[p] = h;
        }
        s
    }

    /// Cards in the other three hands (the observer sees only their count).
    fn hidden_count(&self, observer: Player) -> Option<usize> {
        Some(
            (0..NUM_PLAYERS)
                .filter(|&p| p != observer)
                .map(|p| self.hands[p].count_ones() as usize)
                .sum(),
        )
    }

    /// Every way to split the hidden cards among the other three seats with their
    /// hand sizes (and 3♦ with the opener before the opening play): at most
    /// 12! / (4!·4!·4!) = 34,650 deals for 12 hidden cards, 13!/(5!·4!·4!) = 90,090 for 13.
    fn for_each_deal(&self, observer: Player, f: &mut dyn FnMut(&State)) -> bool {
        let (unknown, fixed) = self.hidden_pool(observer);
        let others: Vec<usize> = (0..NUM_PLAYERS).filter(|&p| p != observer).collect();
        let need = |p: usize| self.hands[p].count_ones() - fixed[p].count_ones();
        let mut s = self.clone();
        for_each_subset(unknown, need(others[0]), &mut |a| {
            for_each_subset(unknown & !a, need(others[1]), &mut |b| {
                let c = unknown & !a & !b;
                debug_assert_eq!(c.count_ones(), need(others[2]));
                s.hands[others[0]] = a | fixed[others[0]];
                s.hands[others[1]] = b | fixed[others[1]];
                s.hands[others[2]] = c | fixed[others[2]];
                f(&s);
            });
        });
        true
    }
}

impl Rewind for State {
    /// The deal (each hand plus the cards that player has played) and the history.
    /// `None` for positions from [`State::from_position`].
    fn rewind(&self) -> Option<(State, Vec<Action>)> {
        if !self.from_deal {
            return None;
        }
        let mut hands = self.hands;
        for t in &self.history {
            hands[t.player as usize] |= t.action.cards();
        }
        let start = State::from_hands(self.rules, hands).ok()?;
        Some((start, self.history.iter().map(|t| t.action).collect()))
    }

    /// The one rule that ties an action to the hidden cards is the last-card duty
    /// (rules §5): a single played while the next player held one card was the
    /// player's highest card then. Everything else a determinization keeps by
    /// construction (played cards go back to whoever played them; 3♦ to the
    /// opener). So this walks the history backwards, adding each play back to its
    /// player's hand, and checks only those singles: the same answer as replaying
    /// the deal ([`big2_games::replay_consistent`]), without building any move list.
    /// Positions from [`State::from_position`] are checked from their own start.
    fn history_consistent(&self) -> bool {
        if !self.rules.last_card_duty {
            return true;
        }
        let mut hands = self.hands;
        for t in self.history.iter().rev() {
            let p = t.player as usize;
            hands[p] |= t.action.cards();
            // `hands` is now the position just before this turn.
            if let Action::Play(c) = t.action {
                if c.kind == Kind::Single
                    && hands[(p + 1) % NUM_PLAYERS].count_ones() == 1
                    && highest_card(hands[p]).map(|h| 1u64 << h) != Some(c.cards)
                {
                    return false;
                }
            }
        }
        true
    }
}

impl State {
    /// The cards hidden from `observer` that are free to be re-dealt, and the cards
    /// each seat is known to hold among them (3♦ before the opening play).
    fn hidden_pool(&self, observer: Player) -> (u64, [u64; NUM_PLAYERS]) {
        let mut fixed = [0u64; NUM_PLAYERS];
        let mut unknown = FULL_DECK & !self.played & !self.hands[observer];
        if self.opening && self.current as usize != observer {
            fixed[self.current as usize] = 1 << THREE_OF_DIAMONDS;
            unknown &= !(1 << THREE_OF_DIAMONDS);
        }
        (unknown, fixed)
    }
}

/// Calls `f` with every subset of `pool` that has `k` cards.
fn for_each_subset(pool: u64, k: u32, f: &mut dyn FnMut(u64)) {
    fn go(bits: &[u8], k: u32, acc: u64, f: &mut dyn FnMut(u64)) {
        if k == 0 {
            f(acc);
            return;
        }
        if (bits.len() as u32) < k {
            return;
        }
        // Either the first card is in the subset or it is not.
        go(&bits[1..], k - 1, acc | 1 << bits[0], f);
        go(&bits[1..], k, acc, f);
    }
    let bits: Vec<u8> = mask::bits(pool).collect();
    go(&bits, k, 0, f);
}

/// Everything that determines the future of a position (the history does not).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PositionKey {
    hands: [u64; NUM_PLAYERS],
    current: u8,
    trick: Option<(u8, u64)>,
    passes: u8,
    locked: u8,
    opening: bool,
}

impl Transposable for State {
    type Key = PositionKey;

    fn position_key(&self) -> PositionKey {
        PositionKey {
            hands: self.hands,
            current: self.current,
            trick: self.trick.map(|(p, c)| (p, c.cards)),
            passes: self.passes,
            locked: self.locked,
            opening: self.opening,
        }
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Big Two ({:?})", self.rules)?;
        for p in 0..NUM_PLAYERS {
            let marker = if self.winner.is_none() && p == self.current as usize {
                ">"
            } else {
                " "
            };
            writeln!(
                f,
                "{marker} P{p} [{:2}] {}",
                self.hands[p].count_ones(),
                format_mask(self.hands[p])
            )?;
        }
        match &self.trick {
            Some((p, c)) => writeln!(f, "  current play: {c:?} by P{p}")?,
            None => writeln!(f, "  leading")?,
        }
        if let Some(w) = self.winner {
            writeln!(f, "  winner: P{w}")?;
        }
        Ok(())
    }
}
