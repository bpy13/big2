//! `big2-app-api`: the app-facing adapter over the Big Two engine.
//!
//! Implements the calls of hk-games `docs/engine-interface.md` (v0) as JSON
//! strings in and out, so the same boundary serves WebAssembly, native and the
//! server. [`Api`] is the pure-Rust core; the `wasm32` build adds a thin
//! `wasm-bindgen` class ([`wasm::Big2Api`]) with the same calls.
//!
//! The shapes, keys and numbers are documented in `README.md`.
//!
//! With the `python` feature the same class is a Python extension module
//! ([`python::Big2Api`]), so the hk-games server runs the phone's exact rules.

pub mod api;
pub mod reasons;
pub mod shapes;
pub mod solver;

pub use api::{Api, BOT_LEVELS};
pub use shapes::API_VERSION;

/// The Python surface (feature `python`): one class, every call a method taking
/// and returning JSON strings, exactly like the WebAssembly class.
#[cfg(feature = "python")]
pub mod python {
    use pyo3::prelude::*;

    /// The engine as Python sees it. Methods mirror [`crate::Api`] one to one.
    #[pyclass(name = "Big2Api", module = "big2_app_api")]
    pub struct Big2Api {
        inner: crate::Api,
    }

    #[pymethods]
    impl Big2Api {
        #[new]
        pub fn new() -> Big2Api {
            Big2Api {
                inner: crate::Api::new(),
            }
        }

        pub fn meta(&self) -> String {
            self.inner.meta()
        }

        pub fn new_game(&mut self, config: &str) -> String {
            self.inner.new_game(config)
        }

        pub fn view(&self, game_id: &str, seat: usize) -> String {
            self.inner.view(game_id, seat)
        }

        pub fn legal_actions(&self, game_id: &str) -> String {
            self.inner.legal_actions(game_id)
        }

        pub fn apply(&mut self, game_id: &str, action: &str) -> String {
            self.inner.apply(game_id, action)
        }

        /// Releases the GIL while the search runs, so the server's other
        /// coroutines keep going during a bot's think time.
        pub fn bot_action(
            &self,
            py: Python<'_>,
            game_id: &str,
            level: &str,
            budget_ms: u32,
        ) -> String {
            py.detach(|| self.inner.bot_action(game_id, level, budget_ms))
        }

        pub fn hint(&self, py: Python<'_>, game_id: &str, seat: usize, budget_ms: u32) -> String {
            py.detach(|| self.inner.hint(game_id, seat, budget_ms))
        }

        pub fn hint_with_iterations(
            &self,
            py: Python<'_>,
            game_id: &str,
            seat: usize,
            budget_ms: u32,
            iterations: u64,
        ) -> String {
            py.detach(|| {
                self.inner
                    .hint_with_iterations(game_id, seat, budget_ms, iterations)
            })
        }

        pub fn review(&self, py: Python<'_>, game_id: &str, seat: usize, budget_ms: u32) -> String {
            py.detach(|| self.inner.review(game_id, seat, budget_ms))
        }

        pub fn grade(
            &self,
            py: Python<'_>,
            position: &str,
            action: &str,
            budget_ms: u32,
        ) -> String {
            py.detach(|| self.inner.grade(position, action, budget_ms))
        }

        pub fn snapshot(&self, game_id: &str) -> String {
            self.inner.snapshot(game_id)
        }

        pub fn restore(&mut self, position: &str) -> String {
            self.inner.restore(position)
        }

        pub fn drop_game(&mut self, game_id: &str) -> String {
            self.inner.drop_game(game_id)
        }

        /// Number of games held.
        pub fn __len__(&self) -> usize {
            self.inner.len()
        }
    }

    impl Default for Big2Api {
        fn default() -> Self {
            Big2Api::new()
        }
    }

    /// The `big2_app_api` module: `from big2_app_api import Big2Api`.
    #[pymodule]
    fn big2_app_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<Big2Api>()?;
        m.add("API_VERSION", crate::API_VERSION)?;
        Ok(())
    }
}

/// The WebAssembly surface: one class, every call a method taking and
/// returning JSON strings.
#[cfg(target_arch = "wasm32")]
pub mod wasm {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct Big2Api {
        inner: crate::Api,
    }

    #[wasm_bindgen]
    impl Big2Api {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Big2Api {
            Big2Api {
                inner: crate::Api::new(),
            }
        }

        pub fn meta(&self) -> String {
            self.inner.meta()
        }

        pub fn new_game(&mut self, config: &str) -> String {
            self.inner.new_game(config)
        }

        pub fn view(&self, game_id: &str, seat: usize) -> String {
            self.inner.view(game_id, seat)
        }

        pub fn legal_actions(&self, game_id: &str) -> String {
            self.inner.legal_actions(game_id)
        }

        pub fn apply(&mut self, game_id: &str, action: &str) -> String {
            self.inner.apply(game_id, action)
        }

        pub fn bot_action(&self, game_id: &str, level: &str, budget_ms: u32) -> String {
            self.inner.bot_action(game_id, level, budget_ms)
        }

        pub fn hint(&self, game_id: &str, seat: usize, budget_ms: u32) -> String {
            self.inner.hint(game_id, seat, budget_ms)
        }

        pub fn hint_with_iterations(
            &self,
            game_id: &str,
            seat: usize,
            budget_ms: u32,
            iterations: u32,
        ) -> String {
            // `u32`, so JavaScript passes a plain number rather than a BigInt.
            self.inner
                .hint_with_iterations(game_id, seat, budget_ms, u64::from(iterations))
        }

        pub fn review(&self, game_id: &str, seat: usize, budget_ms: u32) -> String {
            self.inner.review(game_id, seat, budget_ms)
        }

        pub fn grade(&self, position: &str, action: &str, budget_ms: u32) -> String {
            self.inner.grade(position, action, budget_ms)
        }

        pub fn snapshot(&self, game_id: &str) -> String {
            self.inner.snapshot(game_id)
        }

        pub fn restore(&mut self, position: &str) -> String {
            self.inner.restore(position)
        }

        pub fn drop_game(&mut self, game_id: &str) -> String {
            self.inner.drop_game(game_id)
        }
    }

    impl Default for Big2Api {
        fn default() -> Self {
            Big2Api::new()
        }
    }
}
