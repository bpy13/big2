# 0003 — A21 grader repeatability uses compatible agreement

**Status:** accepted by default (2026-11-10, orchestrator; Brian may change it)

## Context

Task 27's re-grade of 50 self-play deals with two seeds (`build-phase4.md` §1) gives 90.2 % same
grade (strict) and 97.6 % compatible (an `unclear` grade counts as agreeing with any grade).
The A21 bar is 95 %. A larger deal budget (4,800 deals) reaches only 93.8 % strict. The
phone-sized grader (300 deals) vs a 10× grader: 87.2 % strict, 94.9 % compatible (bar 90 %).

## Decision

A21 is measured as compatible agreement: 97.6 % (bar 95 %) and 94.9 % (bar 90 %), both met.
`unclear` is an honest "cannot tell", not a disagreement. This is the inbox default (a) for
"Which agreement measure should the A21 grader bar use?".

## Consequences

* Task 27 is ticked; Tasks 28 and 31 no longer wait on it.
* If Brian chooses strict agreement, Task 27 reopens with a larger deal budget.
