# Native replay fixtures

These JSON files contain action histories produced by this module from native A10 runs, with a public-state fingerprint before each action. They contain no game or CombatSolver binaries.

- `defect_refinement.json`: immediately before a CombatSolver 0.44.0 refinement failure involving Buffer. The solver has already published a complete winning combat route. The regression checks that this route remains executable through native candidates.
- `test_subject_phase_change.json`: immediately before the native action that applies TestSubject's phase-two visual change. The regression checks that the headless engine reaches a boundary instead of stalling on the missing `CanvasItem.SetSelfModulate` method.

Both tests require the locally configured, pinned game and solver dependencies. They do not prove a full-run victory and are never exported as accepted training data.

The Crusher regression enters `KAISER_CRAB_BOSS` directly through the debug API. Debug frames are explicitly excluded from training.
