# Download provenance

Downloads were first verified under `/tmp/spire-unification-deps`. Validation copies were then placed in ignored `combat_solver_cli/artifacts/dependencies/`, and local `artifacts/config.json` was generated. No Steam files were changed; third-party binaries are not committed.

## CombatSolver

- Version: 0.44.0; source tag `v0.44.0`, commit `42e0902874f5f3c329ef13dccf30eb694375ca17`.
- Official release: https://github.com/Torch1230/CombatSolver/releases/tag/v0.44.0
- Asset: https://github.com/Torch1230/CombatSolver/releases/download/v0.44.0/CombatSolver-0.44.0.zip
- Release-page SHA-256 and downloaded archive SHA-256: `e210edcd8b728ed05a5791fea3fe99d0567c3ab90e53ee2534dcb5e823362210` (match).
- Extracted under `CombatSolver-0.44.0/`.
- `CombatSolver.dll`: `e9bacc377a830ce8640693fe50238cdb803cd73847787d8c6039929eadba5d51`.
- `CombatSolver.json`: `48dd412c308b354c6a8599c271326179dab69009665d1b215a112c61d4414fd8`.
- Manifest declares solver 0.44.0, minimum game 0.111.0, RitsuLib minimum 0.6.0.

## STS2-RitsuLib

- Version: 0.6.2, official stable release tag `v0.6.2`.
- Official source/release: https://github.com/BAKAOLC/STS2-RitsuLib/releases/tag/v0.6.2
- Asset: https://github.com/BAKAOLC/STS2-RitsuLib/releases/download/v0.6.2/STS2-RitsuLib.0.6.2.variant-pack.zip
- Downloaded archive SHA-256: `89188e58de017245605295459f5117c70100b6e51a356977ab24b1999ec82b1d`.
- Extracted under `STS2-RitsuLib-0.6.2/`; archive contains `compat/0.111.0/` and `shared/`.
- `compat/0.111.0/STS2-RitsuLib.dll`: `a0f159a722519edfb566b84f19017bba2badd62fc444df4818178642f736d325`.
- `compat/0.111.0/STS2-RitsuLib.Runtime.dll`: `338cbe9b434f1c0aa25907bfae9931605398d37d03a532be6c3811e32520b81c`.
- `shared/STS2-RitsuLib.Shared.dll`: `067836e4fc4ba35419d95ce4181f0e21674f6005bc58accc85eaf76a92e44a09`.
- `shared/STS2-RitsuLib.Ui.dll`: `c67eb41e62b199a503e5945da33e1715de45b025bebff6b33ee390a11931cccc`.
- `shared/STS2-RitsuLib.Settings.dll`: `f3bcb5dd5c57a932a66a63a220bb789b3961cc484a4c69fe91c650aab5fe5147`.
- Archive paths were checked for absolute/traversal entries before extraction.
