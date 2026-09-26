# VTcmAccounting — the claim ledger

`boundary_matrix.json` is the ledger of what the VTCM byte accounting has and has
not proven. It lives here, inside the shipped repository, rather than beside the
probe that grew it, for two reasons that are worth keeping:

* **Committed tests assert on it.** `bin/runtime/test/test_vtcm_accounting_identity_contract.py`
  and `test_vtcm_layer_b_contract.py` both read cells out of it, so it is a
  dependency of the repository's own test suite. A dependency of the deliverable
  cannot live outside the deliverable: a fresh clone would fail before it ran a
  single test.
* **It is a declaration about the product, not about the scaffolding.** Each cell
  states a claim scope and whether that claim is `pass`, `fail` or `not-proven`.
  The probe in `exp/hmx/vtcm_accounting_probe/` is the thing that *checks* those
  claims; this file is the *declaration* of them, and a declaration is part of
  the project.

## What the statuses mean

`pass` is deliberately narrow: the cell's own `claim_scope` says how far the
evidence reaches, and nothing more. `not-proven` is the normal state for most
axes and is not a defect — the ledger exists to make the unknown explicit rather
than to claim completeness. `fail` means a claim was made and refuted.

A `device-observation` cell records a device observation *beside* a frozen
`not-proven` cell by design. Importing device evidence can therefore never
promote a cell, and an unchanged pass/not-proven split after an import is the
expected outcome, not a sign the import did nothing.

## Who reads it

* `qcom_hexagon_backend/bin/runtime/test/test_vtcm_accounting_identity_contract.py`
* `qcom_hexagon_backend/bin/runtime/test/test_vtcm_layer_b_contract.py`
* `exp/hmx/vtcm_accounting_probe/matrix_runner.py` (the evaluator; defaults to
  this copy, and `--matrix` can point elsewhere)

`matrix_runner.py` is the only thing that *computes* a verdict from this file; the
repo tests read the declared cells. When a cell's status must change, change it
here and let the runner's own contract tests object if the change was not
justified — do not hand-edit a status to make a test pass.
