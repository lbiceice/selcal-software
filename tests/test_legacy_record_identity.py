"""Records made before the ALG-01 Pearson numeric-method change stay verifiable.

R11 reference generation (2026-10-03): after the preprocessing identity changed, `verify`
refused every earlier record as invalid content, although the acceptance contract keeps older
records readable and refuses only their replay. Structural verification now accepts exactly the
prior identity, only for records whose recorded Pearson source differs from the current one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from selcal.workflow import WorkflowError, read_workflow, verify_record
from selcal.workflow_store import read_record, write_record

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "docs/status/evidence/windows_check/records_20260930_r6"
CAP = 8 * 1_048_576
PRIOR = b'"no_hidden_transform|formula_centered_after_max_abs_scaling|common_support_max_lag"'


def _legacy_records() -> list[Path]:
    records = sorted(LEGACY.glob("*.sqlite"))
    if not records:
        pytest.skip("legacy reference records are not shipped with this distribution")
    return records


@pytest.mark.parametrize("name", [path.name for path in sorted(LEGACY.glob("*.sqlite"))])
def test_prior_identity_records_verify_but_do_not_replay(name):
    record = LEGACY / name
    members = read_record(record, max_bytes=CAP)
    assert PRIOR in members["result"]
    before = record.read_bytes()
    workflow = read_workflow(record, max_bytes=CAP)
    assert workflow.result.p_value is None or 0 < workflow.result.p_value <= 1
    with pytest.raises(WorkflowError) as error:
        verify_record(record, max_bytes=CAP, replay_decision=True)
    assert (
        str(error.value) == "environment_mismatch" or error.value.args[0] == "environment_mismatch"
    )
    assert record.read_bytes() == before


def _rewritten(tmp_path: Path, change) -> Path:
    members = read_record(_legacy_records()[0], max_bytes=CAP)
    members = dict(members)
    change(members)
    target = tmp_path / "rewritten.sqlite"
    write_record(target, members, max_bytes=CAP)
    return target


def test_rewriting_unchanged_members_keeps_the_record_readable(tmp_path):
    """Control for the two refusals below: re-saving the same members is not itself damage."""
    assert read_workflow(_rewritten(tmp_path, lambda members: None), max_bytes=CAP)


def test_prior_identity_is_refused_when_the_record_claims_the_current_pearson_source(tmp_path):
    current = hashlib.sha256(
        (ROOT / "src/selcal/statistics/lagged_pearson.py").read_bytes()
    ).hexdigest()

    def claim_current_source(members):
        metadata = json.loads(members["metadata"])
        metadata["software"]["source_files"]["statistics/lagged_pearson.py"] = current
        members["metadata"] = json.dumps(metadata, sort_keys=True).encode()

    with pytest.raises(WorkflowError):
        read_workflow(_rewritten(tmp_path, claim_current_source), max_bytes=CAP)


def test_mixed_prior_and_current_identities_are_refused(tmp_path):
    from selcal.statistics.lagged_pearson import _PREPROCESSING_IDENTITY

    def mix(members):
        raw = members["result"]
        index = raw.index(PRIOR)
        members["result"] = (
            raw[:index] + json.dumps(_PREPROCESSING_IDENTITY).encode() + raw[index + len(PRIOR) :]
        )

    with pytest.raises(WorkflowError):
        read_workflow(_rewritten(tmp_path, mix), max_bytes=CAP)
