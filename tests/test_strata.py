"""Unit tests for the Strata project store and the INPUT allow-list (no Norne data needed)."""

import pytest

from strata.config import input_path
from strata.store import Store, Verdict


def test_publish_versions_and_supersede(tmp_path):
    s = Store("t1", root=tmp_path)
    w = lambda d: [(d / "a.txt").write_text("x"), "a.txt"][1:]
    m1 = s.publish("HorizonPicks", "H", "horizon_agent", w, summary="first")
    m2 = s.publish("HorizonPicks", "H", "horizon_agent", w, summary="second")
    assert (m1.version, m2.version) == (1, 2)
    assert s.get("HorizonPicks", "H", 1).status == "superseded"
    assert s.get("HorizonPicks", "H").version == 2
    assert s.path(m2, "a.txt").read_text() == "x"


def test_verdict_sets_status_and_is_logged(tmp_path):
    s = Store("t2", root=tmp_path)
    m = s.publish("HorizonPicks", "H", "horizon_agent")
    m = s.add_verdict(m, Verdict(verifier="well_agent", passed=False, summary="MAE 19 ms"))
    assert m.status == "rejected" and m.verdicts[0].summary == "MAE 19 ms"
    kinds = [e["kind"] for e in s.events()]
    assert kinds == ["publish", "verdict"]


@pytest.mark.parametrize("rel", ["horisons/Top_Aare_2003_Time.txt", "FAULTS/kseim_r06_faults_depth",
                                 "reservoir_models/Norne_ATW_2013/NORNE_ATW2013.DATA",
                                 "time-depth-conversion_tdq/tdq_intVel_norne_6layers", "wells/../horisons/x"])
def test_truth_paths_are_refused(rel):
    with pytest.raises(PermissionError):
        input_path(rel)


def test_input_paths_are_allowed():
    assert input_path("seismics/norne4d_2001-full.sgy").name == "norne4d_2001-full.sgy"
    assert input_path("wells/time-depth").name == "time-depth"
