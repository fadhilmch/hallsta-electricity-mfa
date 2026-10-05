import numpy as np
import pytest

from hallsta_mfa.model import BASE, DIST, PROCS, f_sens, run, steam_v


@pytest.fixture(scope="module")
def b():
    return run(BASE)


def test_exergy_factor_is_zero_at_dead_state_and_grows_with_temperature():
    assert f_sens(7.0, 7.0) == 0.0
    assert 0 < f_sens(35, 7) < f_sens(60, 7) < f_sens(90, 7) < 1


def test_steam_saturation_energy_is_plausible():
    dh, fx = steam_v(3.4e5, 90.0, 7.0)
    assert 0.6 < dh < 0.7          # MWh per tonne, about 2.2 GJ/t
    assert 0.2 < fx < 0.4


def test_electricity_balance_closes(b):
    E = BASE["E_tot"]
    uses = sum(b[k] for k in ["el_TMP", "el_PM", "el_bleach", "el_EB", "el_wood", "el_wwtp", "el_util", "el_dist"])
    assert uses == pytest.approx(E)


def test_steam_balance_closes(b):
    assert b["st_supply"] == pytest.approx(b["st_PM"] + b["st_wood"] + b["st_loss"] + b["st_HW"])


def test_sinks_add_up_to_input(b):
    assert BASE["E_tot"] == pytest.approx(b["sink_water"] + b["sink_DH"] + b["sink_air"])


def test_exergy_balance_per_process(b):
    for p in PROCS:
        assert b[f"Xin|{p}"] == pytest.approx(b[f"Xuse|{p}"] + b[f"Xwaste|{p}"] + b[f"Xdest|{p}"])


def test_headline_numbers_are_unchanged(b):
    assert b["el_TMP"] == pytest.approx(1015.0)
    assert b["el_EB"] == pytest.approx(145.0)
    assert b["rec_ratio"] == pytest.approx(0.77, abs=0.01)


def test_run_accepts_arrays_and_keeps_balance():
    rng = np.random.default_rng(0)
    P = dict(BASE)
    P["sh_TMP"] = rng.uniform(0.6, 0.75, 50)
    r = run(P)
    sums = sum(r[k] for k in ["el_TMP", "el_PM", "el_bleach", "el_EB", "el_wood", "el_wwtp", "el_util", "el_dist"])
    assert np.allclose(sums, BASE["E_tot"])


def test_every_range_contains_its_base_value():
    for k, (d, a, bb, c) in DIST.items():
        if d == "tri":
            assert a <= BASE[k] <= c, k
