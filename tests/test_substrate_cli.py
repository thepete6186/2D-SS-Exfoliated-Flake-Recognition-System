"""Tests for the substrate-override CLI helpers in run_semi_supervised.py."""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "hsv-pipeline-semi"))

import run_semi_supervised as rss  # noqa: E402


def write_json(tmp_path, data):
    p = tmp_path / "points.json"
    p.write_text(json.dumps(data))
    return p


def make_args(substrate_hsv=None, substrate_json=None, substrate_std=None):
    return Namespace(
        substrate_hsv=substrate_hsv,
        substrate_json=substrate_json,
        substrate_std=substrate_std,
    )


class TestParseHsvTriplet:
    def test_valid(self):
        assert rss.parse_hsv_triplet("90,40,180") == (90.0, 40.0, 180.0)
        assert rss.parse_hsv_triplet(" 90.5, 40 ,180 ") == (90.5, 40.0, 180.0)

    def test_malformed(self):
        with pytest.raises(ValueError):
            rss.parse_hsv_triplet("90,40")
        with pytest.raises(ValueError):
            rss.parse_hsv_triplet("a,b,c")


class TestLoadSubstrateFromJson:
    def test_prefers_calibration_block(self, tmp_path):
        p = write_json(tmp_path, {
            "substrate_hsv": [90, 50, 100],
            "substrate_calibration": {
                "peak": [91.2, 51.5, 101.9],
                "std": [1.5, 2.5, 3.5],
                "n_pixels": 450,
                "n_samples": 2,
            },
        })

        peak, std = rss.load_substrate_from_json(p)

        assert peak == pytest.approx((91.2, 51.5, 101.9))
        assert std == pytest.approx((1.5, 2.5, 3.5))

    def test_calibration_block_with_null_std(self, tmp_path):
        p = write_json(tmp_path, {
            "substrate_calibration": {
                "peak": [91.2, 51.5, 101.9], "std": None,
                "n_pixels": 9, "n_samples": 1,
            },
        })

        peak, std = rss.load_substrate_from_json(p)

        assert peak == pytest.approx((91.2, 51.5, 101.9))
        assert std is None

    def test_legacy_fallback(self, tmp_path):
        p = write_json(tmp_path, {"substrate_hsv": [90, 50, 100]})

        peak, std = rss.load_substrate_from_json(p)

        assert peak == (90.0, 50.0, 100.0)
        assert std is None

    def test_neither_key(self, tmp_path):
        p = write_json(tmp_path, {"points": []})

        assert rss.load_substrate_from_json(p) == (None, None)


class TestResolveSubstrateOverride:
    def test_nothing_given(self):
        assert rss.resolve_substrate_override(make_args()) == (None, None)

    def test_json_only(self, tmp_path):
        p = write_json(tmp_path, {
            "substrate_calibration": {
                "peak": [91.0, 51.0, 101.0], "std": [1.0, 2.0, 3.0],
                "n_pixels": 450, "n_samples": 2,
            },
        })

        peak, std = rss.resolve_substrate_override(
            make_args(substrate_json=str(p))
        )

        assert peak == (91.0, 51.0, 101.0)
        assert std == (1.0, 2.0, 3.0)

    def test_hsv_beats_json(self, tmp_path):
        p = write_json(tmp_path, {
            "substrate_calibration": {
                "peak": [91.0, 51.0, 101.0], "std": [1.0, 2.0, 3.0],
                "n_pixels": 450, "n_samples": 2,
            },
        })

        peak, std = rss.resolve_substrate_override(
            make_args(substrate_hsv="90,40,180", substrate_json=str(p))
        )

        assert peak == (90.0, 40.0, 180.0)
        # The json std was measured around the json peak — it must not
        # silently attach to a different, explicit peak.
        assert std is None

    def test_std_overlay(self):
        peak, std = rss.resolve_substrate_override(
            make_args(substrate_hsv="90,40,180", substrate_std="1,2,3")
        )

        assert peak == (90.0, 40.0, 180.0)
        assert std == (1.0, 2.0, 3.0)

    def test_std_alone_rejected(self):
        with pytest.raises(ValueError):
            rss.resolve_substrate_override(make_args(substrate_std="1,2,3"))
