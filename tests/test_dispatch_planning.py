from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from dispatch_planning.commitments import (
    CapacityError,
    CommitmentStore,
    ImmutableCommitmentError,
    assign_units,
    validate_capacity,
)
from dispatch_planning.planning import build_plan, evaluate_truck
from dispatch_planning.probability import (
    frechet_bounds,
    gaussian_copula_all,
    moisture_variance_components,
    estimate_outcome_icc,
)
from dispatch_planning.service import build_submission, submit_day
from dispatch_planning.validation import grouped_split_report
from dispatch_planning.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class PointInTimeTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace(DATA)

    def test_received_time_controls_visibility(self):
        late = self.ws.latest_check("L09", 26)
        self.assertEqual((late.observed, late.percent, late.received), (12, 12.2, 12))
        updated = self.ws.latest_check("L09", 28)
        self.assertEqual((updated.observed, updated.percent, updated.received), (24, 8.9, 27))

        l12_at_28 = self.ws.latest_check("L12", 28)
        self.assertEqual(l12_at_28.percent, 9.5)
        l12_at_29 = self.ws.latest_check("L12", 29)
        self.assertEqual((l12_at_29.percent, l12_at_29.received), (10.9, 29))

    def test_open_is_censored_not_failure(self):
        status = self.ws.lot_status_at("L09", 29)
        self.assertEqual(status["state_at_as_of"], "open")
        self.assertIsNone(status["countable_success"])
        self.assertTrue(status["censored"])

    def test_scrap_and_transfer_are_distinct(self):
        scrap = self.ws.lot_status_at("L04", 23)
        self.assertEqual(scrap["state_at_as_of"], "scrap")
        self.assertFalse(scrap["countable_success"])
        self.assertFalse(scrap["censored"])

        before_transfer = self.ws.lot_status_at("L07", 23)
        self.assertEqual(before_transfer["state_at_as_of"], "open")
        transferred = self.ws.lot_status_at("L07", 24)
        self.assertEqual(transferred["state_at_as_of"], "transferred")
        self.assertIsNone(transferred["countable_success"])
        self.assertTrue(transferred["censored"])
        self.assertFalse(transferred["available_for_original_group"])


class ProbabilityAndDependenceTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace(DATA)

    def test_frechet_bounds_do_not_assume_independence(self):
        lower, upper = frechet_bounds([0.9, 0.9, 0.9])
        self.assertAlmostEqual(lower, 0.7)
        self.assertAlmostEqual(upper, 0.9)
        product = 0.9 ** 3
        self.assertAlmostEqual(product, 0.729)
        self.assertLess(lower, product)
        self.assertLess(product, upper)
        copula = gaussian_copula_all([0.9, 0.8], 0.62)
        self.assertGreaterEqual(copula, 0.72 - 1e-9)
        self.assertLessEqual(copula + 1e-9, 0.8)

    def test_active_groups_do_not_meet_conservative_thresholds(self):
        for truck_id in ("G05", "G06", "G07"):
            evaluation = evaluate_truck(self.ws, truck_id, 29)
            if truck_id == "G06":
                self.assertEqual(evaluation["recommended_ship_lot_ids"], ["L11"])
            elif truck_id == "G07":
                self.assertEqual(evaluation["recommended_ship_lot_ids"], ["L14"])
            else:
                self.assertEqual(evaluation["recommended_ship_lot_ids"], [])
            self.assertFalse(evaluation["full_joint"]["meets_required_conservatively"])

    def test_common_cause_variance_is_material(self):
        variance = moisture_variance_components(self.ws, 24)
        self.assertGreater(variance["icc_moisture"], 0.5)
        icc = estimate_outcome_icc(self.ws)
        self.assertGreater(icc["outcome_icc_1_over_1_plus_phi"], 0.4)

    def test_feed_grouped_validation_has_no_overlap(self):
        report = grouped_split_report(self.ws)
        self.assertTrue(report["feed_grouped_split_valid"])
        self.assertEqual(report["feed_overlap"], [])


class SpareAllocationTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace(DATA)
        self.plan = build_plan(self.ws, 25)
        self.claims = {c["claim_key"]: c for c in self.plan["claims"]}

    def test_f5_orders_get_distinct_units_and_correct_timing(self):
        g05 = self.claims["G05:L09:Z01"]
        g07 = self.claims["G07:L13:Z01"]
        self.assertEqual(g05["spare_id"], "Z01")
        self.assertEqual(g07["spare_id"], "Z01")
        self.assertNotEqual(g05["assigned_unit_id"], g07["assigned_unit_id"])

        g05_z03 = self.claims["G05:L10:Z03"]
        self.assertEqual(g05_z03["assigned_unit_id"], "Z03#1")
        g06 = self.claims["G06:L12:Z02"]
        self.assertEqual(g06["assigned_unit_id"], "Z02#1")

        self.assertEqual(g05["decision_deadline"], 26)
        self.assertGreater(g05["next_reading_at"], g05["decision_deadline"])
        self.assertTrue(g05["decision_type"].startswith("decide_on_visible"))
        self.assertTrue(g06["next_reading_visible_by_deadline"])
        self.assertTrue(g05_z03["next_reading_visible_by_deadline"])

    def test_loaded_lot_is_never_sent_to_spare(self):
        spare_lots = {c["lot_id"] for c in self.plan["claims"]}
        self.assertNotIn("L11", spare_lots)

    def test_deteriorated_loaded_lot_is_immutable_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "data"
            shutil.copytree(DATA, copied)
            checks = copied / "checks.csv"
            checks.write_text(checks.read_text(encoding="utf-8").replace("L11,24,24,7.6", "L11,24,24,16.0"), encoding="utf-8")
            ws = Workspace(copied)
            submission = build_submission(ws, "G06", "2026-11-25", 29, confirm=True)
            loaded_action = next(a for a in submission["actions"] if a["action_id"] == "ship:G06:L11")
            self.assertTrue(loaded_action["immutable"])
            self.assertEqual(loaded_action["status"], "exception_immutable_loaded")
            spare_lots = {a.get("lot_id") for a in submission["actions"] if a["kind"] == "spare_commitment"}
            self.assertNotIn("L11", spare_lots)

    def test_same_unit_rejected_for_shared_bad_scenario(self):
        claims = [
            {
                "claim_key": "A",
                "spare_id": "Z01",
                "assigned_unit_id": "Z01#9",
                "quantity": 1,
                "status": "reserved",
                "truck_id": "A",
                "occupancy_by_scenario": {"F5_BAD": 1},
            },
            {
                "claim_key": "B",
                "spare_id": "Z01",
                "assigned_unit_id": "Z01#9",
                "quantity": 1,
                "status": "reserved",
                "truck_id": "B",
                "occupancy_by_scenario": {"F5_BAD": 1},
            },
        ]
        with self.assertRaises(CapacityError):
            validate_capacity(claims, self.ws)

    def test_mutually_exclusive_conditions_may_share_unit(self):
        claims = [
            {
                "claim_key": "A",
                "spare_id": "Z02",
                "assigned_unit_id": "Z02#1",
                "quantity": 1,
                "status": "conditional_reserved",
                "truck_id": "A",
                "occupancy_by_scenario": {"SCENARIO_A_ONLY": 1},
            },
            {
                "claim_key": "B",
                "spare_id": "Z02",
                "assigned_unit_id": "Z02#1",
                "quantity": 1,
                "status": "conditional_reserved",
                "truck_id": "B",
                "occupancy_by_scenario": {"SCENARIO_B_ONLY": 1},
            },
        ]
        result = validate_capacity(claims, self.ws)
        self.assertEqual(result["active_claim_count"], 2)
        self.assertEqual(result["max_scenario_occupancy_by_spare"]["Z02"], 1)

    def test_held_spare_cannot_be_allocated_to_other_truck(self):
        claim = {
            "claim_key": "G07:L13:Z03",
            "spare_id": "Z03",
            "truck_id": "G07",
            "lot_id": "L13",
            "quantity": 1,
            "occupancy_by_scenario": {"feed:F5:bad": 1},
        }
        with self.assertRaises(CapacityError):
            assign_units([claim], self.ws)


class CommitmentPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace(DATA)
        self.temp = tempfile.TemporaryDirectory()
        self.store = CommitmentStore(Path(self.temp.name) / "ledger.json")

    def tearDown(self):
        self.temp.cleanup()

    def test_repeat_submission_is_idempotent_and_reloadable(self):
        first = submit_day(self.ws, self.store, "2026-11-25", 25, confirm=True)
        self.assertEqual({r["change"] for r in first["results"]}, {"created"})
        self.assertEqual(first["verification"]["submission_count"], 3)
        self.assertEqual(first["verification"]["capacity"]["active_claim_count"], 4)

        second = submit_day(self.ws, self.store, "2026-11-25", 25, confirm=True)
        self.assertEqual({r["change"] for r in second["results"]}, {"duplicate"})

        reloaded = CommitmentStore(self.store.path)
        verified = reloaded.verify_date("2026-11-25", self.ws)
        self.assertEqual(verified["submission_count"], 3)
        self.assertEqual(verified["capacity"]["active_claim_count"], 4)

    def test_reading_update_preserves_units_but_changes_revision(self):
        submit_day(self.ws, self.store, "2026-11-25", 25, confirm=True)
        before = self.store.get("2026-11-25", "G07")
        unit_before = next(
            a["assigned_unit_id"]
            for a in before["actions"]
            if a["action_id"] == "spare:G07:L13:Z01"
        )
        updated = submit_day(self.ws, self.store, "2026-11-25", 29, confirm=True)
        self.assertTrue(all(r["change"] == "updated_unfulfilled_parts" for r in updated["results"]))
        after = self.store.get("2026-11-25", "G07")
        unit_after = next(
            a["assigned_unit_id"]
            for a in after["actions"]
            if a["action_id"] == "spare:G07:L13:Z01"
        )
        self.assertEqual(unit_before, unit_after)
        self.assertEqual(after["revision"], 2)

        repeated = submit_day(self.ws, self.store, "2026-11-25", 29, confirm=True)
        self.assertTrue(all(r["change"] == "duplicate" for r in repeated["results"]))

    def test_loaded_action_is_immutable(self):
        submit_day(self.ws, self.store, "2026-11-25", 25, confirm=True)
        record = self.store.get("2026-11-25", "G06")
        self.assertTrue(any(a.get("immutable") and a.get("lot_id") == "L11" for a in record["actions"]))
        record["actions"] = [a for a in record["actions"] if a["action_id"] != "ship:G06:L11"]
        with self.assertRaises(ImmutableCommitmentError):
            self.store.upsert(record, self.ws)


if __name__ == "__main__":
    unittest.main()
