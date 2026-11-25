"""调度承诺的持久化、幂等提交与情景容量核对。"""
from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .workspace import Workspace


class CommitmentError(ValueError):
    pass


class CapacityError(CommitmentError):
    pass


class ImmutableCommitmentError(CommitmentError):
    pass


def scenarios_overlap(a: dict[str, int], b: dict[str, int]) -> bool:
    if not a or not b:
        return True
    if "*" in a or "*" in b:
        return True
    return any(key in b and int(a[key]) > 0 and int(b[key]) > 0 for key in a)


def source_unit_holds(ws: Workspace) -> dict[str, list[dict[str, Any]]]:
    """资料中 held_for 表示已被指定组单持有的位置。"""
    holds: dict[str, list[dict[str, Any]]] = {}
    for spare_id, spare in ws.spares.items():
        if not spare.get("held_for_s"):
            continue
        # 当前交换资料每个 held_for 舱位只有一个位置；若 places>1，也只锁定一个。
        holds.setdefault(spare_id, []).append(
            {
                "unit_id": f"{spare_id}#1",
                "truck_id": spare["held_for_s"],
                "occupancy_by_scenario": {"*": 1},
                "source": "spares.held_for",
            }
        )
    return holds


def assign_units(claims: list[dict[str, Any]], ws: Workspace) -> list[dict[str, Any]]:
    """按情景冲突把每个 qty=1 的承诺落到具体备用位置。"""
    result = deepcopy(claims)
    source_holds = source_unit_holds(ws)
    assigned_by_spare: dict[str, list[dict[str, Any]]] = {}

    for claim in sorted(
        result,
        key=lambda c: (
            c["spare_id"],
            0 if c.get("assigned_unit_id") else 1,
            c.get("decision_deadline") or 0,
            c["claim_key"],
        ),
    ):
        spare_id = claim["spare_id"]
        spare = ws.spares[spare_id]
        truck_id = claim["truck_id"]
        used = assigned_by_spare.setdefault(spare_id, [])
        prescribed_unit = claim.get("assigned_unit_id")

        unit_id = None
        indices = (
            [int(prescribed_unit.rsplit("#", 1)[1])]
            if prescribed_unit and prescribed_unit.startswith(f"{spare_id}#")
            else range(1, spare["places_i"] + 1)
        )
        for index in indices:
            candidate_unit = f"{spare_id}#{index}"
            source_hold = next(
                (h for h in source_holds.get(spare_id, []) if h["unit_id"] == candidate_unit),
                None,
            )
            if source_hold is not None and source_hold["truck_id"] != truck_id:
                continue
            conflict = any(
                u["unit_id"] == candidate_unit
                and scenarios_overlap(
                    claim.get("occupancy_by_scenario", {}),
                    u.get("occupancy_by_scenario", {}),
                )
                for u in used
            )
            if not conflict:
                unit_id = candidate_unit
                break

        if unit_id is None:
            raise CapacityError(
                f"{spare_id} 没有可在同一联合坏情景下同时分配给 {truck_id}/{claim.get('lot_id')} 的位置"
            )
        claim["assigned_unit_id"] = unit_id

        used.append(
            {
                "unit_id": claim["assigned_unit_id"],
                "claim_key": claim["claim_key"],
                "truck_id": truck_id,
                "occupancy_by_scenario": claim.get("occupancy_by_scenario", {}),
            }
        )
    return result


def validate_capacity(claims: list[dict[str, Any]], ws: Workspace) -> dict[str, Any]:
    """核对情景容量；返回每个 spare/scenario 的占用台账。"""
    active_statuses = {
        "reserved",
        "confirmed",
        "conditional_reserved",
        "activated",
        "fulfilled",
    }
    active = [c for c in claims if c.get("status", "reserved") in active_statuses]
    source_holds = source_unit_holds(ws)
    ledger: dict[str, int] = {}
    unit_maps: dict[str, list[dict[str, Any]]] = {}
    for claim in active:
        spare_id = claim["spare_id"]
        if spare_id not in ws.spares:
            raise CapacityError(f"未知备用舱 {spare_id}")
        for hold in source_holds.get(spare_id, []):
            same_unit = claim.get("assigned_unit_id") in (None, hold["unit_id"])
            only_one_place = ws.spares[spare_id]["places_i"] == 1
            if hold["truck_id"] != claim.get("truck_id") and (same_unit or only_one_place):
                raise CapacityError(
                    f"{spare_id} 已 held_for {hold['truck_id']}，不能核对给 {claim.get('truck_id')}"
                )
        qty = int(claim.get("quantity", 1))
        scenarios = claim.get("occupancy_by_scenario") or {"*": qty}
        for scenario, amount in scenarios.items():
            if scenario == "*":
                continue
            key = f"{spare_id}:{scenario}"
            ledger[key] = ledger.get(key, 0) + int(amount) * qty
        if "*" in scenarios:
            ledger[f"{spare_id}:*"] = ledger.get(f"{spare_id}:*", 0) + int(scenarios["*"]) * qty
        unit_maps.setdefault(spare_id, []).append(claim)

    for key, used_qty in ledger.items():
        spare_id, scenario = key.split(":", 1)
        places = ws.spares[spare_id]["places_i"]
        if used_qty > places:
            raise CapacityError(f"情景 {scenario} 下 {spare_id} 占用 {used_qty} > {places}")

    max_occupancy_by_spare: dict[str, int] = {}
    for key, used_qty in ledger.items():
        spare_id = key.split(":", 1)[0]
        max_occupancy_by_spare[spare_id] = max(max_occupancy_by_spare.get(spare_id, 0), used_qty)

    # 同一 unit 不能在同一具名情景下被两个 positive claim 占用。
    for spare_id, claims_for_spare in unit_maps.items():
        by_unit: dict[str, list[dict[str, Any]]] = {}
        for claim in claims_for_spare:
            unit_id = claim.get("assigned_unit_id") or f"{spare_id}#unassigned"
            by_unit.setdefault(unit_id, []).append(claim)
        for unit_id, unit_claims in by_unit.items():
            for i, left in enumerate(unit_claims):
                for right in unit_claims[i + 1 :]:
                    if scenarios_overlap(
                        left.get("occupancy_by_scenario", {}),
                        right.get("occupancy_by_scenario", {}),
                    ):
                        raise CapacityError(
                            f"{unit_id} 在重叠坏情景下被同时许给 "
                            f"{left.get('truck_id')}/{left.get('lot_id')} 与 "
                            f"{right.get('truck_id')}/{right.get('lot_id')}"
                        )

    return {
        "scenario_ledger": ledger,
        "max_scenario_occupancy_by_spare": max_occupancy_by_spare,
        "source_holds": source_holds,
        "active_claim_count": len(active),
        "interpretation": (
            "同一具名坏情景的占用相加；互斥情景分别记账，因此不会被错误地当作永久双倍占用"
        ),
    }


class CommitmentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, Any] = {"version": 1, "commitments": {}, "history": []}
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if loaded.get("version") != 1:
                raise CommitmentError("不支持的承诺台账版本")
            self.data = loaded

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".commitments-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o644)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()

    @staticmethod
    def key_for(ship_date: str, truck_id: str) -> str:
        return f"{ship_date}:{truck_id}"

    def get(self, ship_date: str, truck_id: str) -> dict[str, Any] | None:
        record = self.data["commitments"].get(self.key_for(ship_date, truck_id))
        return deepcopy(record) if record is not None else None

    def list_date(self, ship_date: str) -> list[dict[str, Any]]:
        prefix = ship_date + ":"
        return [deepcopy(v) for k, v in sorted(self.data["commitments"].items()) if k.startswith(prefix)]

    def upsert(self, submission: dict[str, Any], ws: Workspace) -> tuple[dict[str, Any], str]:
        """重复提交同一 key 时幂等；变更时保护已履行/已装车动作。"""
        ship_date = submission["ship_date"]
        truck_id = submission["truck_id"]
        key = self.key_for(ship_date, truck_id)
        now = datetime.now(timezone.utc).isoformat()
        candidate = deepcopy(submission)
        candidate["key"] = key
        candidate.setdefault("updated_at", now)
        candidate["submission_count"] = 1

        old = self.data["commitments"].get(key)
        if old is not None and semantic_signature(old) == semantic_signature(candidate):
            return old, "duplicate"
        if old is not None:
            self._assert_immutable_actions(old, candidate)
            candidate["created_at"] = old.get("created_at", now)
            candidate["updated_at"] = now
            candidate["revision"] = int(old.get("revision", 1)) + 1
            candidate["submission_count"] = int(old.get("submission_count", 1)) + 1
            self.data["history"].append({"key": key, "replaced_at": now, "record": deepcopy(old)})
            change = "updated_unfulfilled_parts"
        else:
            candidate["created_at"] = now
            candidate["revision"] = 1
            change = "created"

        date_claims = self._date_claim_candidates(ship_date, candidate)
        assigned = assign_units(date_claims, ws)
        unit_by_key = {claim["claim_key"]: claim.get("assigned_unit_id") for claim in assigned}
        for action in candidate.get("actions", []):
            if action.get("kind") == "spare_commitment" and action.get("claim_key") in unit_by_key:
                action["assigned_unit_id"] = unit_by_key[action["claim_key"]]
        capacity = validate_capacity(assigned, ws)
        candidate["capacity_check"] = capacity
        self.data["commitments"][key] = candidate
        self.save()
        return candidate, change

    def _date_claim_candidates(self, ship_date: str, current: dict[str, Any]) -> list[dict[str, Any]]:
        claims: list[dict[str, Any]] = []
        for record in self.list_date(ship_date):
            if record["key"] == current["key"]:
                continue
            claims.extend(a for a in record.get("actions", []) if a.get("kind") == "spare_commitment")
        claims.extend(a for a in current.get("actions", []) if a.get("kind") == "spare_commitment")
        return claims

    @staticmethod
    def _assert_immutable_actions(old: dict[str, Any], new: dict[str, Any]) -> None:
        old_actions = {a["action_id"]: a for a in old.get("actions", [])}
        new_actions = {a["action_id"]: a for a in new.get("actions", [])}
        core_fields = ("kind", "lot_id", "spare_id", "assigned_unit_id")
        for action_id, old_action in old_actions.items():
            locked = old_action.get("immutable") or old_action.get("status") == "fulfilled"
            if not locked:
                continue
            new_action = new_actions.get(action_id)
            if new_action is None:
                raise ImmutableCommitmentError(f"已履行/已装车动作 {action_id} 不能删除")
            for field in core_fields:
                if old_action.get(field) != new_action.get(field):
                    raise ImmutableCommitmentError(
                        f"已履行/已装车动作 {action_id} 的 {field} 不能从 "
                        f"{old_action.get(field)!r} 改为 {new_action.get(field)!r}"
                    )

    def verify_date(self, ship_date: str, ws: Workspace) -> dict[str, Any]:
        records = self.list_date(ship_date)
        claims = [a for r in records for a in r.get("actions", []) if a.get("kind") == "spare_commitment"]
        capacity = validate_capacity(claims, ws)
        return {
            "ship_date": ship_date,
            "submission_count": len(records),
            "truck_ids": [r["truck_id"] for r in records],
            "capacity": capacity,
            "commitments": records,
        }


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_signature(record: dict[str, Any]) -> str:
    """忽略提交时间/revision，使同一日同一组单重复提交保持幂等。"""
    ignored = {"created_at", "updated_at", "revision", "submission_count", "capacity_check"}
    payload = {k: v for k, v in record.items() if k not in ignored}
    return canonical(payload)
