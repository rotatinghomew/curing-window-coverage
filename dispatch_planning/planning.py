"""组单评估、抢救舱匹配与当日计划生成。"""
from __future__ import annotations

from itertools import combinations
from typing import Any

from .commitments import CapacityError, assign_units, validate_capacity
from .probability import (
    fit_outcome_model,
    joint_assessment,
    marginal_for_lot,
    node_forecasts_for_lot,
)
from .workspace import Workspace


def _deadline_marginal(
    ws: Workspace,
    lot_id: str,
    as_of: int | None,
    deadline: int,
    model: Any,
) -> dict[str, Any]:
    """参考用：把组单截止时间替换为改期/latest。"""
    original = ws.lots[lot_id]
    proxy = dict(original)
    proxy["due"] = str(deadline)
    ws.lots[lot_id] = proxy  # type: ignore[assignment]
    try:
        result = marginal_for_lot(ws, lot_id, as_of, model)
    finally:
        ws.lots[lot_id] = original
    max_historical_due = max(int(l["due"]) for l in ws.lots.values() if l["period"] in {"train", "holdout"})
    result["deadline_override"] = deadline
    result["deadline_evidence_supported"] = deadline <= max_historical_due
    result["max_historical_due"] = max_historical_due
    if deadline > max_historical_due:
        result["deadline_warning"] = (
            f"{deadline} 晚于历史最长截止 {max_historical_due}；改期收益无直接样本，不能作为保守保证"
        )
    return result


def evaluate_truck(
    ws: Workspace,
    truck_id: str,
    as_of: int | None,
    model: Any | None = None,
) -> dict[str, Any]:
    model = model or fit_outcome_model(ws)
    truck = ws.trucks[truck_id]
    lot_ids = list(truck["lot_ids"])
    marginals = [marginal_for_lot(ws, lot_id, as_of, model) for lot_id in lot_ids]
    for marginal in marginals:
        marginal["node_forecasts"] = node_forecasts_for_lot(ws, marginal["lot"], as_of, model)
    by_lot = {m["lot"]: m for m in marginals}
    full = joint_assessment(marginals, truck["required_f"])

    loaded = set(truck["loaded_ids"])
    immutable_loaded_issues = [
        lot_id
        for lot_id in truck["loaded_ids"]
        if by_lot[lot_id]["conservative_interval"][0] < truck["required_f"]
        or not by_lot[lot_id]["available"]
    ]

    subset_options: list[dict[str, Any]] = []
    candidate_subsets: list[tuple[str, ...]] = []
    if truck["split_allowed"]:
        unloaded = [x for x in lot_ids if x not in loaded]
        for size in range(len(unloaded) + 1):
            for extra in combinations(unloaded, size):
                subset = tuple(truck["loaded_ids"]) + extra
                if subset:
                    candidate_subsets.append(tuple(dict.fromkeys(subset)))
        if loaded:
            # 已装车段即使不达标也不能撤回；合同拆分仍可能允许未装车部分单独发运。
            for size in range(1, len(unloaded) + 1):
                candidate_subsets.extend(combinations(unloaded, size))
    else:
        candidate_subsets.append(tuple(lot_ids))

    for subset in sorted(set(candidate_subsets), key=lambda x: (-len(x), x)):
        ms = [by_lot[x] for x in subset]
        assessment = joint_assessment(ms, truck["required_f"])
        subset_options.append(
            {
                "lot_ids": list(subset),
                "contains_all_loaded": loaded <= set(subset),
                "joint": assessment,
                "confident": assessment["meets_required_conservatively"],
            }
        )

    confident_loaded_options = [
        x for x in subset_options if x["confident"] and x["contains_all_loaded"]
    ]
    confident_split_options = [
        x for x in subset_options if x["confident"] and not x["contains_all_loaded"]
    ]
    if confident_loaded_options:
        recommended_ship = confident_loaded_options[0]["lot_ids"]
    elif confident_split_options:
        recommended_ship = confident_split_options[0]["lot_ids"]
    else:
        recommended_ship = []

    wait_marginals = [
        _deadline_marginal(ws, lot_id, as_of, truck["latest_i"], model)
        for lot_id in lot_ids
    ]
    wait_joint = joint_assessment(wait_marginals, truck["required_f"])
    wait_supported = all(m.get("deadline_evidence_supported", False) for m in wait_marginals)

    problem_lots = [
        lot_id
        for lot_id in lot_ids
        if lot_id not in recommended_ship
        and (
            by_lot[lot_id]["conservative_interval"][0] < truck["required_f"]
            or not by_lot[lot_id]["available"]
        )
    ]

    return {
        "truck_id": truck_id,
        "period": truck["period"],
        "carrier": truck["carrier"],
        "required": truck["required_f"],
        "depart": truck["depart_i"],
        "latest": truck["latest_i"],
        "split_allowed": truck["split_allowed"],
        "loaded_ids": truck["loaded_ids"],
        "allowed_zones": truck["allowed_zone_list"],
        "marginals": marginals,
        "full_joint": full,
        "subset_options": subset_options,
        "recommended_ship_lot_ids": recommended_ship,
        "problem_lot_ids": problem_lots,
        "immutable_loaded_issues": immutable_loaded_issues,
        "wait_to_latest_reference": {
            "joint": wait_joint,
            "evidence_supported": wait_supported,
            "note": "仅为模型外推；超过历史截止范围时不得据此承诺达标",
        },
        "decision": (
            "confident_whole_group"
            if recommended_ship and set(recommended_ship) == set(lot_ids)
            else "confident_partial_split"
            if recommended_ship
            else "exception_no_conservative_plan"
        ),
    }


def spare_match_for_lot(
    ws: Workspace,
    truck_id: str,
    lot_id: str,
    as_of: int | None = None,
) -> list[dict[str, Any]]:
    truck = ws.trucks[truck_id]
    lot = ws.lots[lot_id]
    matches: list[dict[str, Any]] = []
    for spare_id, spare in ws.spares.items():
        reasons: list[str] = []
        if truck["carrier"] not in spare["carrier_list"]:
            reasons.append("carrier_not_allowed")
        if lot["zone"] not in spare["zone_list"]:
            reasons.append("zone_not_allowed")
        if lot["zone"] not in truck["allowed_zone_list"]:
            reasons.append("truck_contract_zone_not_allowed")
        if spare["held_for_s"] and spare["held_for_s"] != truck_id:
            reasons.append(f"spare_already_held_for_{spare['held_for_s']}")
        dp = ws.decision_points.get(spare_id)
        deadline = dp["last_accept_i"] if dp else spare["window_from_i"]
        next_at = dp["new_reading_at_i"] if dp else None
        future_visible = dp["new_reading_visible_by_deadline"] if dp else True
        if as_of is not None and as_of > deadline:
            reasons.append("last_accept_deadline_passed_without_prior_commitment")
        matches.append(
            {
                "spare_id": spare_id,
                "eligible": not reasons,
                "reasons": reasons,
                "places": spare["places_i"],
                "window": [spare["window_from_i"], spare["window_to_i"]],
                "decision_deadline": deadline,
                "next_reading_at": next_at,
                "next_reading_visible_by_deadline": future_visible,
                "future_condition_promisable": future_visible,
                "held_for": spare["held_for_s"],
            }
        )
    return matches


def _scenario_tags(ws: Workspace, lot_id: str) -> dict[str, int]:
    lot = ws.lots[lot_id]
    # F5 跨 G05/G07 是资料中明确存在的跨单共因；zone 标签保留温区共因压力。
    return {
        f"feed:{lot['feed']}:bad": 1,
        f"zone:{lot['zone']}:bad": 1,
    }


def build_plan(
    ws: Workspace,
    as_of: int | None = None,
    periods: tuple[str, ...] = ("active",),
) -> dict[str, Any]:
    model = fit_outcome_model(ws)
    trucks: list[dict[str, Any]] = []
    requirements: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []

    for truck_id in ws.truck_order:
        truck = ws.trucks[truck_id]
        if truck["period"] not in periods:
            continue
        evaluation = evaluate_truck(ws, truck_id, as_of, model)
        trucks.append(evaluation)

        if evaluation["immutable_loaded_issues"]:
            exceptions.append(
                {
                    "truck_id": truck_id,
                    "kind": "loaded_lot_no_longer_conservative",
                    "lot_ids": evaluation["immutable_loaded_issues"],
                    "explanation": "已装车部分不能搬回拿舱；读数变差后只能作为例外跟踪，不能新增舱位救济",
                }
            )

        for lot_id in evaluation["problem_lot_ids"]:
            if lot_id in truck["loaded_ids"]:
                exceptions.append(
                    {
                        "truck_id": truck_id,
                        "lot_id": lot_id,
                        "kind": "loaded_problem_lot",
                        "explanation": "问题批已装车，不能搬回去申请备用舱",
                    }
                )
                continue
            matches = [m for m in spare_match_for_lot(ws, truck_id, lot_id, as_of) if m["eligible"]]
            if not matches:
                exceptions.append(
                    {
                        "truck_id": truck_id,
                        "lot_id": lot_id,
                        "kind": "no_eligible_spare_or_deadline_passed",
                        "all_matches": spare_match_for_lot(ws, truck_id, lot_id, as_of),
                    }
                )
                continue
            matches.sort(
                key=lambda m: (
                    0 if m["held_for"] == truck_id else 1,
                    m["decision_deadline"],
                    m["spare_id"],
                )
            )
            low = next(m["conservative_interval"][0] for m in evaluation["marginals"] if m["lot"] == lot_id)
            requirements.append(
                {
                    "truck_id": truck_id,
                    "lot_id": lot_id,
                    "probability_low": low,
                    "candidates": matches,
                }
            )

    claims: list[dict[str, Any]] = []
    # 最差且候选少的批先选；这使 held_for Z03 先覆盖 L10，Z01 同时隔离 F5 两单。
    requirements.sort(key=lambda r: (len(r["candidates"]), r["probability_low"], r["truck_id"], r["lot_id"]))
    for requirement in requirements:
        truck_id = requirement["truck_id"]
        lot_id = requirement["lot_id"]
        truck = ws.trucks[truck_id]
        selected: dict[str, Any] | None = None
        last_error: Exception | None = None

        for chosen in requirement["candidates"]:
            spare_id = chosen["spare_id"]
            next_at = chosen["next_reading_at"]
            deadline = chosen["decision_deadline"]
            if next_at is not None and next_at <= deadline:
                decision_type = "wait_for_named_reading_then_decide_by_deadline"
                timing_note = f"可等待 {next_at} 的读数，但必须在 {deadline} 前决定/接收"
            else:
                decision_type = "decide_on_visible_facts_before_deadline"
                timing_note = (
                    f"下一次读数 {next_at} 晚于最迟接收 {deadline}；必须在 {deadline} 前按已送达读数决定，"
                    "不能承诺等新读数后的救济"
                )
            claim = {
                "claim_key": f"{truck_id}:{lot_id}:{spare_id}",
                "spare_id": spare_id,
                "truck_id": truck_id,
                "lot_id": lot_id,
                "quantity": 1,
                "status": "recommended",
                "assigned_unit_id": None,
                "decision_deadline": deadline,
                "must_decide_by": deadline,
                "next_reading_at": next_at,
                "next_reading_visible_by_deadline": chosen["next_reading_visible_by_deadline"],
                "decision_type": decision_type,
                "timing_note": timing_note,
                "coverage_window": chosen["window"],
                "carrier": truck["carrier"],
                "zone": ws.lots[lot_id]["zone"],
                "occupancy_by_scenario": _scenario_tags(ws, lot_id),
                "exclusions": [
                    "不保证含水最终达标或客户一定按时收到",
                    f"仅覆盖备用舱窗口 {chosen['window'][0]}–{chosen['window'][1]}；窗口外不覆盖",
                    "载具/温区不匹配、超过 last_accept、已装车后撤回均不覆盖",
                    "若触发读数在 last_accept 后才可见，不得作该项救济承诺",
                ],
                "match": chosen,
            }
            try:
                assigned = assign_units(claims + [claim], ws)
                validate_capacity(assigned, ws)
            except CapacityError as exc:
                last_error = exc
                continue
            claims = assigned
            selected = next(c for c in claims if c["claim_key"] == claim["claim_key"])
            break

        if selected is None:
            exceptions.append(
                {
                    "truck_id": truck_id,
                    "lot_id": lot_id,
                    "kind": "joint_scenario_capacity_conflict",
                    "explanation": str(last_error) if last_error else "没有可在联合坏情景下隔离的备用位置",
                }
            )

    return {
        "as_of": as_of,
        "clock": ws.current_clock("active"),
        "trucks": trucks,
        "claims": claims,
        "exceptions": exceptions,
        "capacity_warning": (
            "F5 使 G05 与 G07 处于同一联合坏情景；二者各占不同备用位置，不能共享同一个位置。"
            "互斥触发的两份或有承诺可共用一个位置，但必须按具名情景校验。"
        ),
    }
