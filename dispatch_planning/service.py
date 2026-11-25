"""把当日计划转换成可重复提交、可核对的调度承诺。"""
from __future__ import annotations

from typing import Any

from .commitments import CommitmentStore
from .planning import build_plan, evaluate_truck
from .probability import fit_outcome_model
from .workspace import Workspace


def _reconciliation_notes(old: dict[str, Any] | None, evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    if not old:
        return []
    old_marginals = {m["lot"]: m for m in old.get("marginals", [])}
    notes: list[dict[str, Any]] = []
    for marginal in evaluation["marginals"]:
        lot_id = marginal["lot"]
        old_marginal = old_marginals.get(lot_id)
        if not old_marginal:
            continue
        old_latest = old_marginal.get("latest_visible")
        new_latest = marginal.get("latest_visible")

        def reading_signature(value: dict[str, Any] | None):
            if not value:
                return None
            return (value.get("observed"), value.get("received"), value.get("percent"))

        if reading_signature(old_latest) != reading_signature(new_latest):
            notes.append(
                {
                    "lot_id": lot_id,
                    "old_latest_visible": old_latest,
                    "new_latest_visible": new_latest,
                    "old_reference_probability": old_marginal.get("probability"),
                    "new_reference_probability": marginal.get("probability"),
                    "explanation": (
                        f"{lot_id} 的可见读数已更新；只允许修改尚未履行/未装车的动作。"
                        if lot_id not in evaluation["loaded_ids"]
                        else f"{lot_id} 已装车；即使读数变差也不能搬回拿舱，只能列为例外。"
                    ),
                }
            )
            if lot_id == "L12":
                notes[-1]["classification_change"] = (
                    "12h/9.5 曾使跨区 logistic 参考值看起来很高；29 才收到的 24h/10.9 反弹且 new 温区无历史支撑，"
                    "因此当前不能作无条件发运承诺。Z02 仍是在45前可凭43读数决定的或有抢救位置，但给舱不等于达标保证。"
                )
    old_joint = old.get("full_group", {})
    new_joint = evaluation["full_joint"]
    if old_joint.get("meets_required_conservatively") != new_joint.get("meets_required_conservatively"):
        notes.append(
            {
                "truck_id": evaluation["truck_id"],
                "kind": "group_qualification_changed",
                "old": old_joint,
                "new": {
                    "frechet_bounds": new_joint["frechet_bounds"],
                    "conservative_lower": new_joint["conservative_lower"],
                    "meets_required_conservatively": new_joint["meets_required_conservatively"],
                },
            }
        )
    return notes


def _compact_marginals(evaluation: dict[str, Any]) -> list[dict[str, Any]]:    return [
        {
            "lot": m["lot"],
            "state_at_as_of": m["state_at_as_of"],
            "probability": None if m["probability"] is None else round(float(m["probability"]), 6),
            "conservative_low": round(float(m["conservative_interval"][0]), 6),
            "latest_visible": m["latest_visible"],
        }
        for m in evaluation["marginals"]
    ]


def _apply_trigger_state(
    action: dict[str, Any],
    marginal_by_lot: dict[str, dict[str, Any]],
    required: float,
    as_of: int | None,
) -> None:
    if action.get("kind") != "spare_commitment" or action.get("status") == "fulfilled":
        return
    if str(action.get("status", "")).startswith("released"):
        action["coverage_state"] = "released"
        return

    marginal = marginal_by_lot.get(action.get("lot_id"), {})
    low = float(marginal.get("conservative_interval", [0.0, 1.0])[0])
    meets = low + 1e-12 >= required
    deadline = action.get("decision_deadline")
    next_at = action.get("next_reading_at")

    if as_of is None or deadline is None:
        trigger_state = "clock_unspecified"
    elif next_at is not None and as_of < next_at:
        trigger_state = "waiting_for_named_reading"
    elif as_of <= deadline:
        trigger_state = "reading_visible_or_decision_due_by_deadline"
    elif meets:
        action["status"] = "released_after_reading_met_threshold"
        trigger_state = "released_after_conservative_threshold_met"
    else:
        # 承诺是在 last_accept 前按时作出的；窗口后仍为问题批时，保留物理保障。
        # 这仍不表示含水达标或客户时效已被保证。
        action["status"] = "activated"
        trigger_state = "coverage_active_quality_not_guaranteed"

    action["trigger_state"] = trigger_state
    action["lot_conservative_low"] = low
    action["lot_meets_required"] = meets


def build_submission(
    ws: Workspace,
    truck_id: str,
    ship_date: str,
    as_of: int | None,
    *,
    confirm: bool = True,
    store: CommitmentStore | None = None,
) -> dict[str, Any]:
    """构造一张当日组单的承诺文档，并合并此前未履行承诺。"""
    model = fit_outcome_model(ws)
    truck = ws.trucks[truck_id]
    evaluation = evaluate_truck(ws, truck_id, as_of, model)
    plan = build_plan(ws, as_of)
    fresh_claims = {c["claim_key"]: c for c in plan["claims"] if c["truck_id"] == truck_id}
    old = store.get(ship_date, truck_id) if store is not None else None
    old_actions = {a["action_id"]: a for a in (old or {}).get("actions", [])}

    actions: list[dict[str, Any]] = []

    # 发运/拆分动作：loaded 批从第一次提交起就是不可撤回的已装车部分；
    # 即使它已不满足保守准入，也保留该动作并标为不可变更的例外，而不是假装可以搬回拿舱。
    ship_lot_ids = list(dict.fromkeys(truck["loaded_ids"] + evaluation["recommended_ship_lot_ids"]))
    for lot_id in ship_lot_ids:
        action_id = f"ship:{truck_id}:{lot_id}"
        old_action = old_actions.get(action_id, {})
        is_loaded = lot_id in truck["loaded_ids"]
        loaded_issue = lot_id in evaluation["immutable_loaded_issues"]
        default_status = "confirmed" if confirm else "planned"
        if is_loaded and loaded_issue:
            default_status = "exception_immutable_loaded"
        old_status = old_action.get("status")
        if old_status == "fulfilled":
            status = "fulfilled"
        elif is_loaded and loaded_issue:
            status = "exception_immutable_loaded"
        else:
            status = old_status or default_status
        action = {
            "action_id": action_id,
            "kind": "ship_commitment",
            "truck_id": truck_id,
            "lot_id": lot_id,
            "ship_lot_ids": evaluation["recommended_ship_lot_ids"],
            "status": status,
            "immutable": is_loaded,
            "qualification_exception": loaded_issue,
            "coverage": "已装车段不可撤回；其他部分按合同允许的拆分发运，且只包含保守下界达到 required 的部分",
            "assessment_as_of": as_of,
        }
        actions.append(action)

    # 新计划中的备用舱承诺。
    for claim in fresh_claims.values():
        action_id = f"spare:{claim['claim_key']}"
        old_action = old_actions.get(action_id, {})
        if old_action.get("status") == "fulfilled":
            # 已履行部分绝不因读数变化而改写核心资源。
            actions.append(old_action)
            continue
        status = old_action.get("status")
        if status is None:
            if not confirm:
                status = "planned"
            elif claim["decision_type"].startswith("wait_for"):
                status = "conditional_reserved"
            else:
                status = "reserved"
        action = {
            "action_id": action_id,
            "kind": "spare_commitment",
            **{k: v for k, v in claim.items() if k != "match"},
            "status": status,
            "immutable": bool(old_action.get("immutable", False)) or old_action.get("status") == "fulfilled",
            "coverage_summary": (
                f"保留 {claim['assigned_unit_id']}（{claim['spare_id']}）在 "
                f"{claim['coverage_window'][0]}–{claim['coverage_window'][1]} 小时、"
                f"{claim['carrier']} / {claim['zone']} 温区的一个批次位置"
            ),
            "non_coverage": claim["exclusions"],
            "reconciled_as_of": as_of,
        }
        actions.append(action)

    # 时点已过但此前已经提交的舱位承诺必须保留；这正是“重复提交/读数更新只改未履行部分”。
    if old is not None:
        for old_action in old.get("actions", []):
            if old_action.get("kind") != "spare_commitment":
                continue
            action_id = old_action["action_id"]
            if action_id in {a["action_id"] for a in actions}:
                continue
            updated = dict(old_action)
            if updated.get("status") != "fulfilled":
                updated["reconciled_as_of"] = as_of
                updated["reconciliation_note"] = (
                    "合并此前承诺；核心资源不变，只按当前时点和读数更新未履行状态"
                )
            actions.append(updated)

    required = evaluation["required"]
    marginal_by_lot = {m["lot"]: m for m in evaluation["marginals"]}
    for action in actions:
        _apply_trigger_state(action, marginal_by_lot, required, as_of)

    exceptions = [e for e in plan["exceptions"] if e.get("truck_id") == truck_id]
    fresh_reconciliation = _reconciliation_notes(old, evaluation)
    prior_reconciliation = old.get("reconciliation", []) if old else []
    reconciliation = list(prior_reconciliation)
    for note in fresh_reconciliation:
        if note not in reconciliation:
            reconciliation.append(note)
    joint = evaluation["full_joint"]
    status = "confirmed" if confirm else "draft"
    if exceptions or not evaluation["recommended_ship_lot_ids"]:
        # 有可执行部分时仍是部分确认；整组无保守方案则标 exception。
        status = "confirmed_with_exceptions" if confirm and actions else "exception"

    return {
        "ship_date": ship_date,
        "truck_id": truck_id,
        "as_of": as_of,
        "status": status,
        "required_probability": required,
        "full_group": {
            "frechet_bounds": joint["frechet_bounds"],
            "independence_product": joint["independence_product"],
            "conservative_lower": joint["conservative_lower"],
            "meets_required_conservatively": joint["meets_required_conservatively"],
        },
        "recommended_ship_lot_ids": evaluation["recommended_ship_lot_ids"],
        "loaded_ids": evaluation["loaded_ids"],
        "marginals": _compact_marginals(evaluation),
        "actions": actions,
        "exceptions": exceptions,
        "reconciliation": reconciliation,
        "why_not_independence": joint["no_dependence_assumption_note"],
    }


def submit_day(
    ws: Workspace,
    store: CommitmentStore,
    ship_date: str,
    as_of: int | None,
    *,
    confirm: bool = True,
    periods: tuple[str, ...] = ("active",),
) -> dict[str, Any]:
    results = []
    for truck_id in ws.truck_order:
        if ws.trucks[truck_id]["period"] not in periods:
            continue
        submission = build_submission(
            ws,
            truck_id,
            ship_date,
            as_of,
            confirm=confirm,
            store=store,
        )
        record, change = store.upsert(submission, ws)
        results.append({"truck_id": truck_id, "change": change, "revision": record.get("revision")})
    verification = store.verify_date(ship_date, ws)
    return {"ship_date": ship_date, "as_of": as_of, "results": results, "verification": verification}
