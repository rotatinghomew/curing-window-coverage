"""按共因组/温区隔离的时点历史验证。"""
from __future__ import annotations

from typing import Any

import numpy as np

from .probability import fit_outcome_model, joint_assessment, marginal_for_lot
from .workspace import Workspace


def grouped_split_report(ws: Workspace) -> dict[str, Any]:
    train_feeds = {lot["feed"] for lot in ws.lots.values() if lot["period"] == "train"}
    holdout_feeds = {lot["feed"] for lot in ws.lots.values() if lot["period"] == "holdout"}
    train_zones = {lot["zone"] for lot in ws.lots.values() if lot["period"] == "train"}
    holdout_zones = {lot["zone"] for lot in ws.lots.values() if lot["period"] == "holdout"}
    return {
        "feed_overlap": sorted(train_feeds & holdout_feeds),
        "feed_grouped_split_valid": not (train_feeds & holdout_feeds),
        "train_feed_count": len(train_feeds),
        "holdout_feed_count": len(holdout_feeds),
        "train_zones": sorted(train_zones),
        "holdout_zones": sorted(holdout_zones),
        "zone_overlap": sorted(train_zones & holdout_zones),
        "note": (
            "train/holdout 的 feed 已整体隔开；但 cold/warm 温区跨期间重复，"
            "因此另做 leave-one-zone-block-out。new 无历史样本，不能验证。"
        ),
    }


def _binary_final(lot: dict[str, Any]) -> int | None:
    if lot["state"] == "transferred":
        return None
    return 1 if lot["state"] == "pass" else 0


def point_in_time_lot_backtest(ws: Workspace, as_of: int) -> dict[str, Any]:
    """只用 train 拟合，在 holdout 的 as_of 时点预测，再对最终标签评分。"""
    model = fit_outcome_model(ws, periods=("train",))
    ys: list[int] = []
    ps: list[float] = []
    rows: list[dict[str, Any]] = []
    for lot_id, lot in ws.lots.items():
        if lot["period"] != "holdout":
            continue
        y = _binary_final(lot)
        if y is None:
            continue
        m = marginal_for_lot(ws, lot_id, as_of, model)
        if m["state_at_as_of"] in {"pass", "fail", "scrap"}:
            # 已完结事件是当时可知的确定标签，仍可纳入校准但不会检验读数模型。
            p = float(m["decision_probability"])
        else:
            p = float(m["probability"])
        ys.append(y)
        ps.append(p)
        rows.append({"lot": lot_id, "y": y, "p": p, "state_at_as_of": m["state_at_as_of"]})

    y_arr = np.array(ys)
    p_arr = np.clip(np.array(ps), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p_arr - y_arr) ** 2)) if len(y_arr) else None
    log_loss = float(-np.mean(y_arr * np.log(p_arr) + (1 - y_arr) * np.log(1 - p_arr))) if len(y_arr) else None
    return {
        "as_of": as_of,
        "evaluation_rule": "仅使用 received <= as_of 的检测和 terminal_at <= as_of 的完结事件",
        "n": len(ys),
        "brier": brier,
        "log_loss": log_loss,
        "rows": rows,
    }


def zone_blocked_check(ws: Workspace, as_of: int = 28) -> dict[str, Any]:
    """整体留出一个温区：用另一温区拟合，预测留出温区 holdout 批。"""
    output: dict[str, Any] = {}
    for train_zone, predict_zone in [("cold", "warm"), ("warm", "cold")]:
        model = fit_outcome_model(ws, periods=("train", "holdout"), zones=[train_zone])
        ys: list[int] = []
        ps: list[float] = []
        for lot_id, lot in ws.lots.items():
            if lot["zone"] != predict_zone:
                continue
            y = _binary_final(lot)
            if y is None:
                continue
            m = marginal_for_lot(ws, lot_id, as_of, model)
            p = float(m["probability"] if m["probability"] is not None else 0.5)
            ys.append(y)
            ps.append(p)
        if ys:
            y_arr = np.array(ys)
            p_arr = np.clip(np.array(ps), 1e-6, 1 - 1e-6)
            output[f"train_{train_zone}_predict_{predict_zone}"] = {
                "n": len(ys),
                "brier": float(np.mean((p_arr - y_arr) ** 2)),
                "log_loss": float(-np.mean(y_arr * np.log(p_arr) + (1 - y_arr) * np.log(1 - p_arr))),
            }
    output["limitation"] = (
        "温区数量很少且与 due/feed 结构混杂；该结果只能说明跨温区外推不稳，"
        "不能识别独立的温区随机效应。"
    )
    return output


def truck_group_backtest(ws: Workspace, as_of: int = 28) -> dict[str, Any]:
    """按整车/共因 feed 组核对保守准入是否与最终整车结果一致。"""
    model = fit_outcome_model(ws, periods=("train",))
    rows: list[dict[str, Any]] = []
    for truck_id in ws.truck_order:
        truck = ws.trucks[truck_id]
        if truck["period"] != "holdout":
            continue
        marginals = [marginal_for_lot(ws, lot_id, as_of, model) for lot_id in truck["lot_ids"]]
        joint = joint_assessment(marginals, truck["required_f"])
        final_states = [ws.lots[x]["state"] for x in truck["lot_ids"]]
        final_whole_success = all(x == "pass" for x in final_states)
        conservative_admit = bool(joint["meets_required_conservatively"])
        rows.append(
            {
                "truck": truck_id,
                "as_of": as_of,
                "lots": truck["lot_ids"],
                "final_states": final_states,
                "final_whole_success": final_whole_success,
                "conservative_admit": conservative_admit,
                "frechet_bounds": joint["frechet_bounds"],
                "required": truck["required_f"],
                "blocked_by_unavailable_lot": joint.get("blocked_by_unavailable_lot", False),
            }
        )
    tp = sum(r["conservative_admit"] and r["final_whole_success"] for r in rows)
    fp = sum(r["conservative_admit"] and not r["final_whole_success"] for r in rows)
    fn = sum(not r["conservative_admit"] and r["final_whole_success"] for r in rows)
    tn = sum(not r["conservative_admit"] and not r["final_whole_success"] for r in rows)
    return {
        "as_of": as_of,
        "rule": "只有包含边际不确定性的 Fréchet 保守下界达到 required 才准入整组",
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "rows": rows,
    }


def validation_report(ws: Workspace) -> dict[str, Any]:
    return {
        "grouped_split": grouped_split_report(ws),
        "lot_backtest_as_of_26": point_in_time_lot_backtest(ws, 26),
        "lot_backtest_as_of_28": point_in_time_lot_backtest(ws, 28),
        "zone_blocked_as_of_28": zone_blocked_check(ws, 28),
        "truck_group_backtest_as_of_28": truck_group_backtest(ws, 28),
        "unsupported_dependencies": [
            "历史 H* feed 与运输组单基本一一对应，只能估计同一整车/feed 内共因。",
            "F5 同时出现在 G05 与 G07，但历史没有同一 feed 跨两张当前组单的联合结果。",
            "new 温区没有历史完结批。",
            "历史 due/latest 最大为53；54/56小时改期窗口缺少直接样本。",
            "资料没有备用舱干预后的成功率、接收拒绝或释放记录，因此不能把给舱等同于达标保证。",
            "cold/warm 温区数量少且与 feed/due 混杂，不足以单独识别温区因果方差。",
        ],
    }
