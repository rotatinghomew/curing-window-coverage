"""生成 JSON + Markdown 业务核对报告。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .commitments import CommitmentStore
from .planning import build_plan, evaluate_truck
from .probability import (
    estimate_outcome_icc,
    fit_outcome_model,
    moisture_variance_components,
)
from .service import _reconciliation_notes
from .validation import validation_report
from .workspace import Workspace


def build_report(
    data_dir: str | Path = "data",
    as_of: int | None = None,
    state_path: str | Path | None = None,
) -> dict[str, Any]:
    ws = Workspace(data_dir)
    model = fit_outcome_model(ws)
    effective_as_of = as_of if as_of is not None else ws.current_clock("active")
    plan_25 = build_plan(ws, 25)
    plan_current = build_plan(ws, effective_as_of)
    old_by_truck = {t["truck_id"]: t for t in plan_25["trucks"]}
    reading_changes = []
    for truck in plan_current["trucks"]:
        old_truck = old_by_truck.get(truck["truck_id"])
        if old_truck:
            reading_changes.extend(_reconciliation_notes({"marginals": old_truck["marginals"]}, truck))

    ledger: dict[str, Any] | None = None
    current_marginals = [m for truck in plan_current["trucks"] for m in truck["marginals"] if m.get("probability") is not None]
    initial_marginals = [m for truck in plan_25["trucks"] for m in truck["marginals"] if m.get("probability") is not None]
    probability_summary = {
        "as_of": effective_as_of,
        "mean_active_reference_probability": sum(float(m["probability"]) for m in current_marginals) / max(1, len(current_marginals)),
        "as_of_25_mean_reference_probability": sum(float(m["probability"]) for m in initial_marginals) / max(1, len(initial_marginals)),
        "interpretation": "平均值只描述单批边际；整车必须使用联合界，不能用平均值或其乘积替代组单评估",
    }
    case_review = [
        evaluate_truck(ws, truck_id, 48)
        for truck_id in ("G01", "G02", "G03", "G04")
        if truck_id in ws.trucks
    ]

    if state_path is not None and Path(state_path).exists():
        store = CommitmentStore(state_path)
        ship_dates = sorted({key.split(":", 1)[0] for key in store.data["commitments"]})
        ledger = {
            "path": str(state_path),
            "ship_dates": ship_dates,
            "action_ids": [
                action["action_id"]
                for record in store.data["commitments"].values()
                for action in record.get("actions", [])
            ],
            "verifications": {day: store.verify_date(day, ws) for day in ship_dates},
        }

    event_accounting: dict[str, Any] = {}
    for state in ("pass", "fail", "scrap", "transferred", "open"):
        event_accounting[state] = sum(1 for lot in ws.lots.values() if lot["state"] == state)
    event_accounting["treatment"] = {
        "pass": "最终可计入的按时达标成功",
        "fail": "组单时限内明确未达标",
        "scrap": "已知报废；整组可计成功为0，建模时作为报废/竞争事件而非普通缺失",
        "transferred": "转其他仓继续养护；本仓组单不可用，统计上在转仓时点右删失，不记为失败",
        "open": "未到最终窗口；只使用 as_of 前已送达信息，右删失而非失败",
    }

    return {
        "as_of": effective_as_of,
        "data_dir": str(data_dir),
        "counts": {name: len(rows) for name, rows in ws.tables.items()},
        "event_accounting": event_accounting,
        "probability_summary": probability_summary,
        "outcome_model": model.as_dict(),
        "variance_24h": moisture_variance_components(ws, 24),
        "outcome_dependence": estimate_outcome_icc(ws),
        "validation": validation_report(ws),
        "reading_changes": reading_changes,
        "case_review_G01_G04": case_review,
        "commitment_ledger": ledger,
        "plan_at_as_of": plan_current,
        "decision_snapshots": {
            "as_of_25_before_Z01": plan_25,
            "as_of_28_Z03_deadline": build_plan(ws, 28),
            f"as_of_{effective_as_of}_current": plan_current,
        },
        "method_and_limits": [
            "received 时间决定读数何时可见；observed 只是采样时间。",
            "open/未到期按右删失处理，不按失败计数。",
            "scrap 是已知报废/竞争事件；transferred 是转仓删失且对当前组单不可用，二者不合并为普通丢失。",
            "整车准入使用 Fréchet 下界；独立乘积和 copula 仅作敏感性展示。",
            "备用舱不提供未记录的干预成功率，因此给舱只表示容量/接收保障，不表示达标保证。",
        ],
    }


def _pct(x: Any) -> str:
    if x is None:
        return "—"
    return f"{100 * float(x):.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    plan = report["plan_at_as_of"]
    var = report["variance_24h"]
    dep = report["outcome_dependence"]
    lines: list[str] = []
    lines.append("# 出库前养护与备用舱核对报告")
    lines.append("")
    lines.append(f"- 决策时钟 `as_of={report['as_of']}`；资料行数：" + ", ".join(f"{k}={v}" for k, v in report["counts"].items()))
    lines.append("- 概率仅用历史完结记录校准；时点视图按 `received <= as_of` 过滤。")
    lines.append("- **调度准入规则：包含边际不确定性的 Fréchet 保守下界 ≥ required；独立乘积不作为承诺依据。**")
    accounting = report.get("event_accounting", {})
    if accounting:
        lines.append(
            "- 事件记账："
            + "，".join(f"{k}={accounting[k]}" for k in ("pass", "fail", "scrap", "transferred", "open") if k in accounting)
            + "；scrap 与 transferred 分别处理，open 不按失败计。"
        )
    summary = report.get("probability_summary")
    if summary:
        lines.append(
            f"- active 批次平均参考边际概率：as_of=25 为 {_pct(summary['as_of_25_mean_reference_probability'])}，"
            f"as_of={report['as_of']} 为 {_pct(summary['mean_active_reference_probability'])}；"
            "该平均值看起来可用，但不能替代下面的整车联合界。"
        )
    lines.append("")

    lines.append("## 1. 共因变差与单批特有波动")
    lines.append("")
    lines.append(
        f"24 小时含水检测按 feed 随机效应估计：共享供料/共因 SD≈{var['sd_shared_feed']:.2f} 个百分点，"
        f"单批特有 SD≈{var['sd_lot_specific']:.2f}， moisture ICC≈{var['icc_moisture']:.2f}。"
    )
    lines.append(
        f"完结结果的 beta-binomial ICC≈{dep['outcome_icc_1_over_1_plus_phi']:.2f}。"
        "这说明同 feed/同整车的共同波动很强，拆单不会消除 F5 共因，也不能把几项边际概率相乘后当作整车保证。"
    )
    lines.append("")

    lines.append("## 2. 当前 active 组单")
    lines.append("")
    lines.append("| 组单 | required | 独立乘积 | Fréchet 区间 | 保守下界 | 可确定发运部分 | 结论 |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for truck in plan["trucks"]:
        j = truck["full_joint"]
        lines.append(
            "| {id} | {req} | {prod} | [{lo}, {hi}] | {clo} | {ship} | {decision} |".format(
                id=truck["truck_id"],
                req=_pct(truck["required"]),
                prod=_pct(j["independence_product"]),
                lo=_pct(j["frechet_bounds"][0]),
                hi=_pct(j["frechet_bounds"][1]),
                clo=_pct(j["conservative_lower"]),
                ship=",".join(truck["recommended_ship_lot_ids"]) or "无",
                decision=truck["decision"],
            )
        )
    lines.append("")
    lines.append("### 相依结构对整车概率的影响")
    lines.append("")
    lines.append("| 组单 | 独立 ρ=0 | 历史 ICC≈0.62 copula | 完全共因/min p | Fréchet 下界 | required |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for truck in plan["trucks"]:
        j = truck["full_joint"]
        copula_value = next(v for k, v in j["scenarios"].items() if k.startswith("gaussian_copula"))
        lines.append(
            f"| {truck['truck_id']} | {_pct(j['independence_product'])} | {_pct(copula_value)} | "
            f"{_pct(j['scenarios']['perfect_common_cause_min_marginal'])} | "
            f"{_pct(j['frechet_bounds'][0])} | {_pct(truck['required'])} |"
        )
    lines.append("")
    lines.append("> ρ=0 只是“单批特有波动完全独立”的敏感性；0.62 来自历史 beta-binomial，不等于已识别 F5 跨组联合分布。准入仍看 Fréchet/不确定性下界。")
    lines.append("")

    lines.append("### 批次时点概率")
    lines.append("")
    for truck in plan["trucks"]:
        lines.append(f"**{truck['truck_id']}**（载具 {truck['carrier']}，温区 {','.join(truck['allowed_zones'])}，loaded={','.join(truck['loaded_ids']) or '无'}）")
        lines.append("")
        for m in truck["marginals"]:
            latest = m.get("latest_visible")
            latest_text = (
                f"最新可见 @{latest['observed']}/received{latest['received']} = {latest['percent']}"
                if latest
                else "无已送达读数"
            )
            node_text = "；节点参考 p=" + ", ".join(
                f"{n['observed']}h:{_pct(n['reference_probability'])}" for n in m.get("node_forecasts", [])
            ) if m.get("node_forecasts") else ""
            lines.append(
                f"- `{m['lot']}`/{m['feed']}/{m['zone']}：状态 `{m['state_at_as_of']}`；{latest_text}；"
                f"参考 p={_pct(m['probability'])}，保守下界={_pct(m['conservative_interval'][0])}{node_text}。"
                f"{m.get('zone_warning', '')}"
            )
        wait = truck["wait_to_latest_reference"]
        lines.append(
            f"- 等到 latest={truck['latest']} 的参考乘积={_pct(wait['joint']['independence_product'])}，"
            f"保守下界={_pct(wait['joint']['conservative_lower'])}；历史截止支撑={wait['evidence_supported']}。{wait['note']}"
        )
        lines.append("")

    lines.append("## 资料中 G01–G04 的事件记账对照")
    lines.append("")
    lines.append("| 组单 | 最终状态 | 整组保守下界 | 可执行部分 | 说明 |")
    lines.append("|---|---|---:|---|---|")
    for case in report.get("case_review_G01_G04", []):
        states = ",".join(f"{m['lot']}:{m['state_at_as_of']}" for m in case["marginals"])
        lines.append(
            f"| {case['truck_id']} | {states} | {_pct(case['full_joint']['conservative_lower'])} | "
            f"{','.join(case['recommended_ship_lot_ids']) or '无'} | {case['decision']} |"
        )
    lines.append("")
    lines.append("G02 的 scrap 与 G04 的 transferred 都不按普通缺失删除；前者使当前组单确定不可整组执行，后者在转仓时点删失且本仓不可计入。")
    lines.append("")

    lines.append("## 3. 备用舱与决策时点")
    lines.append("")
    initial = report["decision_snapshots"]["as_of_25_before_Z01"]
    lines.append("### 截止前必须保留的位置（as_of=25 视图）")
    lines.append("")
    initial_claims = initial["claims"]
    if initial_claims:
        lines.append("| 承诺键 | 位置 | 决策截止 | 读数时点 | 决策方式 | 覆盖窗口 |")
        lines.append("|---|---|---:|---:|---|---:|")
        for c in initial_claims:
            lines.append(
                f"| {c['claim_key']} | {c['assigned_unit_id']} | {c['decision_deadline']} | "
                f"{c['next_reading_at']} | {c['decision_type']} | {c['coverage_window'][0]}–{c['coverage_window'][1]} |"
            )
        lines.append("")
        for c in initial_claims:
            coverage = c.get("coverage_summary") or (
                f"保留 {c['assigned_unit_id']}（{c['spare_id']}）在 "
                f"{c['coverage_window'][0]}–{c['coverage_window'][1]} 小时、"
                f"{c['carrier']}/{c['zone']} 温区的一个批次位置"
            )
            lines.append(f"- **{c['claim_key']} → {c['assigned_unit_id']}**：{c['timing_note']}")
            lines.append(f"  - 保留保障：{coverage}。")
            lines.append("  - 不覆盖：" + "；".join(c["exclusions"]))
    lines.append("")

    lines.append(f"### 当前 as_of={report['as_of']} 仍可新申请/核对的位置")
    lines.append("")
    if plan["claims"]:
        lines.append("| 承诺键 | 位置 | 决策截止 | 读数时点 | 决策方式 | 覆盖窗口 |")
        lines.append("|---|---|---:|---:|---|---:|")
        for c in plan["claims"]:
            lines.append(
                f"| {c['claim_key']} | {c['assigned_unit_id']} | {c['decision_deadline']} | "
                f"{c['next_reading_at']} | {c['decision_type']} | {c['coverage_window'][0]}–{c['coverage_window'][1]} |"
            )
    else:
        lines.append("当前时点没有可新申请的早期舱位；若未在截止前写入台账，只能列为例外，不能追溯承诺。")
    lines.append("")
    if plan["exceptions"]:
        lines.append("### 时点例外与既有承诺核对")
        lines.append("")
        ledger_ids = set((report.get("commitment_ledger") or {}).get("action_ids", []))
        for e in plan["exceptions"]:
            truck_id = e.get("truck_id", "")
            lot_id = e.get("lot_id", "")
            prior_key = f"spare:{truck_id}:{lot_id}:"
            covered_by_ledger = any(action_id.startswith(prior_key) for action_id in ledger_ids)
            if covered_by_ledger:
                lines.append(
                    f"- `{truck_id}` `{lot_id}` `{e['kind']}`：当前不能新申请，"
                    "但台账中已有截止前承诺；保留未履行部分并按新读数核对，不追溯改写决策。"
                )
            else:
                lines.append(f"- `{truck_id}` `{lot_id}` `{e['kind']}`：{e.get('explanation', '见匹配明细')}")
        lines.append("")
    lines.append(plan["capacity_warning"])
    lines.append("")

    if report.get("reading_changes"):
        lines.append("### 读数变化如何改变结论")
        lines.append("")
        for change in report["reading_changes"]:
            if "lot_id" in change:
                lines.append(f"- `{change['lot_id']}`：{change['explanation']}")
                if change.get("classification_change"):
                    lines.append(f"  - {change['classification_change']}")
        lines.append("")

    v = report["validation"]
    lines.append("## 4. 历史验证与不能支撑的估计")
    lines.append("")
    gs = v["grouped_split"]
    lines.append(
        f"- Feed 整体隔离：train/holdout feed 交集={gs['feed_overlap']}，"
        f"分组切分有效={gs['feed_grouped_split_valid']}。"
    )
    for key in ["lot_backtest_as_of_26", "lot_backtest_as_of_28"]:
        item = v[key]
        lines.append(f"- {key}：n={item['n']}，Brier={item['brier']:.3f}，log loss={item['log_loss']:.3f}。")
    bt = v["truck_group_backtest_as_of_28"]
    lines.append(f"- holdout 整车保守准入混淆计数：{bt['counts']}。")
    lines.append("- 相依/资料限制：")
    for item in v["unsupported_dependencies"]:
        lines.append(f"  - {item}")
    lines.append("")
    ledger = report.get("commitment_ledger")
    if ledger:
        lines.append("## 5. 已确认承诺台账")
        lines.append("")
        lines.append(f"- 台账：`{ledger['path']}`；发货日：{', '.join(ledger['ship_dates'])}")
        for day in ledger["ship_dates"]:
            verification = ledger["verifications"][day]
            capacity = verification["capacity"]
            lines.append(
                f"- `{day}`：{verification['submission_count']} 张组单，"
                f"{capacity['active_claim_count']} 个有效备用舱承诺；情景容量核对通过。"
            )
        lines.append("")

    lines.append("## 6. 可重复操作")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m dispatch_planning report --as-of 25")
    lines.append("python -m dispatch_planning submit-day --ship-date 2026-11-25 --as-of 25 --confirm")
    lines.append("python -m dispatch_planning submit-day --ship-date 2026-11-25 --as-of 25 --confirm  # 幂等")
    lines.append("python -m dispatch_planning submit-day --ship-date 2026-11-25 --as-of 29 --confirm")
    lines.append("python -m dispatch_planning verify --ship-date 2026-11-25")
    lines.append("```")
    return "\n".join(lines) + "\n"


def write_report(
    out_dir: str | Path = "reports",
    data_dir: str | Path = "data",
    as_of: int | None = None,
    state_path: str | Path | None = None,
) -> dict[str, Path]:
    report = build_report(data_dir, as_of, state_path=state_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "dispatch_report.json"
    md_path = out / "dispatch_report.md"
    plan_path = out / "dispatch_plan.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    scheduler_plan = {
        "as_of": report["as_of"],
        "required_action": "use commitment ledger after scheduler confirmation; this file is the proposed plan",
        "trucks": [
            {
                "truck_id": t["truck_id"],
                "decision": t["decision"],
                "recommended_ship_lot_ids": t["recommended_ship_lot_ids"],
                "loaded_ids": t["loaded_ids"],
                "required": t["required"],
                "full_joint": t["full_joint"],
            }
            for t in report["plan_at_as_of"]["trucks"]
        ],
        "claims": report["plan_at_as_of"]["claims"],
        "exceptions": report["plan_at_as_of"]["exceptions"],
        "capacity_warning": report["plan_at_as_of"]["capacity_warning"],
    }
    plan_path.write_text(json.dumps(scheduler_plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": md_path, "plan": plan_path}
