"""成品养护组单的时点视图与数据规范化。

本模块只负责资料读入和“在某个决策时点究竟能看见什么”，不把尚未收到的
检测、尚未到期的 open 批或 transferred/scrap 事件提前解释成普通失败。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 允许从仓库根目录直接运行，也允许测试从任意工作目录启动。
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from check_sources import load_sources  # noqa: E402


def i(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    return int(value)


def f(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


def s(value: Any) -> str:
    return (value or "").strip()


def csv_list(value: Any) -> list[str]:
    text = s(value)
    return [part.strip() for part in text.split(",") if part.strip()]


def bool_flag(value: Any) -> bool:
    return s(value).lower() in {"1", "true", "yes", "y", "是", "允许"}


@dataclass(frozen=True)
class Check:
    lot: str
    observed: int
    received: int
    percent: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "lot": self.lot,
            "observed": self.observed,
            "received": self.received,
            "percent": self.percent,
        }


class Workspace:
    """规范化后的资料集合。"""

    def __init__(self, data_dir: str | Path = _ROOT / "data"):
        self.data_dir = Path(data_dir)
        tables = load_sources(self.data_dir)
        self.tables = tables

        self.lots: dict[str, dict[str, Any]] = {}
        for row in tables["lots"]:
            lot = dict(row)
            lot["start_i"] = i(row["start"], 0)
            lot["due_i"] = i(row["due"], 0)
            lot["terminal_at_i"] = i(row["terminal_at"])
            self.lots[lot["id"]] = lot

        self.checks: list[Check] = []
        self.checks_by_lot: dict[str, list[Check]] = {}
        for row in tables["checks"]:
            check = Check(
                lot=row["lot"],
                observed=i(row["observed"], 0) or 0,
                received=i(row["received"], 0) or 0,
                percent=f(row["percent"], 0.0) or 0.0,
            )
            self.checks.append(check)
            self.checks_by_lot.setdefault(check.lot, []).append(check)
        for items in self.checks_by_lot.values():
            items.sort(key=lambda c: (c.observed, c.received))

        self.trucks: dict[str, dict[str, Any]] = {}
        self.truck_order: list[str] = []
        for row in tables["trucks"]:
            truck = dict(row)
            truck["lot_ids"] = csv_list(row["lots"])
            truck["loaded_ids"] = csv_list(row["loaded"])
            truck["allowed_zone_list"] = csv_list(row["allowed_zones"])
            truck["depart_i"] = i(row["depart"], 0) or 0
            truck["latest_i"] = i(row["latest"], truck["depart_i"]) or truck["depart_i"]
            truck["split_allowed"] = bool_flag(row["split"])
            truck["required_f"] = f(row["required"], 0.9) or 0.9
            self.trucks[truck["id"]] = truck
            self.truck_order.append(truck["id"])

        self.spares: dict[str, dict[str, Any]] = {}
        for row in tables["spares"]:
            spare = dict(row)
            spare["window_from_i"] = i(row["window_from"], 0) or 0
            spare["window_to_i"] = i(row["window_to"], 0) or 0
            spare["places_i"] = i(row["places"], 0) or 0
            spare["carrier_list"] = csv_list(row["carriers"])
            spare["zone_list"] = csv_list(row["zone"])
            spare["held_for_s"] = s(row["held_for"]) or None
            self.spares[spare["id"]] = spare

        self.decision_points: dict[str, dict[str, Any]] = {}
        for row in tables["decision_points"]:
            dp = dict(row)
            dp["last_accept_i"] = i(row["last_accept"], 0) or 0
            dp["new_reading_at_i"] = i(row["new_reading_at"], 0) or 0
            dp["new_reading_visible_by_deadline"] = (
                dp["new_reading_at_i"] <= dp["last_accept_i"]
            )
            self.decision_points[dp["spare"]] = dp

    # ---------- 时点视图 ----------

    def visible_checks(self, lot_id: str, as_of: int | None) -> list[Check]:
        """返回 as_of 时点已送达（received <= as_of）的检测。

        as_of=None 表示使用资料中全部已记录信息；历史回测传入决策时点。
        observed 是采样时点，received 才是决策系统可见时点。
        """
        items = self.checks_by_lot.get(lot_id, [])
        if as_of is None:
            return list(items)
        return [c for c in items if c.received <= as_of]

    def latest_check(self, lot_id: str, as_of: int | None) -> Check | None:
        visible = self.visible_checks(lot_id, as_of)
        if not visible:
            return None
        # 同一观测点可能有重复送达记录；资料身份键保证 observed 唯一。
        return max(visible, key=lambda c: (c.received, c.observed))

    def previous_check(self, lot_id: str, as_of: int | None) -> Check | None:
        visible = sorted(self.visible_checks(lot_id, as_of), key=lambda c: c.observed)
        if len(visible) < 2:
            return None
        return visible[-2]

    def lot_status_at(self, lot_id: str, as_of: int | None) -> dict[str, Any]:
        """返回某时点的批次状态。

        - pass/fail/scrap：只有 terminal_at <= as_of 才可见；
        - transferred：已知转仓后对本仓是不可用、对寿命/结果是右删失，不记 fail；
        - open/未到期：右删失，只把已送达检测纳入估计。
        """
        lot = self.lots[lot_id]
        source_state = lot["state"]
        terminal_at = lot["terminal_at_i"]
        if as_of is None:
            visible_state = source_state
            terminal_visible = terminal_at is not None
        elif terminal_at is not None and terminal_at <= as_of:
            visible_state = source_state
            terminal_visible = True
        else:
            visible_state = "open"
            terminal_visible = False

        if visible_state == "pass":
            available = True
            countable_success: bool | None = True
            adverse_for_group = False
            censored = False
            note = "已有按时达标完结"
        elif visible_state == "fail":
            available = False
            countable_success = False
            adverse_for_group = True
            censored = False
            note = "组单时限内明确未达标"
        elif visible_state == "scrap":
            available = False
            countable_success = False
            adverse_for_group = True
            censored = False
            note = "已明确报废；整车可计成功概率为零，但建模时是报废竞争事件而非检测失败"
        elif visible_state == "transferred":
            # 对当前仓/当前组单不可用；不能当失败标签，只能在转仓时点右删失。
            available = False
            countable_success = None
            adverse_for_group = True
            censored = True
            note = "已转其他仓继续养护；本仓组单不可计入，统计上按转仓/删失处理"
        else:
            available = True
            countable_success = None
            adverse_for_group = False
            censored = True
            note = "尚在观察或最终窗口未到；不是失败"

        return {
            "lot": lot_id,
            "state_at_as_of": visible_state,
            "source_state": source_state,
            "terminal_visible": terminal_visible,
            "terminal_at": terminal_at,
            "available_for_original_group": available,
            "countable_success": countable_success,
            "adverse_for_group": adverse_for_group,
            "censored": censored,
            "note": note,
        }

    def current_clock(self, period: str = "active") -> int:
        """当前演练期已记录的最晚送达时间。"""
        values = [
            c.received
            for c in self.checks
            if self.lots[c.lot]["period"] == period
        ]
        return max(values, default=0)
