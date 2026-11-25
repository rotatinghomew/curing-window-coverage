"""时点概率、相依区间与共因/特有波动分析。

这里刻意同时给出三种数字：

1. 经历史资料校准的边际概率（仅作参考）；
2. 独立性乘积和高斯 copula 敏感性；
3. 不需要联合分布假设的 Fréchet 界。

调度准入只使用可保守解释的下界；缺少温区/跨组联合样本时，不把乘积当作
整车概率。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from .workspace import Check, Workspace

FEATURE_NAMES = [
    "intercept",
    "moisture_centered",
    "recent_drop",
    "no_trend",
    "horizon_scaled",
    "zone_warm",
    "zone_new",
]


def _feature_vector(
    lot: dict[str, Any],
    check: Check,
    previous: Check | None,
) -> np.ndarray:
    drop = (check.percent - previous.percent) if previous is not None else 0.0
    horizon = (int(lot["due"]) - check.observed) / 10.0
    return np.array(
        [
            1.0,
            (check.percent - 10.0) / 3.0,
            -drop / 2.0,  # 正值表示最近一次下降，越有利
            1.0 if previous is None else 0.0,
            horizon,
            1.0 if lot["zone"] == "warm" else 0.0,
            1.0 if lot["zone"] == "new" else 0.0,
        ],
        dtype=float,
    )


@dataclass
class OutcomeModel:
    beta: np.ndarray
    covariance: np.ndarray
    train_rows: int
    train_lots: int
    l2: float = 1.0

    def predict(self, x: np.ndarray) -> tuple[float, float, float]:
        eta = float(x @ self.beta)
        p = float(expit(eta))
        se = float(np.sqrt(max(0.0, x @ self.covariance @ x)))
        lo = float(expit(eta - 1.96 * se))
        hi = float(expit(eta + 1.96 * se))
        return p, lo, hi

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": FEATURE_NAMES,
            "coefficients": [round(float(x), 6) for x in self.beta],
            "train_rows": self.train_rows,
            "train_lots": self.train_lots,
            "l2": self.l2,
        }


def fit_outcome_model(
    ws: Workspace,
    periods: Iterable[str] = ("train", "holdout"),
    zones: Iterable[str] | None = None,
    l2: float = 1.0,
) -> OutcomeModel:
    """用已完结历史批拟合轻量 logistic 校准模型。

    transferred 不进入二元成功/失败似然：它在转仓后没有本仓最终标签。
    scrap/fail 对“本仓当前组单可否计成功”都是 0；scrap 的事件含义另列。
    """
    period_set = set(periods)
    zone_set = set(zones) if zones is not None else None
    xs: list[np.ndarray] = []
    ys: list[float] = []
    lots_used: set[str] = set()
    for lot_id, lot in ws.lots.items():
        if lot["period"] not in period_set:
            continue
        if zone_set is not None and lot["zone"] not in zone_set:
            continue
        if lot["state"] == "transferred":
            continue
        y = 1.0 if lot["state"] == "pass" else 0.0
        checks = sorted(ws.checks_by_lot.get(lot_id, []), key=lambda c: c.observed)
        for idx, check in enumerate(checks):
            prev = checks[idx - 1] if idx > 0 else None
            xs.append(_feature_vector(lot, check, prev))
            ys.append(y)
            lots_used.add(lot_id)

    X = np.vstack(xs)
    y = np.array(ys, dtype=float)
    penalty = np.eye(X.shape[1]) * l2
    penalty[0, 0] = 0.0

    def objective(beta: np.ndarray) -> float:
        eta = X @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * np.sum((penalty @ beta) * beta))

    result = minimize(objective, np.zeros(X.shape[1]), method="BFGS")
    beta = result.x
    probs = expit(X @ beta)
    w = probs * (1.0 - probs)
    hessian = X.T @ (X * w[:, None]) + penalty
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
    return OutcomeModel(beta, covariance, len(ys), len(lots_used), l2)


def node_forecasts_for_lot(
    ws: Workspace,
    lot_id: str,
    as_of: int | None,
    model: OutcomeModel,
) -> list[dict[str, Any]]:
    """给出每个已送达检查节点上的剩余达标参考概率。"""
    lot = ws.lots[lot_id]
    checks = sorted(ws.visible_checks(lot_id, as_of), key=lambda c: c.observed)
    forecasts: list[dict[str, Any]] = []
    previous: Check | None = None
    for check in checks:
        p, lo, hi = model.predict(_feature_vector(lot, check, previous))
        forecasts.append(
            {
                "observed": check.observed,
                "received": check.received,
                "percent": check.percent,
                "reference_probability": p,
                "ci95": [lo, hi],
                "zone_supported": lot["zone"] != "new",
            }
        )
        previous = check
    return forecasts


def beta_interval(successes: float, total: float, level: float = 0.95) -> tuple[float, float]:
    """Beta 后验等尾区间；用弱先验 Beta(1,1)。"""
    from scipy.stats import beta as beta_dist

    a = 1.0 + successes
    b = 1.0 + total - successes
    alpha = (1.0 - level) / 2.0
    return float(beta_dist.ppf(alpha, a, b)), float(beta_dist.ppf(1 - alpha, a, b))


def marginal_for_lot(
    ws: Workspace,
    lot_id: str,
    as_of: int | None,
    model: OutcomeModel | None = None,
) -> dict[str, Any]:
    """单个批次在 as_of 时点的边际达标机会。"""
    status = ws.lot_status_at(lot_id, as_of)
    lot = ws.lots[lot_id]
    result: dict[str, Any] = {
        "lot": lot_id,
        "feed": lot["feed"],
        "zone": lot["zone"],
        "state_at_as_of": status["state_at_as_of"],
        "available": status["available_for_original_group"],
        "censored": status["censored"],
        "status_note": status["note"],
        "latest_visible": None,
        "probability": None,
        "ci95": [None, None],
        "conservative_interval": [0.0, 1.0],
        "support": {"moisture_model": False, "zone": lot["zone"] != "new", "joint": False},
        "decision_probability": 0.0,
    }

    if status["countable_success"] is True:
        result.update(probability=1.0, ci95=[1.0, 1.0], conservative_interval=[1.0, 1.0], decision_probability=1.0)
        return result
    if status["countable_success"] is False and status["state_at_as_of"] in {"fail", "scrap"}:
        result.update(probability=0.0, ci95=[0.0, 0.0], conservative_interval=[0.0, 0.0], decision_probability=0.0)
        return result
    if status["state_at_as_of"] == "transferred":
        # 本仓组单概率确定为 0；质量本身是删失/转仓，不伪造一个失败概率。
        result.update(
            probability=0.0,
            ci95=[0.0, 0.0],
            conservative_interval=[0.0, 0.0],
            decision_probability=0.0,
            group_probability_note="对当前仓组单不可计入，故组单层为0；不是质量失败标签",
        )
        return result

    latest = ws.latest_check(lot_id, as_of)
    if latest is None:
        lo, hi = beta_interval(0, 0)
        result.update(probability=0.5, ci95=[lo, hi], conservative_interval=[lo, hi])
        return result

    prev = ws.previous_check(lot_id, as_of)
    model = model or fit_outcome_model(ws)
    x = _feature_vector(lot, latest, prev)
    p, lo, hi = model.predict(x)
    result["latest_visible"] = {
        "observed": latest.observed,
        "received": latest.received,
        "percent": latest.percent,
        "previous_percent": prev.percent if prev else None,
        "as_of": as_of,
    }
    result["probability"] = p
    result["ci95"] = [lo, hi]
    result["support"]["moisture_model"] = True

    if lot["zone"] == "new":
        direct = sum(1 for x in ws.lots.values() if x["zone"] == "new" and x["period"] in {"train", "holdout"})
        zlo, zhi = beta_interval(0, direct)
        result["support"]["zone"] = False
        result["support"]["new_zone_history_lots"] = direct
        # 模型只能跨温区借用；没有 new 温区直接样本时，准入区间放回弱先验范围。
        result["conservative_interval"] = [min(lo, zlo), max(hi, zhi)]
        result["decision_probability"] = min(lo, zlo)
        result["zone_warning"] = "历史中没有 new 温区完结样本；logistic 数值仅是跨区参考，不能作为有把握准入依据"
    else:
        result["conservative_interval"] = [lo, hi]
        result["decision_probability"] = lo
    return result


def frechet_bounds(probabilities: list[float]) -> tuple[float, float]:
    """任意相依结构下 P(all) 的 Fréchet 界。"""
    if not probabilities:
        return 1.0, 1.0
    lower = max(0.0, sum(probabilities) - (len(probabilities) - 1))
    upper = min(probabilities)
    return lower, upper


def gaussian_copula_all(probabilities: list[float], rho: float) -> float:
    """单因子高斯 copula 下所有批同时达标的概率。

    rho=0 退化为独立；rho=1 趋近 comonotonic 上界。仅作相依敏感性，
    不声称历史已经识别了跨组 F5 的完整联合分布。
    """
    if not probabilities:
        return 1.0
    rho = min(1.0, max(0.0, rho))
    if rho == 0.0:
        return float(np.prod(probabilities))
    if rho == 1.0:
        return min(probabilities)
    thresholds = np.array([norm.ppf(1.0 - min(1 - 1e-12, max(1e-12, p))) for p in probabilities])
    scale = math.sqrt(1.0 - rho)

    def conditional_all(c: float) -> float:
        conditional = norm.cdf((-thresholds + math.sqrt(rho) * c) / scale)
        return float(np.prod(conditional))

    nodes, weights = np.polynomial.hermite_e.hermegauss(61)
    total = 0.0
    for c, weight in zip(nodes, weights):
        # Hermite-E 权重对应 exp(-x^2/2)；除以 sqrt(2π) 后正好是对标准正态 C 的期望。
        total += weight / math.sqrt(2.0 * math.pi) * conditional_all(float(c))
    return float(total)


def joint_assessment(
    marginals: list[dict[str, Any]],
    required: float,
    rho: float = 0.62,
) -> dict[str, Any]:
    unavailable = [m["lot"] for m in marginals if not m.get("available", True)]
    if unavailable:
        zero = [0.0, 0.0]
        return {
            "lot_count": len(marginals),
            "available_count": len(marginals) - len(unavailable),
            "unavailable_lots": unavailable,
            "marginal_probabilities": [],
            "independence_product": 0.0,
            "frechet_bounds": zero,
            "marginal_uncertainty_joint_range": zero,
            "scenarios": {"all": 0.0},
            "required": required,
            "conservative_lower": 0.0,
            "meets_required_conservatively": False,
            "point_lower_meets_required": False,
            "blocked_by_unavailable_lot": True,
            "no_dependence_assumption_note": "报废或转仓使原整组不可执行；这不是用相依性能修复的概率问题",
        }

    available = list(marginals)
    ps = [float(m["probability"]) for m in available if m.get("probability") is not None]
    lows = [float(m["conservative_interval"][0]) for m in available]
    highs = [float(m["conservative_interval"][1]) for m in available]
    lower, upper = frechet_bounds(ps)
    unc_lower, _ = frechet_bounds(lows)
    _, unc_upper = frechet_bounds(highs)
    product = float(np.prod(ps)) if ps else 1.0
    copula = gaussian_copula_all(ps, rho)
    scenarios = {
        "independence_product": product,
        "frechet_lower": lower,
        "frechet_upper": upper,
        f"gaussian_copula_rho_{rho:.2f}": copula,
        "perfect_common_cause_min_marginal": min(ps) if ps else 1.0,
    }
    decision_low = unc_lower
    return {
        "lot_count": len(marginals),
        "available_count": len(available),
        "marginal_probabilities": ps,
        "independence_product": product,
        "frechet_bounds": [lower, upper],
        "marginal_uncertainty_joint_range": [max(0.0, unc_lower), min(1.0, unc_upper)],
        "scenarios": scenarios,
        "required": required,
        "conservative_lower": decision_low,
        "meets_required_conservatively": decision_low + 1e-12 >= required,
        "point_lower_meets_required": lower + 1e-12 >= required,
        "no_dependence_assumption_note": (
            "Fréchet 下界不需要独立、同相关或 copula 假设；乘积只在所有特有波动独立时成立"
        ),
    }


def moisture_variance_components(ws: Workspace, node: int = 24) -> dict[str, Any]:
    """按 feed 单因素随机效应估计含水检测的共因/特有方差。"""
    cells: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for lot_id, lot in ws.lots.items():
        if lot["period"] not in {"train", "holdout"}:
            continue
        visible = ws.checks_by_lot.get(lot_id, [])
        match = [c for c in visible if c.observed == node]
        if not match:
            continue
        key = (lot["due"], lot["zone"])
        cells.setdefault(key, []).append((lot["feed"], match[0].percent))

    residuals_by_feed: dict[str, list[float]] = {}
    n = 0
    for items in cells.values():
        center = sum(v for _, v in items) / len(items)
        for feed, value in items:
            residuals_by_feed.setdefault(feed, []).append(value - center)
            n += 1

    feeds = list(residuals_by_feed)
    k = len(feeds)
    if k < 2 or n <= k:
        return {"node": node, "icc": None, "warning": "样本不足"}
    grand_mean = sum(v for vals in residuals_by_feed.values() for v in vals) / n
    ss_between = sum(len(vals) * (sum(vals) / len(vals) - grand_mean) ** 2 for vals in residuals_by_feed.values())
    ss_within = sum((v - sum(vals) / len(vals)) ** 2 for vals in residuals_by_feed.values() for v in vals)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    sizes = [len(vals) for vals in residuals_by_feed.values()]
    n0 = (n - sum(x * x for x in sizes) / n) / (k - 1)
    var_feed = max(0.0, (ms_between - ms_within) / n0)
    var_lot = max(1e-12, ms_within)
    icc = var_feed / (var_feed + var_lot)
    return {
        "node": node,
        "lots": n,
        "feeds": k,
        "variance_shared_feed": var_feed,
        "variance_lot_specific": var_lot,
        "sd_shared_feed": math.sqrt(var_feed),
        "sd_lot_specific": math.sqrt(var_lot),
        "icc_moisture": icc,
        "interpretation": "ICC 越高，共享供料/温区共因越不能通过拆单或相乘边际概率消除",
    }


def estimate_outcome_icc(ws: Workspace) -> dict[str, Any]:
    """用 beta-binomial 粗略估计完结结果的组内相关。"""
    grouped: dict[str, list[int]] = {}
    for lot_id, lot in ws.lots.items():
        if lot["period"] not in {"train", "holdout"} or lot["state"] == "transferred":
            continue
        grouped.setdefault(lot["feed"], []).append(1 if lot["state"] == "pass" else 0)
    counts = [(sum(v), len(v)) for v in grouped.values()]

    def neg_ll(params: np.ndarray) -> float:
        mu, phi = float(expit(params[0])), float(np.exp(params[1]))
        if not (0 < mu < 1 and phi > 0):
            return 1e10

        from scipy.special import betaln

        a, b = mu * phi, (1 - mu) * phi
        total = 0.0
        for x, n in counts:
            total += betaln(x + a, n - x + b) - betaln(a, b)
        return -total

    fit = minimize(neg_ll, np.array([1.0, math.log(1.6)]), method="Nelder-Mead")
    mu = float(expit(fit.x[0]))
    phi = float(np.exp(fit.x[1]))
    return {
        "beta_binomial_mean": mu,
        "precision_phi": phi,
        "outcome_icc_1_over_1_plus_phi": 1.0 / (1.0 + phi),
        "feed_groups": len(counts),
        "warning": "历史 H* feed 与整车基本一一对应；可估计车内共因，不能识别 F5 跨 G05/G07 的跨组联合结果",
    }


def estimate_outcome_icc_safe(ws: Workspace | None) -> float:
    if ws is None:
        return 0.62
    try:
        return float(estimate_outcome_icc(ws)["outcome_icc_1_over_1_plus_phi"])
    except Exception:
        return 0.62
