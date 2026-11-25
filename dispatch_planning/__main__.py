from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dispatch_planning.commitments import CommitmentStore  # noqa: E402
from dispatch_planning.report import write_report  # noqa: E402
from dispatch_planning.service import submit_day as submit_day_service  # noqa: E402
from dispatch_planning.workspace import Workspace  # noqa: E402


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="成品养护整车概率、备用舱与重复提交")
    parser.add_argument("--data", default=str(_ROOT / "data"), help="资料目录")
    parser.add_argument("--state", default=str(_ROOT / "commitments" / "ledger.json"), help="承诺台账 JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="生成 JSON/MARKDOWN 报告")
    report.add_argument("--as-of", type=int, default=None)
    report.add_argument("--out", default=str(_ROOT / "reports"))

    submit = sub.add_parser("submit-day", help="重复提交某日全部 active 组单")
    submit.add_argument("--ship-date", required=True)
    submit.add_argument("--as-of", type=int, default=None)
    submit.add_argument("--confirm", action="store_true", default=True, help="确认并锁定容量承诺（默认）")
    submit.add_argument("--draft", action="store_true", help="只保存草案，不确认容量承诺")

    verify = sub.add_parser("verify", help="核对某日承诺和情景容量")
    verify.add_argument("--ship-date", required=True)

    show = sub.add_parser("show", help="查看一张组单承诺")
    show.add_argument("--ship-date", required=True)
    show.add_argument("--truck", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ws = Workspace(args.data)

    if args.command == "report":
        paths = write_report(args.out, args.data, args.as_of, state_path=args.state)
        _print({k: str(v) for k, v in paths.items()})
        return 0

    store = CommitmentStore(args.state)
    if args.command == "submit-day":
        result = submit_day_service(
            ws,
            store,
            args.ship_date,
            args.as_of if args.as_of is not None else ws.current_clock("active"),
            confirm=args.confirm and not args.draft,
        )
        # 完整 verification 含全部承诺；CLI 摘要避免刷屏。
        summary = {
            "ship_date": result["ship_date"],
            "as_of": result["as_of"],
            "results": result["results"],
            "capacity": result["verification"]["capacity"],
            "submission_count": result["verification"]["submission_count"],
            "state_path": args.state,
        }
        _print(summary)
        return 0

    if args.command == "verify":
        _print(store.verify_date(args.ship_date, ws))
        return 0

    if args.command == "show":
        record = store.get(args.ship_date, args.truck)
        if record is None:
            print(f"没有 {args.ship_date} / {args.truck} 的承诺", file=sys.stderr)
            return 1
        _print(record)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
