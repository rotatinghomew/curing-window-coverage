# 调度承诺台账

默认文件 `ledger.json` 由以下命令维护，不要手工拼接重复键：

```bash
python3 dispatch_cli.py submit-day --ship-date YYYY-MM-DD --as-of <hour> --confirm
python3 dispatch_cli.py verify --ship-date YYYY-MM-DD
```

台账按 `ship_date:truck_id` 幂等。相同语义的内容重复提交只会返回 `duplicate`；读数变化后重提会追加历史版本并更新尚未履行的动作。已装车或 `fulfilled` 的核心资源字段不可更改。

当前 `ledger.json` 是 2026-11-25 演练日的示例：先在 `as_of=25` 截止前提交，再在收到延迟读数后的 `as_of=29` 核对。
