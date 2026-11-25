# 成品养护组单资料交换

`data/` 收录供演练使用的合成记录，不含真实企业或人员资料。资料使用 UTF-8 逗号分隔，首行为字段名，带逗号的单值用双引号括起。日期采用 ISO 表示，空字段表示来源未填或不适用，不代表数值零。`sources.json` 记录各表列名、身份键和可直接核对的引用。

## 业务资料

- `lots.csv`：养护批的共同供料、温区、状态和开始，时间单位为逐日同源时钟的小时。
- `checks.csv`：逐批含水率百分点，observed 与 received 可不同。
- `trucks.csv`：组单拆分许可和最迟的替代离场窗口；required 是按整份承诺获准的最低达标概率。
- `spares.csv`：舱位为可独立承诺的物理资源，共享供料与温区导致的相关不因拆单而消失。
- `decision_points.csv`：是否能申请舱位按最后接收时点；下一次结果不可见时同组上次结果仍是现有事实。

不同阶段的时钟各自从零起算，不把训练、留出或当前组单的时点直接串在一起。pass/fail 是在所属组单时限内有明确信息的完结，scrap 不能继续供当前组，transferred 非当前仓完结，open 尚在观察。备用舱单个位置占一个批，若该位置已持有，只可用于指定组。

## 本地核对

需要 Python 3.11。分析环境可使用 `python3 -m pip install -r requirements.txt` 安装锁定的数值依赖。

从仓库根目录执行 `python3 check_sources.py`，返回各表行数；其他同格式资料可用 `python3 check_sources.py 路径` 检查。脚本只读资料并报告结构或引用错误，不给出业务结论。执行 `python3 -m unittest discover -s tests -v` 检查资料约定及拒绝重复身份的行为。

## 养护估计与调度承诺

新增代码位于 `dispatch_planning/`：

- `workspace.py`：规范化资料，并按 `received <= as_of` 给出时点视图；区分 open 删失、scrap 报废、transferred 转仓。
- `probability.py`：时点边际校准、Fréchet 相依界、独立乘积、高斯 copula 敏感性，以及 feed/批次方差分解。
- `validation.py`：train/holdout feed 整体隔离、温区整块留出和 holdout 整车核对。
- `planning.py`：整车/拆分评估、载具温区窗口匹配、决策截止和 F5 共因容量安排。
- `commitments.py`：JSON 台账、按发货日幂等提交、已装车/已履行动作保护、具名情景容量校验。
- `service.py` / `__main__.py`：当日组单提交、读数变化后的未履行部分更新和核对命令。

生成报告：

```bash
python3 -m dispatch_planning report --as-of 29
# 或 python3 dispatch_cli.py report --as-of 29
```

产物为 `reports/dispatch_report.json` 与 `reports/dispatch_report.md`。

### 当前演练数据的保守结论

- **G05 不能整组承诺 0.92**：F5 的 L09/L10 共享供料；as_of=29 时参考乘积约 0.735，Fréchet 点估计区间约 `[0.725, 0.768]`，计入边际不确定性的保守下界约 0.42。G05 不允许拆分，因此没有“有把握整组发运”的计划。截止前安排为 `Z01#2 -> L09`（必须在 26 前按已送达读数决定，不能等 28 的读数）和 held-for G05 的 `Z03#1 -> L10`（可等 27 读数并在 28 前接收）。
- **G06 可确定发运已装车的 L11**：其保守下界约 0.967，达到 0.86；L11 已装车，不能再搬回去占备用舱。L12 的 new 温区没有历史样本，且 29 才收到的 24h 读数反弹到 10.9，不能作无条件发运承诺。可保留 `Z02#1 -> L12`，在 43 读数可见后、最迟 45 前决定；该舱窗口只到 55。
- **G07 可按合同拆分先发 L14**：L14 保守下界约 0.958，达到 0.90。L13 保守下界约 0.774，安排 `Z01#1 -> L13`，必须在 26 前决定，不能以等待 28 读数为条件。
- **F5 跨 G05/G07 的容量隔离**：`Z01#1` 与 `Z01#2` 是两个不同位置；同一 F5 联合坏情景下 G05 的 L09 与 G07 的 L13 都会需要 Z01，不能把同一个位置同时许给二者。若两份承诺只在两个互不相交的具名情景下生效，`validate_capacity` 会把它们记为同一位置的或有共用，而不是永久双倍占用。

概率上不把独立乘积当保证：任意相依结构下，所有 `n` 批同时达标的 Fréchet 界为

```text
max(0, Σp_i - (n-1)) ≤ P(all) ≤ min(p_i)
```

24h 检测的 feed ICC≈0.75、完结结果 beta-binomial ICC≈0.62，说明共因很强。历史 H* feed 与整车基本一一对应，因此只能支撑车内共因估计；F5 跨 G05/G07、new 温区、超过 53 的 54/56 改期窗口以及备用舱干预效果均缺少直接联合样本，报告只给区间/敏感性，不伪造精确联合概率。

## 重复提交与核对

默认台账是 `commitments/ledger.json`。同一发货日 + 组单重复提交相同内容会返回 `duplicate`，不会新增记录：

```bash
python3 -m dispatch_planning submit-day \
  --ship-date 2026-11-25 --as-of 25 --confirm

# 完全相同的命令可安全重复执行
python3 -m dispatch_planning submit-day \
  --ship-date 2026-11-25 --as-of 25 --confirm

# 读数送达后重提：只更新尚未履行/未装车的动作
python3 -m dispatch_planning submit-day \
  --ship-date 2026-11-25 --as-of 29 --confirm

python3 -m dispatch_planning verify --ship-date 2026-11-25
python3 -m dispatch_planning show --ship-date 2026-11-25 --truck G06
```

下一发货日可使用新资料目录和新的日期键：

```bash
python3 -m dispatch_planning --data data-next \
  submit-day --ship-date 2026-11-26 --as-of 24 --confirm
python3 -m dispatch_planning verify --ship-date 2026-11-26
```

调度确认后，每张备用舱动作都记录保留的具体位置、小时窗口、载具/温区、决策截止、触发情景和“不覆盖”清单。已经装车或状态为 `fulfilled` 的动作不能更改 lot、spare 或 unit；尝试删除/替换会抛出 `ImmutableCommitmentError`。
