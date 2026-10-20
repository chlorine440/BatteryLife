# 13 模型 / 95% 商飞跨域 benchmark

> 最终套件为 `benchmark95_20261013_autoformer_retry_v1`。2026-10-13
> 18:43完成最后一个Autoformer源域复核及六条正式商飞评价，原66条复用。
> 13模型范围全部形成终态：12模型/36有效种子、72目标预测；PatchTST
> 三次源复核失败保留，不填成绩。最终全量源哈希、81项回归、独立算术
> 和目标重放验收通过，见最终报告。旧v1/v2/v3/v4留作历史，不另行启动。

冻结方案记录在 v1；最终实际套件是 `benchmark95_20261013_autoformer_retry_v1`，
与旧 v3 的权重引用/源域复核修复过程见下方历史记录。
CPTransformer 导入既有审计通过的
三个种子和六条商飞结果；新增12模型×3种子原36次尝试，外加用户明确
批准的Autoformer2024一次基础设施重训，不重跑其他已完成训练/评价。

## 不变的协议

- 现有 MIX_large 文件、95% 标签、283/89/96 个训练/验证/测试电芯、8 个有效数据集。
- 原 `run_main.py`、模型及曲线特征算法不改；加载器仅修复授权的无标签
  分支返回数量（5个None改6个）。Adam/MSE/FP32/恒定学习率，双GPU ZeRO-2。
- `seq_len=1`、早期窗口 1–100、每圈 3×300；每进程 32 workers。
- 验证 MAPE 选 checkpoint；原入口仍每 epoch 计算源域测试。
- 100 epoch 上限、至少 5 epoch、patience=5；不搜索参数，不改变 batch、精度或模型层数。
- 额定容量/数据集 SOC 区间的原特征处理不修改；训练电芯单独拟合寿命 scaler。
- 商飞 I/Q 特征÷183；标签 `Q_discharge÷0.94÷183`，C1=379、C2=419。
  标签是日常循环代理寿命，不是完整 RPT 的实测阈值寿命。

### 循环编号与特征边界

源域沿用原加载器的编号：`cycle_data`列表的记录顺序重新编号为1、2、…；
95%源标签也使用同样的记录顺序，而不是另用原始`cycle_index`。
源域的100圈窗口因此指前100个预处理循环记录；原始测试若有删圈/缺圈，
不能未经核对将它解释成原始仪器日志的物理圈号。商飞单独保留全局物理
1–100圈，第一圈RPT置零，四个CP-RNN兼容适配仍只使用物理2–100圈。

原`read_cell_df`在退出循环前会先临时追加第101条记录；实际特征提取
`get_charge_discharge_curves`只遍历1–100，容量、电流和电压处理都在选定
单圈内部完成，第101条不会成为模型曲线或其归一化统计。电压按该圈
重采样曲线最大电压归一化；电流/容量按额定容量归一化，商飞为183 Ah。
这项说明仅明确原代码和标签的共同计数依据，没有修改数据、模型或加载器。

## 配置与可追溯性

`benchmark_config.py` 用 AST 读取原入口默认参数，用受限文本解析读取 shell
**实际传入**的参数，不执行 shell、不读取未传入的同名变量。CPTransformer、
CPMLP、CPGRU、CPLSTM 优先应用公开 Li-ion 参数表；另外九个使用仓库脚本。
CPGRU 实际 `lstm_layers=4`、CPLSTM=2；不能把不生效的 d_layers 当循环层数。
全部参数、参数来源和源文件 SHA256 保存在 manifest；不是严格 v4 复现。

`benchmark_baseline.json` 冻结 2026-10-09 审计的 843 个源文件哈希、划分与
标签决定、核心代码、商飞输入、既有 CPTransformer 权重/结果哈希与独立复核指标。
预检流式重新计算所有源文件 SHA256。数据不复制、不重新预处理、不覆盖 80% 标签。

`benchmark_worker.py` 调用未经修改的 `run_main.py`。它仅观察
`Accelerator.save_model`：保存前取选中模型的状态快照，与磁盘状态及重载后
合成输入预测逐项精确核对。诊断使用副本、不改变训练模型模式/权重；恢复 CPU
随机状态。它不接触商飞数据，不改优化器、梯度、损失或选型规则。
原 Autoformer 的 AutoCorrelation 内含 `.cuda()`；诊断和目标推理用 CUDA，
不为支持 CPU 修改其代码。其他模型诊断/商飞推理默认 CPU。

包装入口会从 `sys.path` 中移除自身目录，保留仓库根目录，避免同目录
`evaluate.py` 遮蔽原入口使用的第三方 evaluate 包。每次运行前，通过实际直接
脚本 `benchmark_worker.py --import-check` 检查模块位置及 `evaluate.load`，
再用实际包装入口的 `--help` 检查原入口全部依赖导入及参数解析。
后者在原解析器退出，不初始化 accelerator、不加载训练数据；公共导入失败
停止整个套件，避免把公共环境问题当作 12 个模型的科学失败。
修复已同步到服务器并通过同一组 19 项测试。

## RNN 缺口适配

公共 NPZ 始终是 100 个物理位置，第一圈 RPT 为零且 mask=0。仅 CPGRU、
CPLSTM、CPBiGRU、CPBiLSTM 将有效物理 2–100 圈排至前 99 格，尾格为零。
数值不变；保存每格物理圈映射；映射 0 表示 padding，不是新圈号。
源域连续有效前缀不变，其他九个模型保留物理位置。绝不引入第 101 圈。

## 服务器命令

从服务器仓库根目录运行，用已安装的环境，不租卡/扩盘/清理旧实验：

```sh
cd /root/autodl-tmp/BatteryLife
dataset/.conda_envs/mixed95/bin/python -m unittest experiments.mixed95.test_benchmark -v
dataset/.conda_envs/mixed95/bin/python -m experiments.mixed95.benchmark dry-run --suite-id benchmark95_20261011_v4 --check-data
# 队列当前已在运行，不要重复执行 train。确认无现存协调器后才可显式续跑：
dataset/.conda_envs/mixed95/bin/python -m experiments.mixed95.benchmark train --suite-id benchmark95_20261011_v4 --models iTransformer MICN Autoformer --then-evaluate
```

`train --with-smoke` 先做所有选中模型的完整配置一 epoch 试跑，再顺序运行正式
三个种子。`--then-evaluate` 等全部源任务结束才做目标评价和报告。每次只用一个
双 GPU 任务。完整 source SHA256 和 CPTransformer 导入核验通过后才启动。

也可分阶段运行：

```sh
dataset/.conda_envs/mixed95/bin/python -m experiments.mixed95.benchmark smoke --suite-id benchmark95_20261011_v4
dataset/.conda_envs/mixed95/bin/python -m experiments.mixed95.benchmark train --suite-id benchmark95_20261011_v4
dataset/.conda_envs/mixed95/bin/python -m experiments.mixed95.benchmark evaluate --suite-id benchmark95_20261011_v4
dataset/.conda_envs/mixed95/bin/python -m experiments.mixed95.benchmark report --suite-id benchmark95_20261011_v4
```

`--models CPMLP CPGRU` 可限定源任务，但目标阶段仍需全套源任务达到终止状态。
`--suite-id` 可另存实验；不要重复启动同一服务器的训练。套件锁防同 suite 重入；
GPU 空闲与磁盘检查防资源冲突。已完成/失败任务都不自动重跑；中断任务记失败。
OOM/NaN/运行失败保存原因并继续其余模型，失败试跑阻止该模型三个正式种子；
公共数据/环境失败停止整批。商飞评价一旦开始即设保护状态，中断也不自动重测。

## 产物与统计

实际运行目录：`dataset/mixed95/benchmarks/benchmark95_20261011_v4`（已忽略）。
为兼容历史命令，入口未传 suite-id 时仍默认 v1；本轮命令必须显式传入 v4。

### 2026-10-11 首次启动诊断

第一次启动的 v1 在源文件/旧 CPTransformer 核验通过后，CPMLP/CPGRU 仍处于
试跑源数据加载阶段。启动器按字节偏移读取文本日志时恰好切到 UTF-8 进度条字符
中间，触发解码异常。已停止该协调器及其当前试跑进程，保留原目录；没有新的
正式训练或商飞评价结果。这不是科学模型的失败成绩。

修复版本先读取字节再使用容错 UTF-8 解码；新增多字节边界及完整试跑监控回归
测试。本地 benchmark 16 项测试通过；原有 23 项数据/标签测试通过。
旧 v1 身份哈希已与修复版本不同，保护机制会拒绝直接续跑覆盖它。
2026-10-11 08:58 核验服务器无训练进程、两张 GPU 空闲。用户恢复 SSH 后，
已同步日志修复，服务器 16 项测试通过，本地/远端六个关键文件哈希一致。
随后启动新的独立 suite（协调器 PID 16064）：

```sh
dataset/.conda_envs/mixed95/bin/python -m experiments.mixed95.benchmark train --suite-id benchmark95_20261011_v2 --with-smoke --then-evaluate
```

v1 原目录保留，并单独标注基础设施中断；没有把它当成科学模型失败成绩。
v2 使用相同的冻结实验协议；后台协调器不依赖 SSH 终端继续运行。
v2 随后遇到包装脚本目录遮蔽第三方 evaluate 包的问题；09:49 核验前五个
试跑均在首轮计算之前因同一原因失败，MLP 仍加载源数据。已确认停止协调器
16064 与当前试跑进程组 23678，保留所有记录，单列为基础设施中断。
修复后启动 v3；查当前进度应读取 v3 的 manifest 和
`dataset/mixed95/setup/benchmark95_20261011_v3_console.log`。
不要把测试用的模拟 COMPLETE 输出或启动成功当成正式训练完成。

- `manifest.json`：完整配置/状态/命令/权重与 scaler 索引/运行时间/轮询显存峰值。
- `smoke/`、`runs/`：逐模型逐种子日志、checkpoint、保存前后诊断记录。
- `source_test_predictions.npz`：9600 条源测试窗口预测、标签、圈数、电芯 ID。
- `target_mapping/`：商飞各模型的物理圈映射；NPZ 保持不变。
- `comparison.csv`、`report.json`、`REPORT.md`：固定 13 行，无成绩项留空并注明原因。
- `comac_predictions.csv`：成功模型最多 78 条模型×种子×电芯预测，含导入的 CP 六条。
- `comac_mape.png`：各完整模型商飞 MAPE 及三种子总体标准差。

源测试报告 1–100 窗口汇总、仅第 100 圈窗口和统一去掉 `ISU-ILCC_G43C3.pkl`
的敏感性分析。正式 96 电芯指标不排除它。独立预测数组核算
MAE/RMSE/MAPE/15%-Acc/10%-Acc，并与原日志容差核对。
MAPE 存储为比例，acc15/acc10 存储为百分数，MAE/RMSE 单位是圈。
聚合只用于三个种子均完整的模型，标准差 ddof=0；六条商飞预测只对应两枚电芯。
CPTransformer 原实验未记录时间/峰值显存，这些字段标不可用，不补造。

C1/C2 已在此前实验看过，本轮不是新的盲测，也不根据目标成绩调参或选 checkpoint。
后续方法开发若使用本轮目标结果，需要新的独立测试数据。

## v3 推理后端审计修复（v4 已通过重核并恢复训练）

21:16 恢复完成：11 个已训练 checkpoint 均通过相同容差的重核，两个
CPBiGRU 的准确率差异消失。随后独立使用 NumPy 对 11 个预测数组复算
五项指标，全部与审核记录一致；旧 manifest、权重/scaler 文件 SHA256
均保持不变。确认差异来自独立推理后端设置，不是训练失败。
v4 自动继续原队列；21:22 实际状态为 11 次完成、CPLSTM/2024 正在运行、
24 次待运行，新增商飞评价尚未开始。当前源任务查看 v4，不再运行 v3。

晚间：CPLSTM seed=42 完成后，在任务边界退出旧协调器；两张 GPU 空闲。
修复已部署，远端 22 项测试通过，关键代码哈希与本地一致。
v4 恢复进程 PID 730280 已启动；正式新增状态仍以 v4 的 manifest 及
`dataset/mixed95/setup/benchmark95_20261011_v4_console.log` 为准。
旧 v3 有 11 个已结束训练的 checkpoint（9 个审核通过、2 个待修复审核），
另外 25 次待运行。当前不声称实际指标重核已通过或新增商飞评价已完成。

v3 的原训练始终调用未改动的 `run_main.set_seed`，其中 cuDNN 被关闭。
独立 checkpoint 推理原先未同步此设置，CPBiGRU 两个种子的阈值准确率
复核相差 1–2 个窗口。训练正常结束，不据此重新训练或放宽容差。
`benchmark_inference.configure_inference_backend` 现在同步上述原后端设置。

在旧协调器和当前任务确实结束、GPU 空闲后，显式恢复到独立 suite：

```sh
dataset/.conda_envs/mixed95/bin/python -u -m experiments.mixed95.benchmark_recovery \
  --source-suite benchmark95_20261011_v3 --suite-id benchmark95_20261011_v4
```

恢复入口核对旧数据/标签、配置、训练包装器和 checkpoint 哈希，保留历史
manifest；只引用既有权重，重新做源域严格审核。两次已定位审核失败优先
验证；若仍不匹配则停止恢复，不放宽验收。真正训练失败、运行中及未完成
任务不能当成已训练权重导入。恢复通过后只启动原队列尚未训练的任务。
旧 v3 的日志、数组和权重不覆盖；新的后端设置和审核结果记录于 v4。

## 独立结果交付核验（不重新推理）

`audit_benchmark_results.py` 不调用 benchmark 的指标/聚合函数，不实例化模型，
直接用保存的预测数组重算五项指标、仅第 100 圈指标及 95 电芯敏感性分析。
再核对三种子总体标准差、scaler/完整配置、权重哈希、物理圈映射、目标
逐条误差，以及最终 report.json 的明细/比较表/预测。最终模式拒绝尚在运行
的源任务；`--live` 明确标记 final=false，不能用作全套完成证明。
最终模式同时只读 comparison.csv / comac_predictions.csv，对照 JSON 检查
列名、记录唯一性、每个数值和空白字段；真实 0 不得代替缺失。
新增回归测试后独立核验测试共 6 项，服务器全部通过；已验证当前部分结果
快照的13行/54列对比表与六条旧 CP 商飞记录，不能据此声称全套已完成。

CPTransformer 使用此前独立审计的三个源域预测 NPZ（哈希固定），不再次
预测。它们作为约 107 KB 的核验输入放在 v4/audit_inputs，不是新的训练数据。
首次 live 核验通过 14 份源结果（原 CP 三份 + 新增十一份），只包含既有 CP
六条商飞预测；新增商飞结果尚未产生。核验也补齐 CP 的敏感性 10%-Acc。

```sh
dataset/.conda_envs/mixed95/bin/python -m unittest experiments.mixed95.test_audit_benchmark_results -v
dataset/.conda_envs/mixed95/bin/python -m experiments.mixed95.audit_benchmark_results \
  --suite dataset/mixed95/benchmarks/benchmark95_20261011_v4 \
  --cp-arrays dataset/mixed95/benchmarks/benchmark95_20261011_v4/audit_inputs --live
# 全套完成后，去掉 --live 做最终交付核验；不重新训练或预测。
```

结束后完整回归命令（原边界/Excel 23 项、benchmark 22 项、独立核验
8 项、顺序控制 8 项，共 61 项）：

```sh
dataset/.conda_envs/mixed95/bin/python -m unittest \
  experiments.mixed95.test_mixed95 experiments.comac95.test_data \
  experiments.mixed95.test_benchmark experiments.mixed95.test_audit_benchmark_results \
  experiments.mixed95.test_priority_handoff -v
```

本机替换 Python 路径为 `dataset/.conda_envs/comac95/bin/python`。远端
正式队列运行期间不要执行会实例化模型的服务器回归测试，避免抢占 GPU。
本机有原 Excel 时必须确认两个 `test_real_*` 实际通过而非 skipped；
远端没有原 Excel 时的跳过项只由本机实际读取验证补足，不写成远端通过。
训练中通过的回归不替代全部任务结束后的最终验收。

## 用户指定的后续队列顺序（2026-10-12）

用户选择保留当前Autoformer seed=2021的完整训练，等它自然结束、原
checkpoint/源域复核形成终态，再优先训练iTransformer和MICN；Autoformer
seed=42/2024最后跑。独立`priority_handoff.py`只管理Linux调度器边界，不
修改冻结的模型、训练、推理代码或超参数。它核对实际PID及启动标识，
禁止停止训练worker，禁止在另一个源任务已启动时接管；既有源结果、
smoke和商飞记录必须不变。接管仍调用原benchmark入口，同一v4 suite，
模型顺序`--models iTransformer MICN Autoformer --then-evaluate`，成功/失败
种子不重跑，全部39个源任务终态后才评商飞的门槛不变。

服务器控制状态见`priority_handoff.json`，实际接管后的PID与命令也记录
其中；`waiting_current_trial`不是已切换或全套完成的证明。边界前的完整
manifest保存为`priority_handoff_boundary_manifest.json`，新的队列日志为
`priority_queue_console.log`。6项只用合成数据的控制测试：

```sh
dataset/.conda_envs/mixed95/bin/python -m unittest experiments.mixed95.test_priority_handoff -v
```

实际移交：当前Autoformer seed2021在10:58完成训练及严格源复核，旧
调度器在完成边界退出。即时suite锁检查曾失败；旧控制记录及边界manifest
完整保留。12:19确认锁空闲、全部28份已完成源权重哈希不变后，通过原
benchmark入口恢复同一v4套件，新协调器242946；12:30 iTransformer
seed2021正式训练开始。恢复记录另存priority_handoff_recovery.json，不
覆盖原移交失败；没有重训已完成的Autoformer种子。控制程序后来增加
30秒有界锁释放等待，8项控制测试在本机及服务器通过，旧armed版本
另存为priority_handoff_armed_version.py。此为调度保护，不改变模型、
训练配置、数据、选择规则、原失败状态或全部源任务结束后再测商飞的门槛。

## 用户授权的11模型阶段评价（2026-10-13）

用户随后明确要求暂停Autoformer，先测试已有11模型。10:25以SIGSTOP
暂停原协调器242946及Autoformer seed2024精确后代69进程（包括训练
rank和数据worker），核对全部状态T。没有杀进程、覆盖权重、改source
终态或绕过原suite锁写入主manifest。首次按进程组识别仅找到launcher，
安全断言在发信号前拒绝，随后按真实后代树精确暂停。

新增`staged_evaluate.py`只写独立`v4/staged11_20261013/`。11模型均要求
三个源种子complete，排除Autoformer/PatchTST；CPTransformer只导入
此前6条结果，其余10模型各新增6条。此门槛是本次用户授权的协议修改，
不能声称原“39源任务先终态”门槛已满足。原训练/模型/加载器/标签/
输入和scaler不变；CPU/FP32沿用原target_evaluation的缺省推理设备。

四项安全门槛测试本机/远端通过；阶段入口有独立锁，已有阶段manifest
则拒绝重复推理。测试前完整843份源SHA256核验及源数值审计通过；
10:33阶段完成66条唯一预测。本机独立重算全部单条误差、三种子五项
指标及总体标准差、各电芯汇总，1e-9一致，检查20份实际圈映射。

Autoformer继续保持暂停，未授权自动恢复。恢复完整队列前必须合法
交接主suite锁并导入已完成阶段的目标记录，保证已有权重不再测商飞；
不可直接SIGCONT原`--then-evaluate`协调器并任其按旧pending状态
重复推理。阶段产物含future_reuse_required说明，不改冻结核心代码。

## 10月13日恢复与明确授权的单种子重训

14:38用户要求继续训练后，`resume_paused.py`只恢复原Autoformer2024
的69个后代，旧协调器保持停止以保留真实子进程退出码。NCCL日志记录
ALLREDUCE等待15161351毫秒，超过600000毫秒限制，两rank以SIGABRT
退出，launcher实际exit=1。此前约4.2小时的进程暂停没有实现无损续训；
这是暂停/恢复引起的运行失败，不是Autoformer模型数值失败，也不应
声称成功续训。原代码没有优化器/调度器断点，不能从第4轮继续。

训练真正退出后才终止旧协调器并合法取得原suite锁。11模型阶段66条
目标结果导入主manifest、无再次正式推理；843源文件完整SHA核验及
终态数值审计通过。v4的Autoformer2024失败、目标blocked_incomplete，
PatchTST原三个失败保持不变。控制记录及原日志全部保留。

用户随后明确批准“重训这个种子，完成Autoformer”。`explicit_retry.py`
仅为这个已确认的基础设施失败创建新套件
`benchmark95_20261013_autoformer_retry_v1`，原v4终态不覆盖；旧35份
有效源记录及11模型66条目标记录引用/复用。只有Autoformer2024和
尚未进行的Autoformer目标组在新套件设为pending，PatchTST不重试。
调用冻结原入口、原双卡FP32/参数/验证选权重/失败隔离。输出路径及
运行标识改变，不改变学习率、batch、模型结构、数据或标签。

14:47实际协调器768031及launcher768976存活，源数据加载中；旧
Autoformer2021/42被原入口明确跳过。15:01已进入第1轮1070/3538
迭代，GPU利用率90%/98%，旧66条目标结果仍逐值保留。开始运行
不是正式训练完成证明。
结束后只有Autoformer三种子全通过才追加6条新商飞预测；其他66条必须
逐值不变。新套件独立最终数值核验及旧suite/阶段结果哈希保护由该入口
自动执行；交付与全部验收仍需核对实际终态。恢复/重训控制的11项测试
在本机及服务器通过。不要再以SIGSTOP长时间暂停正在双卡训练的任务。

重训套件的最终数值核验仍按当前suite的`runs/<model>/seed_<seed>/`
读取源预测数组。创建新套件时仅引用旧checkpoint不够：15:13补上32份
已接受非CP源数组的文件级symlink，原数组SHA256、9600窗口/第100圈/
9500敏感性五项指标逐份核对。`retained_source_links.py`不写任何manifest、
旧数组或正在训练的Autoformer2024路径；证据为新suite的
`retained_source_array_links.json`。五项控制测试本机与远端通过。
此为交付/核验路径补齐，没有重新源推理、商飞推理或改变训练。最终
下载新suite时要读取这些数组链接的内容，不能交付指向服务器的坏链接。
