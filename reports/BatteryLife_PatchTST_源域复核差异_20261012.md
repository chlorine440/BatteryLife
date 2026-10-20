# PatchTST：训练正常退出，源域重载复核未通过

记录时间：2026-10-12 03:16–03:52（北京时间）。这是失败诊断记录，不是
成功模型的正式成绩，也不是已经精确证明根因的报告。

## 已证明的事实

- 实际 suite：`benchmark95_20261011_v4`，模型 PatchTST，种子2021。
- 原训练入口运行10轮，正常退出（exit_code=0），最佳源域验证 MAPE 在
  第4轮：0.4698122。第1、3、4轮保存，共3次保存诊断全部通过。
- 保存前/磁盘中的完整状态逐张量相等；诊断副本的预测相等；两次独立
  重载预测相等且有限。没有证据表明权重丢失或保存损坏。
- 原日志最佳 Test MAE 为199.0834圈。独立重载比较报错为
  `mae_cycles: 0.30109338595070767`，因此重载 MAE 约为199.3845圈
  （由比较差值与日志数值还原，不冒充新的成功正式评分）。
- 差异超过固定0.002圈容差；严格比较抛出 ValueError，任务记录为failed。
  没有放宽容差或改成“通过”。源域预测 NPZ 在比较通过之后才保存，
  因此此失败种子目前没有通过审核的正式预测数组。
- 本任务训练数据、标签、划分、模型、参数和优化流程未被修改。训练为
  原入口实际启用的 DeepSpeed 0.15.0 ZeRO-2、FP32、Adam，不是仅凭
  `--multi_gpu` 字样推断的普通 DDP。实际日志明确打印上述引擎/优化器。

## 有依据、但未精确证明的原因方向

PatchTST 原模型 `encoder.norm.1` 是普通 BatchNorm1d，含running_mean、
running_var 等缓冲区。原源域评估用 `accelerator.gather_for_metrics` 汇总
两张卡的预测；当前独立评估使用已保存的一份 checkpoint。

已检查服务器安装版本的 `DeepSpeedEngine`：`_broadcast_model` 遍历
`named_parameters()`，不是BatchNorm的运行统计缓冲区。保存模型中该层
`track_running_stats=True`、`num_batches_tracked=3536`，最小running_var为
6.708987143611012e-9，最大为0.032042261213064194。

这些事实支持“多卡局部运行统计量/数值敏感性导致日志汇总与单份重载
不一致”的排查方向。但是没有保存当时另一张卡的完整缓冲区，尚未用
实际训练现场精确复现0.301圈差异；不能将这一推断写成已证实根因，
也不能据此断言商飞数据有问题或PatchTST训练算法失败。

## 保留的权重证据

服务器目录为：

```text
/root/autodl-tmp/BatteryLife/dataset/mixed95/benchmarks/benchmark95_20261011_v4/runs/PatchTST/seed_2021/checkpoint/PatchTST_sl1_lr0.001_dm64_nh4_el2_dl2_df64_lradjconstant_datasetMIX_large_lossMSE_wd0.0_wlFalse_bs16_s2021-benchmark95_20261011_v4_formal
```

只读核对的SHA256：

- model.safetensors：`39fb46a240bc43e4a30943845e6a33b3b95adb1366300729a69f72dcf2742c4c`
- args.json：`c0180d2a443c2160cfd277da12053e8f62f1d102abcd17e500640ee238ca625a`
- label_scaler：`534ecde69bf48c82460a2c6c0eb178655c608415f2e6a6addfba4fd1bc1e1a6d`
- life_class_scaler：`db0c6f4785ec780b3e63d18852c4fb73d32a984d925ba800dc22fa5833bd9aa9`
- checkpoint_reload_probes.json：`ca6d9af00b72938f9b32ec3f8c1d5b483609c8826c5a37bfb5d69ab9f8b40a06`

## 当前处理

03:27追加：seed=42也正常训练退出（exit_code=0），运行13轮，最佳源域
验证MAPE在第8轮（0.3822638）。5次保存诊断全部通过，两次独立重载
预测一致且有限；原日志Test MAE为210.8005圈，重载比较的有符号差值为
`+0.8269732144673583`圈，超过同一0.002圈容差，因此也记录为failed。
该值是`actual - logged`，重载MAE可由它与日志还原为约211.6275圈；
仍不是通过验收的正式成绩，不生成成功源域预测数组或商飞成绩。
此任务包含加载、训练、复核的总耗时为1210.5736秒。

seed=42保留目录与上述相同的suite，位于`runs/PatchTST/seed_42/checkpoint/`。
只读核验SHA256：

- model.safetensors：`a1765b694d7f8a2f04e861c132f89bed4fba1c519082a2f275084c1a0619371e`
- args.json：`838262aadf6d8006cb5332f7f4b079a0c496e18d66667280c38e852dc6095718`
- label_scaler与life_class_scaler均与seed=2021上述哈希相同。
- checkpoint_reload_probes.json：`89ad267ba372545a31c30a6b0dda10417594ce96f8e2ce9c388e7b880abf7878`

第二次出现相同类型差异使其更值得单列诊断，但不代替对根因的实证。
尚未修改训练算法、BatchNorm或验收方式。03:27确认第三个种子2024已
启动（子进程PID708328），协调器PID730280仍在运行；其余模型继续排队。

03:30已将上述两个种子的完整日志、参数、权重/scaler和保存诊断同步到
本机对应suite目录。再次逐文件计算SHA256，与服务器证据全部相等；
完整解析参数与checkpoint args一致，3/5次保存诊断通过，成功源域NPZ
均不存在。记录为`failed_PatchTST_copy_integrity_20261012.json`，scope仅
为传输完整性和失败证据核验，final=false；不代表源域指标验收通过。

03:51追加：seed=2024运行18轮后正常退出（exit_code=0），最佳checkpoint
在第13轮保存，最佳验证MAPE约0.3548。4次保存诊断通过，两次独立重载
预测相等且有限。日志Test MAE=189.4749圈，重载比较差值为
`+0.5093962265014795`圈，超过同一0.002圈容差，也记为failed；由有符号
差值还原的重载MAE约189.9843圈，仍不是通过验收的正式成绩。
总耗时1404.2576秒，包含加载/训练/核验。第三个种子权重SHA256为
`eb9e58b88a5ce4438cd5456ea9f5a662cd4a5c97f341b69c1f8ed19e9c31d319`，
args.json为`a6e6a8e33dafb1584f65c4fd3ae8d4ce8f5a5378f3d599913eb0e24de30e2195`，
保存诊断为`4e752c0a3bb0456010536e9b3dd12c2b831d608f7d246c1b22d8c7cd2c3a3925`。
两个scaler哈希与前两种子相同。全部三种子的权重、参数、日志和诊断已
同步本机；参数相等、权重等五项文件哈希相等、3/5/4次保存诊断通过，
三个成功源域预测NPZ均不存在。合并核验记录为
`failed_PatchTST_copy_integrity_all3_20261012.json`，final=false，仅证明
失败证据/文件完整性，不证明源域成绩或根因。

PatchTST三个种子现均为真实终态failed；按冻结方案不能进入商飞评价，
其最终对比行留空成绩并给出三个复核差异。其他模型不受此失败影响：
03:51确认Autoformer seed=2021已启动，子进程PID940969，协调器730280
仍在运行。根因仍以“未精确证明的BatchNorm/多卡统计方向”记录；本次
没有通过改模型或放宽容差来使它通过，也不将此视为商飞数据的问题。

保留训练日志、权重和失败原因，不重训、不改变BatchNorm、不改多卡策略、
不放宽指标容差，不把这三个种子算作正式成功。队列按已约定的失败隔离
继续后续模型。只有三个种子全部成功的模型才进行商飞评价；PatchTST
本轮不满足此条件，其对比行将写明失败原因、正式成绩留空。

本次调查没有运行商飞推理、重新选择checkpoint、进行目标调参，或覆盖
已完成模型的正式结果。整套实验尚未完成。
