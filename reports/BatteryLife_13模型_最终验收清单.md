# BatteryLife 13模型最终验收清单

日期：2026-10-13。最终套件为benchmark95_20261013_autoformer_retry_v1。
完成约定的13模型范围及失败隔离流程：**12模型有三种子有效商飞成绩，
PatchTST三个源复核失败保留，无虚构成绩**。这不表示13模型都成功。

结果与方法：[最终报告](BatteryLife_13模型_最终结果与验收_20261013.md)。
证据位于最终suite的final_audit_20261013_v1目录。

|要求|实际验收证据与结果|
|---|---|
|13模型范围|最终manifest和13行/54列comparison.csv恰为约定白名单；CP三种子导入，新增12模型各三个种子都有真实终态|
|任务与进程|39个唯一模型×种子记录：36通过、3失败；36次原新增尝试外仅有用户批准的一次Autoformer2024基础设施重训。768031/768976已实际退出，最终没有活动训练进程|
|完整试跑|12个完整配置一轮smoke均通过；结束后重新核对全部smoke权重哈希和保存前后诊断，不把smoke用于目标评价|
|数据及隔离|843候选、283/89/96有效、8个有效来源（源测试7个），468寿命均>100；MIX_large顺序、无重叠及独立标签重算通过|
|结束后数据哈希|end_of_run_data_preflight_fullhash.json明确full_source_sha256=true，结束后重新核验843份PKL；95标签、核心代码、商飞输入/标签不变|
|冻结配置与原代码|314受保护文件及18旧CP产物哈希通过；六个冻结文件、13来源配置和成功args逐项核对；模型/特征/训练算法不改，只修复授权的无标签返回数量错误|
|训练协议|Adam/MSE/constant LR/FP32/双GPU ZeRO-2、32 workers/进程、max100/least5/patience5；各模型保持公开batch/累计/HP，完整参数和来源哈希交付|
|选择隔离|按原source validation MAPE选权重，每轮source test日志行为保留；无C1/C2微调、调参、选checkpoint|
|scaler|全部36份成功权重对应283训练电芯，均值566.2897526501766、标准差479.30134018240085、n_samples_seen=283；无验证/测试/目标拟合|
|保存与重载|33份新权重保存诊断state/prediction一致；CP3份按同意的历史导入核验，保存前内存probe不可获得，不假造；72条目标均有只读重放证据|
|两套源指标|成功种子96×100=9600窗口及96个第100圈窗口；五项与日志固定容差一致；全部36份数组远端和本机math.fsum独立重算|
|异常敏感性|G43C3仍在正式96电芯结果；补充95枚/9500窗口同五项敏感性，CP补充acc10=34.5158%，不补零或改baseline|
|商飞原始数据和标签|独立Excel及测试复现379/419；0.94只用于日常代理标签，完整RPT单列；本机两个real Excel测试实际通过|
|商飞输入|原曲线算法、I/Q÷183、三通道300点、仅物理1–100、RPT1零值mask0；独立XML180000特征逐位相同，结束后NPZ哈希不变|
|RNN缺口|四CP-RNN物理2–100前99格加尾零，保留第100圈；source前缀不变；22份新模型映射校验，旧CP共享canonical哈希|
|阶段评价及授权|用户提前评价11模型，66行只导入复用；不冒充原39源终态门槛。Autoformer3源通过后才追加6行，原66逐值不变|
|预测和失败|72唯一组合/12完整目标组；PatchTST失败无聚合/目标分数；18个≤100及4个负寿命保留计入；只2枚独立目标电芯|
|独立算术/CSV|两个CSV与JSON一致；math.fsum重算36源数组、72目标、三种子均值/总体std，最大差5.6843e−13<1e−9；总寿命与100圈后RUL分开保存|
|独立NN重放|已有66行33权重CPU FP32重放最大差0.0005714<0.005圈；Autoformer3权重6行CUDA FP32重放差0，无state更新或正式结果替换|
|最终回归|本机81全通过123.194秒；远端79通过/2 Excel缺失跳过21.574秒，由本机实际覆盖；训练结束后执行，不把skip称通过|
|完整交付|13行对比、72行预测、源/目标逐种子JSON、敏感性、RUL补充、图、日志/配置/权重/scaler索引及命令；36成功checkpoint的177文件本机逐大小/SHA匹配，CP旧归档复用|
|结论边界|95日常代理寿命非RPT实测；非严格论文v4/80%复现；有效源8个；目标此前已观察非盲测，不反调；物理一致性/泛化不能保证|

## 关键证据

- retry_completion_integrity.json：原66行和原v4 manifest不变、剩余任务终态。
- end_of_run_data_preflight_fullhash.json：结束后的843源SHA检查。
- protected_artifact_hashes.json：314受保护文件、18旧CP产物、实际进程缺席。
- independent_final_numeric_audit.json：36源和72目标、两份CSV的远端核验。
- local_independent_math_audit.json：另一套math.fsum、RUL补充、所有异常输出。
- autoformer_diagnostic_replay_6.json：三个Autoformer权重只读重放。
- checkpoint_scaler_configuration_index.json及local_checkpoint_integrity.json：完整参数、来源、177文件本机实际位置。
- uniform_sensitivity_summary.json：包括CP补充acc10的同五项指标。
- final_validation_tests_local.log及final_validation_tests_remote.log：81项结束后回归与真实skip。

## 保留的历史中断和资源记录

v1/v2试跑基础设施中断、v3到v4的推理后端审核恢复完整留档，不混作
模型科学失败。Autoformer2021自然结束后保留，队列移交suite锁错误
造成约81分钟调度间隔，随后合法恢复。11模型阶段后长时SIGSTOP
触发NCCL超时；用户只批准重训2024，从头10轮自然早停，旧失败日志
和中间权重未删。见[恢复记录](BatteryLife_Autoformer_暂停恢复与单种子重训_20261013.md)。

历史资源快照：容器memory.max=160GiB，而非宿主机free显示754GiB；
曾观察memory.current≈59.43GiB、max/oom/oom_kill均0、数据盘约24.8GiB
空闲。没有自动租新机、扩盘或删除旧实验。最终计时/显存使用实测记录，
CP历史数据未知，不补造。服务器是否关机由用户决定。
