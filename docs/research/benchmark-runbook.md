# Brevis 单晚实验

`scripts/run_benchmarks.py` 按以下顺序执行当前实验计划：

1. 本地 Qwen2.5-7B 上五个通用方法的单 worker end-to-end；
2. Brevis search-budget sweep、worker sweep 和 PHOG/A* ablation；
3. 十个正式 checkpoint 的 archive-size matrix；
4. DFloat11/ECF8 native conversion（配置后）；
5. 从 append-only JSONL 生成论文表格和 Figure 1/2 数据。

每个 timing cell 只运行一次。成功记录会自动续跑跳过；失败记录保留，下一次执行会重试。
所有 archive 使用 `.brv` 后缀。恢复文件和临时 sweep archive 在精确校验后删除。

## 依赖

```bash
zig build test
zig build -Doptimize=ReleaseFast

# 按发行版安装 zstd 和 lz4
python3 -m pip install zipnn==0.5.4 python-snappy==0.7.3 safetensors torch
```

`preflight` 会记录 Brevis revision、CPU、RAM、方法版本和 cache-control 状态：

```bash
python3 scripts/run_benchmarks.py preflight \
  --methods brevis zstd-9 zipnn lz4-hc-9 snappy
```

## 先检查执行计划

```bash
python3 scripts/run_benchmarks.py all \
  --models-root /data/brevis-checkpoints \
  --core-model /data/Qwen2.5-7B \
  --results /data/brevis-results \
  --methods brevis zstd-9 zipnn lz4-hc-9 snappy \
  --workers 32 \
  --shard-jobs 32 \
  --deadline-hours 12 \
  --dry-run
```

`--core-model` 可以指向单个 safetensors 或完整 checkpoint 目录。
`--models-root` 是下载脚本的输出目录；harness 只读取各模型
`download-manifest.json` 引用的 canonical shard，不会纳入重复的 consolidated 权重。

## 正式运行

Linux root 可直接 drop page cache。非 root 机器需预先允许一个无交互 cache-drop 命令：

```bash
python3 scripts/run_benchmarks.py all \
  --models-root /data/brevis-checkpoints \
  --core-model /data/Qwen2.5-7B \
  --results /data/brevis-results \
  --methods brevis zstd-9 zipnn lz4-hc-9 snappy \
  --workers 32 \
  --shard-jobs 32 \
  --deadline-hours 12 \
  --drop-caches-command "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
```

没有 cache-drop 权限时，脚本仍运行 hot 和 unconditioned size measurement，但 cold cell
明确写成 `not measured`，不会伪装成 cold cache。

也可以分阶段执行：

```bash
python3 scripts/run_benchmarks.py core     --core-model /data/Qwen2.5-7B ...
python3 scripts/run_benchmarks.py sweeps   --core-model /data/Qwen2.5-7B ...
python3 scripts/run_benchmarks.py ablation --core-model /data/Qwen2.5-7B ...
python3 scripts/run_benchmarks.py corpus   --models-root /data/brevis-checkpoints ...
python3 scripts/run_benchmarks.py summarize --results /data/brevis-results
```

默认 Brevis 配置为 `--max-expansions 512 --tensors 256`。Pareto budgets 为
`0,1,8,32,128,512`，worker sweep 为 `1,2,4,8,16,32`。完整配置的 worker-1 run
直接复用 core hot measurement，不重复运行。

## DFloat11 和 ECF8

两者是 checkpoint-level native converters，不是具有相同 CLI 的逐文件 codec。
DFloat11 还要求为模型结构提供 block pattern；ECF8 只应用于 FP8 checkpoint。
因此 harness 不猜测模型结构，通过 `--specialized-config` 调用你按官方仓库固定
revision 的薄 wrapper。wrapper 必须接收显式 source/output，并做 CUDA bit-exact
validation：

```json
{
  "dfloat11": {
    "checkpoint_pattern": "bf16$",
    "workers": 32,
    "cwd": "/opt/dfloat11-benchmark",
    "version_command": ["git", "rev-parse", "HEAD"],
    "compress_command": [
      "python3", "run_dfloat11.py",
      "--source", "{source_dir}",
      "--output", "{archive}",
      "--workers", "{workers}",
      "--check-correctness"
    ],
    "validates_during_compression": true
  },
  "ecf8": {
    "checkpoint_pattern": "^qwen3-32b-fp8$",
    "workers": 32,
    "cwd": "/opt/ecf8-benchmark",
    "version_command": ["git", "rev-parse", "HEAD"],
    "compress_command": [
      "python3", "run_ecf8.py",
      "--source", "{source_dir}",
      "--output", "{archive}",
      "--workers", "{workers}",
      "--validate-cuda"
    ],
    "validates_during_compression": true
  }
}
```

可用占位符为 `{source_dir}`、`{archive}`、`{workers}`、`{checkpoint}`、
`{repo_id}` 和 `{revision}`。若 converter 自行决定输出目录，增加
`"output_path": "/actual/path/{checkpoint}.brv"`。
若验证可与转换分开，使用 `"validate_command": [...]` 代替
`validates_during_compression`，这样验证时间不会计入 compression wall time。
官方 DFloat11 示例入口是 `compress_model(..., check_correctness=True)`；官方 ECF8
入口是 `scripts/compress.py --save_model --n_processes N --validate_cuda`。wrapper
只负责把这两个入口的模型专用路径和输出目录统一为上述接口，不应重实现 codec。

未配置 specialized converters 时，可加 `--allow-missing` 先完成五个通用方法；结果会
明确缺失，不会写成 `N/A`。

## 输出

```text
results/
  environment.json
  raw/runs.jsonl
  logs/
  archives/<checkpoint>/<method>/*.brv
  tables/table2-compression-effectiveness.csv
  tables/table3-end-to-end.csv
  tables/figure1-archive-size-vs-time.csv
  tables/figure2-throughput-rss-vs-workers.csv
  tables/table4-ablation.csv
  status.md
```

`runs.jsonl` 是唯一原始事实来源。wall time 包含进程启动、读写和输出 `fsync`；
exactness verification 在计时区间之外。Peak RSS 是周期性汇总主进程及其子进程 RSS
的单次峰值。跨 shard 的 wall time 和 bytes 求和，peak RSS 取最大值，不把 shard
当成独立模型平均。

如果 specialized wrapper 在转换命令内部做 correctness check，其 raw wall time
也会包含该验证，不与五个通用方法的 compression wall time直接比较；它在本计划中
只进入 archive-size matrix。

Table 5 的 information-theoretic headroom 和 operator attribution 需要读取
BRTA/BRPG 的独立 archive analyzer。当前 CLI 只输出 aggregate search counters，脚本
不会用不充分的数据伪造这两项分析。
