# Brevis 单晚实验

`scripts/run_benchmarks.py` 按以下顺序执行当前实验计划：

1. 本地 Qwen2.5-7B 上六个通用方法的单 worker end-to-end；
2. Brevis search-budget sweep、worker sweep 和 PHOG/A* ablation；
3. 十个正式 checkpoint 的 archive-size matrix；
4. DFloat11/ECF8 native conversion（配置后）；
5. 从 append-only JSONL 生成论文表格和 Figure 1/2 数据。

每个 timing cell 只运行一次。成功记录会自动续跑跳过；失败记录保留，下一次执行会重试。
所有 archive 使用 `.brv` 后缀。恢复文件和临时 sweep archive 在精确校验后删除。
压缩和 native conversion 默认每 5 秒输出 elapsed time 和 RSS；普通 file codec
还会显示当前 archive 大小。`output/input` 是当前输出体积比，不是虚构的完成百分比。
DFloat11/ECF8 输出目录不会被周期性遍历，以免干扰计时。可用
`--progress-interval 10` 调整频率，或设为 `0` 关闭；该显示选项不影响断点续跑身份。
进度 heartbeat 在所有并发任务之间全局节流；每次只增加一次文件 `stat` 和终端输出。
它的开销很小但不是数学上的零，投稿用的最终 timing 建议设为 `0`。

## 依赖

```bash
zig build test
zig build -Doptimize=ReleaseFast

# 按发行版安装 zstd、lz4 和 libdeflate-gzip
python3 -m pip install zipnn==0.5.4 python-snappy==0.7.3 safetensors torch
```

`preflight` 会记录 Brevis revision、执行脚本摘要、主机/CPU/RAM、方法版本和
cache-control 状态；这些 provenance 也会进入 run ID，换版本或换机器后不会误复用旧结果：

```bash
python3 scripts/run_benchmarks.py preflight \
  --methods brevis zstd-9 zipnn lz4-hc-9 libdeflate-6 snappy
```

## 先检查执行计划

```bash
python3 scripts/run_benchmarks.py all \
  --models-root /data/brevis-checkpoints \
  --core-model /data/Qwen2.5-7B \
  --results /data/brevis-results \
  --methods brevis zstd-9 zipnn lz4-hc-9 libdeflate-6 snappy dfloat11 ecf8 \
  --specialized-config /data/specialized-baselines.json \
  --workers 32 \
  --shard-jobs 32 \
  --deadline-hours 12 \
  --dry-run
```

`--core-model` 可以指向单个 safetensors 或完整 checkpoint 目录。
`--models-root` 是下载脚本的输出目录；harness 只读取各模型
`download-manifest.json` 引用的 canonical shard，不会纳入重复的 consolidated 权重。
下载脚本也会保存 model config 和 tokenizer 资产，供 DFloat11 和 ECF8 的官方
model/tokenizer loader 使用，但 archive size 仍只统计 manifest 中的权重。
默认还会核对十个模型的 repo ID 和固定 commit，并要求十个全部存在。临时开发 fixture
只能显式加 `--allow-custom-corpus`，不能误产出正式表。

## 正式运行

Linux root 可直接 drop page cache。非 root 机器需预先允许一个无交互 cache-drop 命令：

```bash
python3 scripts/run_benchmarks.py all \
  --models-root /data/brevis-checkpoints \
  --core-model /data/Qwen2.5-7B \
  --results /data/brevis-results \
  --methods brevis zstd-9 zipnn lz4-hc-9 libdeflate-6 snappy dfloat11 ecf8 \
  --specialized-config /data/specialized-baselines.json \
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
仓库内的 `scripts/specialized_baselines.py` 把两个官方入口统一成显式
source/output。正式环境固定 DFloat11 `457733886ce6ebc6d8dda1621fad1ffa2661e028`
和 ECF8 `9cbf3d5cf77d6db8cf6f29df1fe6d52bc88fa01e`，按各自 README 安装 CUDA
依赖后创建配置：

```json
{
  "dfloat11": {
    "checkpoint_pattern": "^(llama-3\\.1-(8b|70b)-bf16|qwen3-32b-bf16)$",
    "workers": 32,
    "cwd": "/path/to/Brevis",
    "version_command": ["git", "-C", "/opt/DFloat11", "rev-parse", "HEAD"],
    "compress_command": [
      "python3", "scripts/specialized_baselines.py", "dfloat11",
      "--source", "{source_dir}",
      "--output", "{archive}",
      "--workers", "{workers}",
      "--validate-cuda"
    ],
    "validates_during_compression": true
  },
  "ecf8": {
    "checkpoint_pattern": "^qwen3-32b-fp8$",
    "workers": 32,
    "cwd": "/path/to/Brevis",
    "version_command": ["git", "-C", "/opt/ecf8", "rev-parse", "HEAD"],
    "compress_command": [
      "python3", "scripts/specialized_baselines.py", "ecf8",
      "--source", "{source_dir}",
      "--output", "{archive}",
      "--workers", "{workers}",
      "--upstream", "/opt/ecf8",
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
DFloat11 wrapper 只接受官方已支持且共享标准 decoder-block layout 的 Llama/Qwen3
BF16 三个 checkpoint；其他 cell 写 `unsupported checkpoint`。ECF8 只运行
Qwen3-32B-FP8。其官方入口分别仍是
`compress_model(..., check_correctness=True)` 和
`scripts/compress.py --save_model --n_processes N --validate_cuda`；wrapper 不重实现
codec，并会从 ECF8 输出中移除重复的 converter cache 后再统计 archive size。

`libdeflate-6` 使用 `libdeflate-gzip` 的官方默认 level 6 和 gzip stream。它是
whole-buffer、单进程 codec；harness 通过 shard 并发利用多核，运行时要给每个并发
shard 留出输入和输出 buffer 的内存。

未配置 specialized converters 时，可加 `--allow-missing` 先完成六个通用方法；
Table 2 对应 cell 会写 `missing dependency/config`，不会伪装成成功。

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

所有表和图只消费有成功 exactness record 的 timing。Table 2 固定展开
`checkpoint × requested method`，只有全部 canonical shards 都通过才写 `ok` 和
archive size；依赖、配置、支持范围、deadline 或失败造成的空 cell 都保留明确状态。
`--rerun` 仍追加 raw record，但汇总只使用同一 run ID 的最后一条记录。

如果 specialized wrapper 在转换命令内部做 correctness check，其 raw wall time
也会包含该验证，不与六个通用方法的 compression wall time直接比较；它在本计划中
只进入 archive-size matrix。

Table 5 的 information-theoretic headroom 和 operator attribution 需要读取
BRTA/BRPG 的独立 archive analyzer。当前 CLI 只输出 aggregate search counters，脚本
不会用不充分的数据伪造这两项分析。
