# BF16 empirical information analysis

`scripts/analyze_bf16_information.py` 对 safetensors 中的 BF16 物理 words
进行只读、分块、内存受控的统计。它直接 mmap shard，不加载 Transformers
模型，不使用 GPU，也不修改 checkpoint。

当前吞吐实验运行期间不要扫描真实 checkpoint。待吞吐 baseline 完成后，
Llama-3.1-8B 的建议命令为：

```bash
python3 scripts/analyze_bf16_information.py \
  /workspace/checkpoints/llama-3.1-8b-bf16 \
  --output-dir results/information/llama-3.1-8b-bf16 \
  --jobs 4 \
  --chunk-mib 64 \
  --adjacent-exponent-order1 \
  --method-ratio Brevis=1.518121 \
  --method-ratio ZipNN=1.506628 \
  --method-size DFloat11=10895502042 \
  --progress
```

Ratio 必须与被扫描 checkpoint 使用相同的 source-file byte 口径。
`--method-size METHOD=BYTES` 接受完整压缩输出的整数 byte size。对目录格式，
先用论文 harness 的同一 `tree_size` 口径求和，再传入此参数。
正式论文表优先使用 `--method-size`：ratio 输入只能反推出可能为小数的
implied output bytes，不能冒充精确测得的输出大小。

默认是单 job、每 job 64 MiB BF16 输入 chunk。并行时内存和存储带宽大致随
`--jobs` 增长；`chunk-mib` 只限制每个 worker 的输入 words，并不是 RSS
硬上限。NumPy histogram、exponent 和 transition 临时数组会让峰值内存达到
`chunk-mib × jobs` 的数倍。需要最小干扰时使用
`--jobs 1 --chunk-mib 32`。

## 输入发现

位置参数支持：

- checkpoint 目录；
- `download-manifest.json`；
- `model.safetensors.index.json`；
- 单个 `.safetensors`。

目录中存在 `download-manifest.json` 时，只扫描 manifest 声明的 canonical
safetensors weights；否则使用根 model index、根单文件，最后才递归回退。
报告记录发现模式、shard 路径、size 和 mtime。非 BF16 tensors 不进入 entropy
统计，但其数量和 payload bytes 会明确记录。

index 模式会验证 `weight_map` 中每个 tensor 的 shard 归属，并拒绝 shard
里的漏项或额外 tensor。manifest 中的 SHA256 状态只作为历史声明记录；本工具
本次不重新计算 hash，也不会把它表示成本次校验结果。扫描前后的 shard
size、mtime 和 inode/device fingerprint 必须一致，否则结果作废。

## 输出量

`groups.csv` 和 JSON 按 overall 及固定 role taxonomy
（embedding、attention、MLP、norm、other）报告：

- BF16 tensor 数和参数数；
- nominal `16 bits/weight`；
- 完整 BF16 16-bit word 的 empirical zero-order entropy
  `H0(W)`；
- 8-bit exponent 的 empirical zero-order entropy `H0(E)`；
- idealized exponent-only reference `8 + H0(E)`；
- 可选的 tensor 内相邻 exponent 条件熵 `H(E_i | E_{i-1})`；
- 对有限 tensor 序列正确计入首 exponent 的 adjacent reference：

  ```text
  8 + [K × H0(E_first) + P × H(E_i | E_{i-1})] / N
  ```

  其中 `K` 是非空 tensor 数，`P = N - K` 是 tensor 内 transition 数，
  `N` 是 BF16 参数数。

相邻统计使用每个 tensor 的 row-major 物理顺序，正确连接 chunk 边界，但绝不
在两个 tensor 之间人为创建 transition。为保持内存有界，只实现 256×256 的
exponent transition table；没有实现潜在 65,536×65,536 的完整 BF16-word
transition matrix。

输出文件：

- `information-analysis.json`：完整配置、inventory、分组结果、逐 tensor
  结果、exponent histograms 和 compressor comparisons；
- `groups.csv`：overall/role 表；
- `tensors.csv`：逐 tensor 审计表；
- `methods.csv`：外部 ratio/size 的 amortized whole-output BPW 与 signed
  differences；没有传入 method 时仍写出只有 header 的 CSV；
- `report.md`：可直接检查的论文式表格和解释限制。

## 必须保留的解释

Empirical `H0(W)` 是“把该 checkpoint 的完整 16-bit words 当成 iid symbols”
时的描述性参考。它不是允许利用顺序、重复、tensor role、程序结构或 side
information 的任意 compressor 的绝对 lower bound。

同样，`8 + H0(E)` 假设：

- sign 与 mantissa 固定付出 8 raw bits/weight；
- exponent 使用理想 iid entropy code；
- codebook、framing、padding、alignment 和 decoder 成本全为零。

因此它也不是绝对 lower bound。结构化 lossless compressor 完全可能低于某个
empirical iid reference。`methods.csv` 使用
`signed_amortized_bpw_difference_from_...` 字段，负值合法；论文中不得把它
改写成“不可能”或“剩余可压缩空间”的证明。

Amortized whole-output BPW 的定义为：

```text
8 × complete compressed bytes / analyzed BF16 parameter count
```

完整压缩大小包含 method framing 和所有非 BF16 内容；当输入含非 BF16 tensors
时，这一点会让该值成为“每个 BF16 参数分摊的 whole-checkpoint 成本”，
不能称作 BF16 payload 本身的精确 BPW。报告会显式显示 skipped non-BF16
coverage。

## 合成测试

```bash
python3 -m unittest scripts/test_analyze_bf16_information.py -v
```

测试仅创建很小的临时 safetensors，覆盖 entropy golden、chunk-boundary
transition、tensor-boundary exclusion、manifest 选择、role 分组、ratio/size
BPW、index 映射校验以及 JSON/CSV/Markdown 输出；不会扫描真实 checkpoint。
