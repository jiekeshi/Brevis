# BRTA tensor/operator attribution

`scripts/analyze_brevis_attribution.py` 对已经生成的 Brevis `BRTA v3`
归档做只读分析。它不会运行压缩、解压或 GPU kernel，也不会物化 literal：
大 literal body 通过 seek 跳过。因此分析时间主要取决于 tensor/program 数量和
文件系统 metadata，而不是模型参数量。

## Llama-3.1-8B 调用

正式结果应同时给出 Brevis shard 归档和原始 checkpoint：

```bash
python3 scripts/analyze_brevis_attribution.py \
  --archives \
    results/paper-dsl-refactor-defaults-hot/archives/llama-3.1-8b-bf16/brevis \
  --sources /workspace/checkpoints/llama-3.1-8b-bf16 \
  --output-dir \
    results/paper-dsl-refactor-defaults-hot/analysis/llama-3.1-8b-bf16-attribution
```

`--archives` 和 `--sources` 都可接受一个或多个文件/目录。目录递归查找文件；
source 目录存在 `download-manifest.json` 时只读取 manifest 声明的 canonical
weight shards，避免把重复的 consolidated 权重纳入分析。每个归档通过其内嵌的
完整 safetensors prefix 与 source 精确匹配，不依赖文件名猜测。

没有 source 时也可运行：

```bash
python3 scripts/analyze_brevis_attribution.py \
  --archives path/to/brevis-archives \
  --output-dir path/to/attribution
```

此时 tensor 大小来自 BRTA 内嵌的原始 safetensors metadata，仍是精确值，但
报告会明确标记没有做外部 source-prefix matching。

## 输出

- `report.md`：可直接检查的总体、role、dtype、root program、operator 和
  literal codec 表。
- `attribution.json`：包含校验范围和所有结构化结果。
- `archives.csv`：逐 shard 的 source/archive 总量、全文件压缩倍数和 embedded
  prefix SHA-256。
- `tensors.csv`：逐 tensor 的源 payload、完整 record、program/framing 字节，
  role、dtype、root program、stored XXH3-64 little-endian wire bytes、节点和
  literal codec 统计。
- `by-role.csv`：`embedding / attention / mlp / norm / other`。
- `by-dtype.csv`：按 safetensors dtype。
- `by-selection.csv`：根 `Lit` fallback 与所有 synthesized program。
- `by-program.csv`：根 `Lit` fallback 与 `synthesized:<root-operator>`。
- `by-role-program.csv`：role 与 root program 的交叉表。
- `operators.csv`：每种 operator 的实例数、出现 tensor 数和 exclusive wire
  bytes。
- `literal-codecs.csv`：raw/bitpack/Huffman/rANS 的 body、payload 和语义叶子
  storage。

Role taxonomy 是固定、可审计的名字规则，匹配优先级为 embedding、norm、
attention、MLP、other。norm 优先于位置描述词，因此
`post_attention_layernorm` 和 `self_attn_layer_norm` 归入 norm，而不是
attention。Llama 的 `lm_head.weight` 归入 `other`。论文中应保留这一规则，
不根据结果重新分类。

## 精确量与不可识别量

以下量可从格式精确恢复：

- 每个 tensor 的 dtype、shape 和 source payload bytes；
- 每个 tensor 的完整 BRTA record bytes；
- program bytes 与 record framing/checksum bytes；
- 根节点是 universal `Lit` fallback 还是 synthesized program；
- 每个 BRPG operator 的计数和它在 winning program 中实际占用的 exclusive
  wire bytes；
- 每个 literal 选择的物理 codec 和 body/payload bytes；
- 全局 source/archive 大小及 `source / archive` 倍数。

Role/dtype/program 表使用 `source tensor payload / complete tensor record`。
它们不分摊全局 safetensors/BRTA header；`archives.csv` 和 `report.md` 的
whole-archive 总计包含全部 header，因而精确闭合。

以下量不能从归档诚实恢复：

- “某个 operator 单独节省了多少 bytes”。BRTA 只保存 winning program，
  不保存删除该 operator 后仍然正确的反事实 program 或搜索轨迹。
- search 中失败/落选 program 的成本。
- 不执行 program 时的 XXH3 校验和 source tensor 内容一致性。

因此 `operators.csv` 的 `exclusive_program_bytes` 只能描述 winning program
的存储构成，不能写成 operator 带来的因果节省。若论文需要 operator
causal ablation，必须重新压缩并显式禁用对应 grammar production。

发布前仍需对每个 shard 独立运行：

```bash
zig-out/bin/brevis verify archive.brv source.safetensors
```

analyzer 会做 BRTA/BRPG framing、embedded metadata cross-binding、静态 stream
bits/length 和 literal body 的 tag/declared-length framing 检查；它不会套用
decoder 的 output/literal/execution-work resource limits，也不会完整验证
bitpack/Huffman padding、Huffman/rANS table 或 entropy payload。上述 `verify`
才会执行完整 decoder、校验 XXH3 并比较原始 tensor。

## 测试

```bash
python3 -m unittest scripts/test_analyze_brevis_attribution.py -v
```

测试使用小型合成 BRTA/BRPG fixture，不运行模型压缩。
