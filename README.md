# Brevis

Brevis 对神经网络张量做逐比特精确的无损压缩。核心不是手写一个固定编解码器，而是在有限、类型化、构造上可逆的 DSL 中，为每个张量合成一个短程序。

压缩产物是“程序 + 叶子数据 + 必要元数据”。解压不做搜索，只执行已保存的程序，精确重建原始 safetensors 文件。

项目使用 Zig 0.16。

## 快速开始

```bash
zig build -Doptimize=ReleaseFast

# 可选：从当前模型校准 PHOG 先验
./zig-out/bin/brevis calibrate model.safetensors prior.bin

# 不传 --prior 即为 uniform 搜索
./zig-out/bin/brevis compress model.safetensors model.brv --prior prior.bin
./zig-out/bin/brevis decompress model.brv restored.safetensors
./zig-out/bin/brevis verify model.brv model.safetensors

# 控制线程数或校准张量数
./zig-out/bin/brevis compress model.safetensors model.brv --jobs 12
./zig-out/bin/brevis calibrate model.safetensors prior.bin --tensors 200
```

```bash
zig build test -Doptimize=ReleaseFast
python3 -m unittest discover -s eval -p 'test_*.py'
```

## 整体流程

```text
safetensors
  → mmap 读取并按张量规划 block
  → 在张量样本上搜索可逆程序模板
  → 为每个 block 拟合少量参数并执行程序
  → 终端用 raw / bitpack / Huffman / rANS 编码
  → 流式写入 .brv

.brv
  → 读取程序、side information 和 payload
  → 执行终端解码
  → 逆序执行可逆变换
  → 恢复原始 safetensors header 与张量字节
```

输入和 archive 均使用 mmap。压缩结果分批生成并顺序写盘，不在内存中保存整模型 archive，因此可以处理多分片大模型。

## 可逆 DSL

搜索状态是一棵带 hole 的类型化程序树。每个非终端算子都定义了 `forward` 和对应逆变换；终端算子把最后的整数流编码成字节。

变换包括常量异或、模加、相邻异或、差分、Gray、位旋转、字段拆分、浮点字段拆分、RLE、codebook、bit-plane 和 byte-plane。终端为 `raw`、`bitpack`、canonical Huffman 与 rANS。

算子的合法性由输入位宽、dtype、树深和 arity 决定。`raw` 始终提供可逆回退，因此任意受支持张量都有合法程序。

## PHOG 引导的 A*

搜索同时维护两种彼此独立的成本：

- `p` 是 PHOG 给出的语法描述长度，只决定 A* 先展开哪个 partial program。
- `g_bytes + lowerBound` 是序列化字节下界，用于安全剪枝；完整程序最终按真实 archive 字节数比较。

PHOG 不改变压缩目标，只在有限搜索预算内改变候选到达顺序。没有 prior 时使用 uniform 分布，便于直接测量先验的贡献。

在变换仍可使用的浅层 hole 上，经验熵不是合法下界，因为可逆变换可能大幅降低它。此时只计算必需的 terminal frame 成本；达到最大变换深度后，才加入 Shannon payload 下界。

### 模型内校准

校准不依赖外部训练语料。它按 dtype 和张量规模分层抽样，在每个张量的四个连续窗口上搜索候选，再把前八个候选放到四个代表 block 上按真实编码字节复排。

获胜程序被转换成 `(context, production)` 计数。Context 包含树位置、dtype、位宽以及分桶后的熵、零值比例和差分特征，并使用三级 backoff。最终 prior 与 uniform 混合，避免过度相信稀疏统计。

## Archive 与解码

当前 writer 生成 schema 6 archive。每个 block frame 保存程序 bytecode、side information 和 payload；footer 保存张量名称、dtype、shape、block 数量以及原始 safetensors header。

新 archive 不生成跨 block back-reference，使每个 frame 保持独立且易于并行。Reader 仍能读取旧 schema 5/6 back-reference，并拒绝前向引用、自引用和损坏长度。

单线程解码直接在调用线程执行。多线程模式复用固定 worker pool，并让当前 batch 写盘时并行解码下一批；输出顺序和逐张量长度会在写入前后验证。

## 完整 8B 实测

Qwen3-8B-Base 五个 BF16 分片共 16,381,516,776 B，在 12 核 M4 Pro 上的完整结果如下：

| 模式 | Archive | 压缩 | 解码 |
| --- | ---: | ---: | ---: |
| PHOG，首次使用 | 10,922,326,421 B | 8.19s 校准 + 11.04s 压缩 | 6.49s（12 线程） |
| PHOG，复用 prior | 10,922,326,421 B | 11.04s | 40.41s（单线程） |
| Uniform | 10,922,803,714 B | 24.60s | 已逐比特验证 |

PHOG 在该模型上比 uniform 少 477,293 B，收益很小但稳定；它的主要作用是更快到达同一批高质量程序，而不是替代真实字节目标。

完整逐分片结果及 gzip、zstd、xz、OpenZL baseline 位于 [`eval/results-large.json`](eval/results-large.json)。评测入口是 [`eval/run_eval.py`](eval/run_eval.py)。

## 代码结构

```text
src/types.zig        dtype、Stream、TensorView、block 规划
src/ops.zig          可逆 DSL 算子
src/program.zig      程序执行、逆执行与序列化
src/search.zig       PHOG 引导的 A* 与字节下界
src/prior.zig        Context、三级 backoff prior
src/calibrate.zig    模型内抽样、候选复排和先验训练
src/codec.zig        bitpack、Huffman、rANS
src/archive.zig      流式 .brv 格式与兼容 reader
src/safetensors.zig  safetensors mmap 读写
src/main.zig         CLI、批处理与并行流水线
eval/                多分片端到端评测
```

## 正确性

测试覆盖所有算子的随机往返、程序和 archive 往返、uniform 等价性、搜索剪枝、FP8/u32、损坏 Huffman/rANS、旧 archive 引用以及单线程/并行解码。

大型评测会对 uniform、PHOG、`--jobs 1`、默认并行解码和所有 baseline 分别做完整字节比较。

## License

见 [LICENSE](LICENSE)。
