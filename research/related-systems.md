# Related systems: tensor compression as program synthesis

研究范围：用户所附 ICLR 2025 论文 *The KoLMogorov Test: Compression by Code Generation*，以及 OpenZL、weight-compression、Euphony、DFloat11 的论文、官方文档和源码。一手源码版本固定为 `facebookresearch/kolmogorov@690236b`、`facebook/openzl@79fa667`、`brianbell-x/weight-compression@b9510aa`、`wslee/euphony@0b9a62a`、`LeanModels/DFloat11@4577338`。

## 结论

Brevis 最清楚的论文主线不是“自动挑一个 codec”，而是：

> 对每个张量，在有限、类型化、构造上可逆的 DSL 中合成一个短程序；归档保存程序、参数和残余 payload，解压只执行程序的逆语义，逐比特重建张量。

它与几项相关工作的边界很清楚：KoLMogorov-Test 提供“程序即压缩”的问题表述；Euphony 提供 PHOG 引导的 A*；OpenZL 证明“自描述变换图 + 通用解码器”可工程化；DFloat11 和 weight-compression 提供张量位域与高速解码的领域原语。Brevis 的新组合是 **constructive reversibility + tensor-typed DSL + PHOG-A* + bit-exact high-throughput executor**。

## 系统对照

| 系统 | 可搜索/可配置表示 | 解码 | 最值得借鉴 | 不应照搬 |
|---|---|---|---|---|
| KoLMogorov-Test | Python 程序或生成序列的 DSL | 执行生成程序 | 两部码目标、程序失败时回退原文、合成数据训练先验 | 任意 Python、非可逆生成算子、LLM 参与解码 |
| Euphony | SyGuS 类型化语法树 | 求值/验证合成程序 | PHOG 的上下文概率、`g+h` A*、最小完成代价 | Python/SMT 框架、基于样例签名的有损等价剪枝 |
| OpenZL | 类型化 codec DAG | 通用解码器执行逆 DAG | 标准 op ID、自描述程序、类型校验、decoder fusion | 完整 graph/registry/parser 体系和当前内存模型 |
| DFloat11 | 固定 BF16 位域分解 + Huffman | GPU 在线重建 BF16 | exponent 分离、块级批量解码、并行 seek 元数据、LUT | 现在就引入 CUDA 专用两阶段内核 |
| weight-compression | 每张量 K=15 固定宽 codebook + escape | codebook/escape 精确逆变换 | SIMD 友好的 4-bit 热符号原语、逐 shard 流式实模验证 | 把 bit accounting 当成已序列化结果或把原型计时当端到端结果 |

## 可直接落地的机制

### 1. 把 PHOG 用作 A* 导航，不混淆“概率”与“真实归档长度”

Euphony 对一次产生式展开收取 `-log2 p(rule | context)`，并把已付代价与未完成非终结符的最小完成代价相加后入优先队列；其实现分别见[最小完成代价的固定点](https://github.com/wslee/euphony/blob/0b9a62a294897b707c86ba3c0f63561ff503cb3b/bin/phogs/phog.py#L76-L167)和[`priority = new_cost + future_cost`](https://github.com/wslee/euphony/blob/0b9a62a294897b707c86ba3c0f63561ff503cb3b/bin/phogs/phog.py#L344-L380)。PHOG 的上下文不是只看父节点，而是由 `UP/LEFT/PREV_DFS/WRITE_VALUE` 等小型树导航程序提取；[源码定义](https://github.com/wslee/euphony/blob/0b9a62a294897b707c86ba3c0f63561ff503cb3b/bin/phogs/phog_utils.py#L23-L31)与[上下文解释器](https://github.com/wslee/euphony/blob/0b9a62a294897b707c86ba3c0f63561ff503cb3b/bin/phogs/phog_utils.py#L127-L229)对应原始 [PHOG 论文](https://proceedings.mlr.press/v48/bielik16.html)和 [Euphony 论文](https://doi.org/10.1145/3192366.3192410)。

Brevis 应保留两个分离的量：

- `f_phog = g_phog + h_phog`：只决定先扩展谁；首版上下文保持为 `(dtype, parent_op, child_slot)` 即可。
- `lb_archive` 与完整候选的 `archive_bytes = program + side_info + payload`：只用可证明的字节下界剪枝，并以真实序列化长度更新 incumbent。

只有当 PHOG 概率本身定义了程序的前缀码时，`-log p` 才等于真实程序长度。否则把它直接称为“最短程序 A*”会破坏最优性表述。最简洁、也最诚实的论文说法是 **PHOG-guided A* ordering with an independent admissible archive-size bound**。解码器不需要 PHOG；先验只影响压缩时的搜索。

PHOG 应从真实张量上已发现的程序训练，并按模型家族做 held-out 验证。Euphony也是从已有 SyGuS 解学习先验，[训练接口与正则化提示](https://github.com/wslee/euphony/blob/0b9a62a294897b707c86ba3c0f63561ff503cb3b/README.md#L10-L38)。KoLMogorov-Test 明确发现合成分布上的提升难以迁移到真实数据，[论文](https://openreview.net/forum?id=C45YqeBDUM)因此支持把 `uniform / PCFG / PHOG` 作为真实模型上的消融，而不是先堆一个大训练管线。

### 2. DSL 保持小，但加入一个真正面向权重的分支原语

OpenZL 把 codec 作为类型化边上的节点，要求 `decode(encode(x)) = x`，并在归档中记录足以让通用解码器执行逆 DAG 的信息；[图与 codec 契约](https://github.com/facebook/openzl/blob/79fa6674b027e398544419cd51afe22b205e8699/doc/mkdocs/doc/getting-started/concepts.md#L3-L42)。其 PyTorch 示例对 BF16/F32 做位域 deconstruct，把一条输出原样保存、另一条送 Huffman，再按 dtype 路由；[具体图定义](https://github.com/facebook/openzl/blob/79fa6674b027e398544419cd51afe22b205e8699/custom_parsers/pytorch_model_parser.c#L289-L328)。DFloat11 同样把 BF16 exponent 与 sign+mantissa 分开，只熵编码 exponent；[编码源码](https://github.com/LeanModels/DFloat11/blob/457733886ce6ebc6d8dda1621fad1ffa2661e028/dfloat11/dfloat11_utils.py#L173-L187)和[论文](https://arxiv.org/abs/2504.11651v3)都直接支持这一点。

因此 DSL 最有价值的下一项不是增加很多通用算子，而是一个类型明确、逆构造明确的 `bitfield_split/join`，至少覆盖 BF16/F16/F32。它可以是 DSL 中唯一的多输出节点；其余仍保持短链或小树。终端只需 `raw`、`bitpack`、`huffman/rANS`，再加入一个 `topk4_escape` 候选即可。后者来自 weight-compression：每张量取 15 个常见 sign+exponent 符号，用 4-bit index 表示，罕见值进入精确 escape stream；[可独立逆变换的实现](https://github.com/brianbell-x/weight-compression/blob/b9510aaac657b04e11d2f0d0d51a9b94af159590/src/tools/reproduce.py#L39-L115)。

不要把 KoLMogorov-Test 的 `filter/modulo/subsequence` 等生成算子直接搬进可逆变换 DSL。其 DSL 是“从零生成序列”，[原语列表](https://github.com/facebookresearch/kolmogorov/blob/690236b2192e5bc8d591f0f475367c92434a60a1/src/synthetic_data_generation/generate_data.py#L17-L92)并不保证输入到输出可逆；Brevis 的所有节点则应由类型系统保证逆程序存在。这个约束使“任意合成结果都可安全解码”成为比 KT 更强、也更适合系统论文的性质。

### 3. 搜索、拟合与执行必须是三个窄接口

建议唯一主流程为：

1. `synthesize(sample, dtype, prior) -> Program`
2. `fit(program, tensor_or_block) -> Params`
3. `encode(program, params, bytes) -> Frame`
4. `decode(frame) -> bytes`

每张量只搜索一次，块只重拟合小参数并编码；最终完整块必须实际序列化，若 `program + side_info + payload >= raw frame` 就写 `raw`。KoLMogorov-Test 的均匀程序码显式计算函数和参数的 bit cost，[源码](https://github.com/facebookresearch/kolmogorov/blob/690236b2192e5bc8d591f0f475367c92434a60a1/src/utils.py#L14-L82)，并以选择位加程序/原文回退，[评测实现](https://github.com/facebookresearch/kolmogorov/blob/690236b2192e5bc8d591f0f475367c92434a60a1/src/synthetic_data_generation/evaluate.py#L228-L249)。这支持 Brevis 始终按真实 frame 成本决策，而不是估算 payload 后再忽略 program/table/header。

候选评估应先走采样、统计量和下界，不应把每个候选在完整张量上跑一遍。OpenZL 自己也把“试运行整张图得到压缩尺寸”标为 CPU/内存浪费，[API 注释](https://github.com/facebook/openzl/blob/79fa6674b027e398544419cd51afe22b205e8699/include/openzl/zl_graph_api.h#L218-L249)。

### 4. 高速解压先靠块独立和融合，再考虑 GPU

CPU 路径应让 interpreter 每个节点 dispatch 一次，而不是逐元素解释；核心循环再做 SIMD：byte shuffle、xor/delta、bit unpack、table decode。独立块天然提供 seek point，可直接多线程解压，不需要为每个线程维护可变长码状态。

DFloat11 为 GPU Huffman 解码保存 thread gap 与 block output position，并用两阶段 decode + prefix sum 得到写位置；[编码侧元数据](https://github.com/LeanModels/DFloat11/blob/457733886ce6ebc6d8dda1621fad1ffa2661e028/dfloat11/dfloat11_utils.py#L118-L170)、[CUDA 两阶段主体](https://github.com/LeanModels/DFloat11/blob/457733886ce6ebc6d8dda1621fad1ffa2661e028/dfloat11/decode.cu#L25-L176)。它还把同一 transformer block 的多个矩阵一起解压以提高占用，[论文 §2.3.3](https://arxiv.org/abs/2504.11651v3)。这些是后续 GPU executor 的设计素材，不是当前 DSL/search 的依赖。

OpenZL 的 decoder 会由 frame 中的 transform ID 重建依赖、验证输入类型，并可识别 decoder fusion；[重建与 fusion](https://github.com/facebook/openzl/blob/79fa6674b027e398544419cd51afe22b205e8699/src/openzl/decompress/decompress2.c#L790-L857)。Brevis 可以晚些只为高频短程序增加少量预定义 fused kernels，不需要引入完整动态图注册系统。

### 5. 大模型评测应把“证据类别”分开

weight-compression 的优点不是它的固定 codec 本身，而是按 safetensors shard 逐个 mmap、逐张量验证、处理后删除 shard，使峰值磁盘约为一个 shard；[流式验证实现](https://github.com/brianbell-x/weight-compression/blob/b9510aaac657b04e11d2f0d0d51a9b94af159590/src/tools/stream_validate.py#L1-L18)及[mmap 逐张量路径](https://github.com/brianbell-x/weight-compression/blob/b9510aaac657b04e11d2f0d0d51a9b94af159590/src/tools/stream_validate.py#L104-L164)。它也明确区分“30.168% charged estimate”“24.967% 已独立 decode”与“未融合 sparse correction 的 kernel timing”，[项目声明](https://github.com/brianbell-x/weight-compression/blob/b9510aaac657b04e11d2f0d0d51a9b94af159590/README.md#L3-L9)。Brevis 的论文证据也应这样分层，避免把尺寸估算、bit-exact、microbenchmark 和端到端速度混成一个结论。

`eval/run_eval.py` 应固定模型 revision，并对相同输入字节运行 Brevis、gzip、zstd、xz、OpenZL，至少报告：

- 完整 archive bytes，以及 program / side-info / payload 分解；
- synthesis、encode、decode、端到端 wall time 与 GB/s；
- `jobs=1` 和多线程、cold/warm、峰值 RSS；
- 每个 dtype/shape bucket 的压缩率、raw fallback 比例；
- 解压后逐文件 SHA-256 或逐张量 byte equality；
- 小张量、随机不可压缩张量和至少一个完整真实 safetensors 模型。

OpenZL 的 `pytorch` profile 只接受 `torch.save()` 产生的 ZIP 模型，[接口契约](https://github.com/facebook/openzl/blob/79fa6674b027e398544419cd51afe22b205e8699/custom_parsers/pytorch_model_parser.h#L11-L24)，不能直接把 safetensors 喂给它。公平评测应同时给出：同一完整 safetensors 文件上的 OpenZL `serial`，以及把每个 tensor payload 按 dtype 交给 OpenZL numeric/float graph 的 payload-only 结果；不要把不同输入格式的数字放在同一列而不说明。

OpenZL 可作为重要概念与速度基线，但不要复制其系统规模：当前官方文档仍声明单 payload 超过 500 MB 需切块、典型压缩内存可到约 payload 的 10 倍且核心库不支持 streaming，[限制说明](https://github.com/facebook/openzl/blob/79fa6674b027e398544419cd51afe22b205e8699/doc/mkdocs/doc/getting-started/library-limitations.md#L1-L9)。这恰好强化 Brevis 的论文差异：小型 DSL、每张量程序、显式块边界和 bounded-memory streaming。

## 明确不做

- 不让 LLM、PHOG 或训练权重参与解压；归档只依赖版本化 op ID、参数和 payload。
- 不执行任意 Python。KoLMogorov-Test 的官方评测需要多进程 timeout 包住 `exec`，[源码](https://github.com/facebookresearch/kolmogorov/blob/690236b2192e5bc8d591f0f475367c92434a60a1/src/synthetic_data_generation/evaluate.py#L135-L172)，这不适合归档格式。
- 不把 sampled observational equivalence 当成正确性剪枝。Euphony 会按样例签名合并 partial programs，[实现](https://github.com/wslee/euphony/blob/0b9a62a294897b707c86ba3c0f63561ff503cb3b/bin/phogs/phog.py#L382-L403)；张量采样之外的数据可能不同，只能将它作为搜索提示，最终完整 frame 必须 bit-exact 验证。
- 不先扩充大量 transform。首轮论文 DSL 应能完整列出、形式化每个节点的类型与逆语义。
- 不在 CPU 主线稳定前引入 JIT/CUDA。先用块并行、SIMD primitive 和少量 fused pattern 证明 executor 不再是瓶颈。

## 最小论文消融

1. `uniform vs PCFG(parent) vs PHOG(parent + child_slot + dtype)`：expanded states、synthesis time、archive bytes。
2. `generic DSL` vs `+ bitfield_split` vs `+ topk4_escape`：按 BF16/F16/F32 报告收益。
3. `search per block` vs `search once per tensor + refit per block`：时间、峰值内存、压缩率。
4. `single-thread` vs `block-parallel` vs `SIMD/fused`：只测 decode，另报端到端。
5. sampled winner 在完整块上的 raw fallback 和 program change 率：直接量化采样搜索的风险。

这五组实验足够支撑核心论点；其余 parser、模型框架集成和 GPU serving 都应作为后续工作，而不是进入首篇论文的核心实现。

当前 smoke ablation 只能说明实现路径已打通，不能说明先验已学好：同模型校准的 PHOG 相对 uniform，BF16 归档缩小 0.011%，F32 和混合 I8 分别增大 0.00086% 和 0.00011%。因此现阶段应把 PHOG 表述为搜索顺序机制；压缩率收益必须等 held-out prior 实验后再下结论。相反，块并行解码已有清晰信号：三组模型相对 `--jobs 1` 分别加速 6.53×、6.18×、4.39×。原始结果固定在 `eval/results.json`。
