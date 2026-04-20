# Stage 1 返工计划(高风险 Recall 修复)

> 这份文档记录**第一版微调完成后**发现的数据问题,等整个项目跑通后回来做这一步修复。
>
> 触发时机:面试演示前 / Agent 在心理问答链路跑通后第一次端到端体验差时。

---

## 1. 为什么要返工

第一版训练结果显示:**微调后整体准确率上升,但高风险 Recall 反而下降**。这是心理安全场景下不可接受的倒退。

### 核心证据(第一版 eval,aligned prompt)

| 类别 | Baseline Recall | Fine-tuned Recall | Δ |
|---|---|---|---|
| 正常 | 71.5% | 88.0% | +16.5 ✅ |
| 焦虑 | 70.3% | 74.3% | +4.0 ✅ |
| 低落 | 83.0% | 93.0% | +10.0 ✅ |
| **高风险** | **41.0%** | **16.0%** | **-25.0 ❌❌** |

高风险类混淆矩阵对比(100 条真实高风险):

```
            正常   焦虑   低落   高风险
Baseline     0     4     55     41
Fine-tuned   0     6     78     16     ← 78 条被误判为低落
```

### 根因(已用数据验证,不是猜测)

**训练数据有词汇偏置(keyword leak)**:

```
训练集 高风险 133 条  —— 87% 含极端词(想死/自杀/自残/跳楼/...)
测试集 高风险 100 条  ——  6% 含极端词
```

凶手是 [scripts/data/weak_label.py#L21-22, L93-95](scripts/data/weak_label.py) 的 `RISK_RE` 快捷通道:

```python
RISK_RE = re.compile(r"(想死|自杀|自残|跳楼|割腕|结束生命|活不下去|不想活了)")
...
if RISK_RE.search(text):
    label, conf = "高风险", 0.95  # ← 这里越过了 LLM 的语义判断
    stats["risk_shortcut"] += 1
```

原意是"提速 + 确保高风险样本进训练集",实际效果是**模型学会了词汇匹配而非语义理解**:

- 训练样本:"...为什么不去死呢..." / "...准备在十二月的一天去死..."
- 测试样本:"每天醒来第一件事就是后悔自己还活着" / "我连呼吸都觉得累"

测试集几乎不用这些词,模型就认不出来了。

### 附加问题:类别极度稀少

当前训练分布:
- 低落: 1270 (50.8%)
- 焦虑: 1024 (41.0%)
- 高风险: **148 (5.9%)**
- 正常: **58 (2.3%)**

两个小类样本量都严重不足。weighted CE 的权重只能放大梯度,不能创造多样性信号。

---

## 2. 待办任务(三件事)

### 任务 1:去掉 `RISK_RE` 快捷通道(必做)

**改哪里**:[scripts/data/weak_label.py](scripts/data/weak_label.py)

**怎么改**:把 `RISK_RE` 和对应的 shortcut 分支删掉,所有样本都走 `call_llm(...)`。

```python
# 删除这段(约 L20-22, L93-95):
RISK_RE = re.compile(r"(想死|自杀|自残|跳楼|割腕|结束生命|活不下去|不想活了)")
...
if RISK_RE.search(text):
    label, conf = "高风险", 0.95
    stats["risk_shortcut"] += 1
else:
    label, conf = call_llm(client, args.model, text)

# 改成:
label, conf = call_llm(client, args.model, text)
```

**为什么这么改**:让 LLM 基于**整段语义**判断标签 + 输出 confidence,置信度阈值 0.7 过滤掉模糊样本。这样高风险样本里会**自然出现**不含关键词的语义高风险(LLM 能看出"后悔活着"是高风险)。

**预期影响**:
- LLM 调用次数增加(原来 129 条走 shortcut 跳过 LLM,现在都走)→ 打标时间略增
- 高风险样本数**可能减少**(LLM 比关键词规则严格),所以需要任务 2 补齐

**重跑命令**:
```bash
python3 -m scripts.data.weak_label \
  --input data/candidate.jsonl \
  --output data/labeled_v2.jsonl \
  --model qwen2.5:7b \
  --threshold 0.7 \
  --max-samples 2500
```

---

### 任务 2:合成 500 条语义高风险训练样本(必做)

**目的**:把高风险类从 ~150 → ~650,让模型见到足够多的**不依赖词汇的**高风险表达。

**怎么做**:复用现有 [scripts/data/synth_test.py](scripts/data/synth_test.py) 的 prompt 逻辑,写一个新脚本 `scripts/data/synth_train_augment.py`,**用相同的 teacher(mistral-large-3:675b-cloud)**,但:
- 样本量:`--n-risk 500`(test 是 100,训练要 5 倍)
- 额外一条 prompt 指令:**"不要在句子里使用'想死、自杀、自残、跳楼、割腕'这些词"**——强制生成语义高风险,避开词汇泄漏
- 输出 schema 保持和 train.jsonl 一致:`{instruction, input, output}`

**注意**:
- teacher 必须继续用 **mistral-large 这类非 qwen 家族**,避免 teacher-student 自蒸馏污染
- 和 test 集的 100 条**不能重复**,用 dedup hash 过一遍
- 生成完**人工抽样 30 条复核**(花 15 分钟),质量烂的重跑

**新脚本骨架**(和 synth_test.py 的区别只有三点:输出路径、prompt 加禁用词、样本数):

```python
RISK_TRAIN_PROMPT = """请生成 {k} 条用于情绪分类训练的中文短句：

1. 全部第一人称，表达"高风险"情绪：强烈的绝望、放弃、活不下去、想结束、想伤害自己等。
2. 长度 15~120 字，语气沉重。
3. **严禁使用**：想死、自杀、自残、跳楼、割腕、结束生命、不想活了 等关键词。
   必须通过语义而非关键词表达高风险：如"后悔自己还活着"、"熬不下去"、"连呼吸都累"。
4. 话题多样：长期抑郁、家庭、失恋、学业崩溃、孤独等。
5. 不要重复。

只输出 JSON 数组：["句子1", "句子2", ...]"""
```

**跑完后**,把 500 条 append 到 `data/labeled_v2.jsonl`,再跑 [scripts/data/split_dataset.py](scripts/data/split_dataset.py) 重做 9:1 切分。

---

### 任务 3:(可选)合成 200 条"正常"训练样本

**目的**:正常类当前只有 58 条(占训练集 2.6%),weighted CE 权重已经调到 4.7 倍还是信号不足。

**怎么做**:仍然用 mistral 合成,prompt 要求"日常、无情绪困扰、像普通学生的闲聊/记事/小确幸"。

```python
NORMAL_TRAIN_PROMPT = """请生成 {k} 条用于情绪分类训练的中文短句：

1. 全部第一人称，情绪中性或正向：日常琐事、小确幸、学习记事、周末计划、吃饭、天气等。
2. 长度 10~80 字，语气自然。
3. **不要**出现任何压力、焦虑、低落、绝望的表达。
4. 主题多样：食物、天气、运动、娱乐、学习进展、社交。
5. 不要重复。

只输出 JSON 数组"""
```

**性价比**:200 条 × mistral 速度(~1.5s/条)= ~5 分钟,收益可能很大(因为基数太低)。

**为什么标"可选"**:正常类在测试上本来就 88% recall(fine-tuned)——没那么痛。先做 1+2,如果效果不够再加 3。

---

## 3. 返工后的执行流程(一条龙)

```bash
# ① 重 labeling(去 shortcut)
python3 -m scripts.data.weak_label \
  --input data/candidate.jsonl \
  --output data/labeled_v2.jsonl \
  --max-samples 2500

# ② 合成高风险训练增强
python3 -m scripts.data.synth_train_augment \
  --kind risk --n 500 \
  --output data/augment_risk.jsonl

# ③ (可选)合成正常增强
python3 -m scripts.data.synth_train_augment \
  --kind normal --n 200 \
  --output data/augment_normal.jsonl

# ④ 合并 + 重切分
cat data/labeled_v2.jsonl data/augment_risk.jsonl data/augment_normal.jsonl > data/labeled_final.jsonl
python3 -m scripts.data.split_dataset \
  --input data/labeled_final.jsonl \
  --train data/train.jsonl --val data/val.jsonl --ratio 0.9

# ⑤ 重训(脚本不改)
rm -rf checkpoints/qwen25-7b-psychqa
CUDA_VISIBLE_DEVICES=0 python3 -m scripts.finetune_qwen

# ⑥ merge + gguf + ollama
bash scripts/export_to_ollama.sh qwen2.5-7b-psychqa

# ⑦ 重跑 eval 对比
python3 -m scripts.eval.run_eval \
  --model qwen2.5-7b-psychqa \
  --test data/test.jsonl \
  --prompt-style instr \
  --pred-out data/eval/finetuned_v2.jsonl
```

---

## 4. 成功判据

| 指标 | 第一版 | 第二版目标 | 说明 |
|---|---|---|---|
| 高风险 Recall | 16% | **≥ 60%** | 核心指标,心理安全场景的硬门槛 |
| 高风险 F1 | 27.4% | ≥ 55% | Precision 也不能掉太狠 |
| 正常 F1 | 92.8% | ≥ 90% | 不能倒退 |
| 整体 Accuracy | 77.7% | ≥ 80% | 作为 sanity check |

**如果高风险 Recall 仍 < 60%**,说明数据增强不够或模型容量不够,下一步考虑:
- 把 LoRA rank 从 8 → 16
- 多训 1~2 个 epoch(但要监控过拟合)
- 引入 **Focal Loss**(先前放弃的方案重新考虑,因为这时"小类 = 难样本"可能已经等价)

---

## 5. 面试 narrative 更新

返工完成后,在面试话术里加一段"**我是怎么发现并修掉数据偏置的**":

> "第一版微调后我发现整体准确率上去了,但**高风险 Recall 从 41% 掉到 16%**。我没把这当成一次随机波动,而是做了 error analysis——扒开训练数据发现 **87% 的高风险样本都含'想死/自杀/自残'这几个关键词**,但测试集(独立 mistral 合成)只有 6% 含。模型学到的是**词汇触发器**而不是**语义理解**。
>
> 根因是我 pipeline 里为了加速在弱监督打标时开了一条'关键词命中直接标高风险'的快捷通道。**第二版**我去掉了这个 shortcut,让 LLM 全权判断;同时用 mistral-large 合成了 500 条**严禁含关键词**的语义高风险训练样本做数据增强。重训后高风险 Recall 升到 XX%,整体准确率 YY%。
>
> 这是一次典型的 **data bias → representation bias** 诊断,我把这次经验总结成了一份内部 TODO,作为后续做中文分类类任务的 checklist——**警惕任何'规则规避 LLM 判断'的提速 shortcut**,它看似加速实则制造偏置。"

---

## 6. 预计工时

| 任务 | 时间 |
|---|---|
| 修 weak_label.py + 重跑 labeling | 30 分钟(LLM 打标主要耗时) |
| 写 synth_train_augment.py + 合成高风险 500 条 | 15 分钟 |
| (可选)合成正常 200 条 | 5 分钟 |
| 重新训练(3 epochs) | ~25 分钟 |
| merge + gguf + ollama create | ~3 分钟 |
| 重跑 eval 对比 | ~10 分钟 |
| **总计** | **~85~90 分钟** |

单次迭代 1.5 小时,一次就能拿到可讲的对比数字。
