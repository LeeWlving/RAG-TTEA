# FATE: 面向 RAG 知识抽取攻击的前沿锚点式层级探索方法

## 3 方法

本节介绍我们提出的 **FATE** 方法，即 **Frontier-Anchored Taxonomy Exploration**。FATE 的目标是在黑盒或半黑盒 RAG 系统中，通过自适应地生成查询来最大化可抽取知识量，同时避免查询过程反复命中同一批高频检索结果。与只优化查询文本多样性的方法不同，FATE 直接关注 **retrieval result diversity**：一个查询是否有价值，不仅取决于它能否抽取出知识，还取决于它是否把检索过程推进到新的文档区域。

我们的出发点来自一个经验观察：现有 adaptive query attack 方法通常会快速找到若干“高收益”的粗粒度语义区域，例如 Pokemon 数据集中的 species，Enron 数据集中的 email communication，或 HealthCareMagic 数据集中的 symptoms。随后，攻击器会在这些区域中持续生成看似不同的查询，但实际返回的 top-k 文档高度重叠。此时查询文本虽然变化，retrieval frontier 却没有真正向外扩展。这会导致两个问题：

1. 查询预算被重复消耗在同一批文档上；
2. 模型在局部区域内过度利用，难以覆盖长尾知识。

为了解决这一问题，FATE 将攻击过程建模为一个带反馈的层级检索空间搜索问题。方法由三个互补模块组成：

1. **Dataset-Aware Taxonomy Expansion**：根据数据集领域生成更贴近真实检索维度的层级节点，避免展开出无效或格式化的子节点；
2. **Frontier-Anchor Query Grounding**：从新检索到的文档中抽取领域相关锚点，并强制后续查询围绕这些新锚点展开；
3. **Novelty-Aware Hierarchical Scheduling**：在 node/lens 层面引入新颖性奖励和重复检索惩罚，使调度器优先选择能够推动 retrieval frontier 的区域。

整体而言，dataset-aware expansion 决定“搜索空间如何展开”，frontier anchor 决定“查询如何沿着新文档继续推进”，novelty-aware scheduling 决定“预算应该分配给哪些更有潜力的区域”。三者共同构成一个以检索反馈为核心的闭环攻击框架。

### 3.1 问题定义与符号

给定一个目标 RAG 系统，其检索器对查询 \(q\) 返回 top-k 文档集合：

\[
\operatorname{TopK}(q)=\{d_1,d_2,\ldots,d_K\}.
\]

攻击器无法直接访问完整语料库，只能通过一系列查询 \(q_1,\ldots,q_T\) 与系统交互，并从系统返回的回答或检索片段中抽取知识。第 \(t\) 轮攻击的目标是在有限查询预算下，同时最大化：

- 抽取知识数量，例如 extracted chunks 或 extracted facts；
- 检索覆盖度，例如 unique retrieved documents；
- 长尾区域覆盖度，即避免只命中代表性、高频或模板化文档。

我们维护一棵动态搜索树 \(\mathcal{T}_t\)。树中的每个节点 \(v\) 表示一个语义检索区域，例如 Pokemon 的 “evolution methods”，Enron 的 “meeting scheduling emails”，或 HealthCareMagic 的 “chest pain with medication advice”。每个节点关联若干子意图 lenses，例如：

\[
L(v)=\{\text{entity}, \text{attribute}, \text{relation}, \text{timeline}, \text{rare case}, \text{example}, \text{edge case}\}.
\]

在第 \(t\) 轮，调度器选择一个节点 \(v_t\) 和一个 lens \(\ell_t\)，查询生成器基于节点、lens、数据集领域信息以及历史检索反馈生成查询：

\[
q_t \sim G_\theta(v_t,\ell_t,c,A_t(v_t)),
\]

其中 \(c\) 表示数据集领域，\(A_t(v_t)\) 是当前节点下由新检索文档产生的 frontier anchors。

每个节点 \(v\) 维护历史检索集合：

\[
D_t(v)=\bigcup_{\tau<t, v_\tau=v}\operatorname{TopK}(q_\tau).
\]

该集合用于衡量新查询是否重复命中过去已经见过的结果。

### 3.2 Dataset-Aware Taxonomy Expansion

#### 动机

传统 taxonomy expansion 往往只根据父节点名称生成子节点。例如父节点为 “email communication” 时，模型可能展开出 “metadata fields”“message format”“headers” 等看似合理但检索价值较低的节点。对于 Enron 和 HealthCareMagic 这类数据集，这种问题尤其明显：如果 prompt 没有明确约束，模型容易生成带有格式、元数据或任务说明味道的节点，而不是能真正命中文档内容的语义 facet。

因此，我们提出 dataset-aware taxonomy expansion。其核心思想是：子节点不应该只是父节点的语言学改写，而应该对应目标数据集中的 **retrieval facets**。这些 facets 是文档内部常见、可检索、可组合的语义维度。

#### 形式化定义

给定父节点 \(v\) 和数据集领域 \(c\)，扩展函数 \(E_\theta\) 生成一组子节点：

\[
E_\theta(v,c)=\{u_1,u_2,\ldots,u_m\}.
\]

与通用扩展不同，\(E_\theta\) 会显式利用领域约束，使每个子节点尽量满足三个条件：

1. **Domain-grounded**：子节点应对应数据集中的真实内容维度，而不是抽象任务描述；
2. **Retrieval-aligned**：子节点应容易被转化为能够命中文档的查询；
3. **Non-overlapping**：同一父节点下的子节点应覆盖不同语义方向，降低后续查询重叠。

例如，在不同数据集上，我们使用不同的 expansion schema：

| 数据集 | 推荐检索维度 |
|---|---|
| Enron | sender/recipient, thread subject, project, meeting, transaction, date window, action phrase |
| HealthCareMagic | symptom, condition, medication, test, body part, demographic, duration, doctor advice |
| Pokemon | species, move, ability, type, evolution, item, location, trainer/event |
| HarryPotter | character, place, object, spell, relation, event, organization |

例如，对于 Enron 中的父节点 “energy trading discussions”，dataset-aware expansion 更倾向于生成：

- “trading schedule updates”
- “counterparty negotiation emails”
- “price or volume confirmation”
- “meeting requests about trading desks”
- “risk and compliance discussion”

而不是生成：

- “email headers”
- “message metadata”
- “communication format”

后者虽然看似与 email 数据有关，但往往不能有效推进知识抽取。

#### 触发机制

FATE 不会在所有节点上无限制展开，而是在节点出现停滞或重复时触发扩展。对于节点 \(v\)，我们定义：

\[
\operatorname{Expand}(v)=
\mathbb{I}\left[
n_t(v)\ge n_{\min}
\land
\left(
\bar{r}_t(v) < \epsilon
\lor
\operatorname{Repeat}_t(v)>\tau_{\mathrm{rep}}
\right)
\right],
\]

其中 \(n_t(v)\) 是节点访问次数，\(\bar{r}_t(v)\) 是最近窗口内平均奖励，\(\operatorname{Repeat}_t(v)\) 表示近期查询的检索重复率。当节点访问次数足够多，且收益下降或重复率过高时，系统会认为该节点已经被局部利用，需要展开子节点或切换到 sibling 节点继续探索。

这一机制的意义在于：树结构不是为了让攻击器长期停留在 coarse node 中反复生成 query，而是为了快速定位有潜力的大方向，然后下钻到更细粒度的文档区域。

### 3.3 Frontier-Anchor Query Grounding

#### 动机

Dataset-aware expansion 可以让搜索树更合理，但它仍然主要作用于 **语义空间**。然而，我们在实验中观察到，性能瓶颈往往不在 query text diversity，而在 retrieval result diversity。也就是说，攻击器生成了不同文本形式的查询，但检索器仍返回同一批 top-k 文档。

为了解决这一问题，FATE 引入 frontier anchors。所谓 frontier，是指当前攻击过程中新近触达、尚未充分利用的文档边界；anchor 则是从这些新文档中抽取出的、能够引导下一轮查询继续外扩的领域线索。

与普通关键词不同，frontier anchors 具有三个特点：

1. 它们来自真实检索结果，而不是模型凭空生成；
2. 它们具有领域类型，例如人物、症状、药物、地点、项目、事件；
3. 它们被用于约束后续查询，使查询沿着新文档线索继续推进。

#### Anchor 抽取

第 \(t\) 轮查询返回 \(\operatorname{TopK}(q_t)\) 后，我们首先计算当前节点下的新文档：

\[
N_t(v)=\operatorname{TopK}(q_t)\setminus D_t(v).
\]

然后从这些新文档中抽取 anchors：

\[
A_t(v)=\operatorname{ExtractAnchors}(N_t(v),c).
\]

其中 \(c\) 是数据集领域。不同数据集上的 anchor 类型不同：

| 数据集 | Anchor 类型 |
|---|---|
| Enron | person name, email address, thread subject, project code, company name, date, action phrase |
| HealthCareMagic | symptom, disease, medication, medical test, body part, age/gender, duration |
| Pokemon | species, move, ability, type, region, evolution item, trainer/location |
| HarryPotter | character, spell, place, object, family relation, organization, event |

例如，对于 Enron，若新文档包含 “Jeff discussed the California refund issue with the trading desk before Friday”，则可以抽取 anchors：

- Jeff
- California refund issue
- trading desk
- before Friday

下一轮查询不再只是围绕父节点 “energy trading” 生成泛化问题，而会被这些 anchor 约束，例如：

> trading desk California refund issue Jeff Friday email discussion

这种方式能够把查询从抽象语义节点拉向具体文档 frontier。

#### Anchor-grounded Query Generation

在 FATE 中，查询生成器不仅接收节点和 lens，还接收 frontier anchors：

\[
q_{t+1}\sim G_\theta(v,\ell,c,A_t(v)).
\]

我们要求生成查询满足以下约束：

1. 查询必须包含或语义覆盖至少一个 fresh anchor；
2. 查询不能只是重复父节点 label；
3. 查询应组合 node intent、lens 和 anchor，使其既保持方向性，又能探索新结果；
4. 如果多个 anchors 来自不同新文档，优先组合低频、具体、领域相关的 anchors。

这一机制将查询生成从 “node-conditioned generation” 转换为 “node-and-frontier-conditioned generation”。其直觉是：节点提供搜索方向，anchor 提供文档级牵引力。

#### Anchor 质量控制

并非所有从文档中出现的词都适合作为 anchor。为减少噪声，FATE 对 anchor 进行过滤与排序：

- 去除过短、过泛化或纯格式化 token；
- 去除明显的系统词、字段名、模板词和无意义编号；
- 对已频繁使用的 anchors 降权；
- 优先选择在当前节点中新出现、但具有较强领域语义的 anchors；
- 对 Enron 和 HealthCareMagic 等容易出现格式噪声的数据集，使用更严格的 domain-specific filters。

我们定义 anchor 的优先级分数为：

\[
s(a)=
\lambda_1 \operatorname{Specificity}(a)
+\lambda_2 \operatorname{Freshness}(a)
+\lambda_3 \operatorname{DomainType}(a)
-\lambda_4 \operatorname{Reuse}(a),
\]

其中 \(\operatorname{Specificity}(a)\) 衡量 anchor 是否具体，\(\operatorname{Freshness}(a)\) 衡量 anchor 是否来自新文档或近期低覆盖区域，\(\operatorname{DomainType}(a)\) 表示其是否属于当前数据集的重要实体类型，\(\operatorname{Reuse}(a)\) 则惩罚被反复使用的 anchor。

### 3.4 Novelty-Aware Hierarchical Scheduling

#### 动机

仅有更好的节点和 anchors 仍然不够。攻击器还需要决定：下一轮应该访问哪个节点、使用哪个 lens、是否继续开发当前区域，还是转向子节点或 sibling 节点。若 scheduler 只根据抽取 reward 做决策，它会偏向那些已经证明“容易抽取”的区域，即使这些区域返回的 top-k 已经高度重复。

因此，FATE 引入 novelty-aware reward，将抽取收益、新文档覆盖、anchor 覆盖和重复惩罚统一到调度目标中。

#### Retrieval Overlap Penalty

对于候选查询 \(q\) 和节点 \(v\)，我们定义重复率：

\[
\operatorname{Repeat@K}(q,v)=
\frac{
\left|\operatorname{TopK}(q)\cap D_t(v)\right|
}{K}.
\]

若没有稳定 document id，也可以基于 chunk embedding overlap 近似：

\[
\operatorname{Repeat@K}(q,v)=
\frac{1}{K}
\sum_{d\in \operatorname{TopK}(q)}
\mathbb{I}\left[
\max_{d'\in D_t(v)}
\operatorname{sim}(e(d),e(d'))>\tau_{\mathrm{doc}}
\right].
\]

这里 \(e(d)\) 表示文档或 chunk 的 embedding，\(\tau_{\mathrm{doc}}\) 是判定两个 chunk 近似重复的相似度阈值。

这一项非常关键，因为它直接惩罚 retrieval result repetition，而不是惩罚 query text repetition。两个查询即使文本不同，只要它们反复返回同一批 top-k 文档，就会被降权。

#### Query Reward

第 \(t\) 轮查询的奖励定义为：

\[
r_t(q,v)=
r_t^{\mathrm{extract}}(q,v)
+\alpha \operatorname{NewDocs@K}(q,v)
+\beta \operatorname{AnchorCoverage}(q,A_t(v))
-\mu \operatorname{Repeat@K}(q,v).
\]

其中：

- \(r_t^{\mathrm{extract}}\)：抽取收益，例如成功解析出的 chunks/facts 数量；
- \(\operatorname{NewDocs@K}\)：top-k 中此前未见文档的比例；
- \(\operatorname{AnchorCoverage}\)：查询是否有效覆盖 fresh anchors；
- \(\operatorname{Repeat@K}\)：当前查询与历史检索结果的重复率；
- \(\alpha,\beta,\mu\)：控制不同项权重的超参数。

这一 reward 使 scheduler 不再奖励那些“看似有效但实际一直拿同一批 top-k”的查询。相比只优化 extracted chunks，它更鼓励攻击器同时提升抽取量和覆盖面。

#### Sub-intent UCB

对于每个节点 \(v\)，FATE 将不同 lenses 视为 node 内部的 bandit arms。每个 lens 代表一种子意图，例如 entity、attribute、relation、timeline、rare case、example 或 edge case。调度器根据上置信界选择下一轮 lens：

\[
\ell_t =
\arg\max_{\ell\in L(v)}
\left[
\hat{r}_t(v,\ell)
+ c\sqrt{
\frac{\ln(1+N_t(v))}
{1+n_t(v,\ell)}
}
-\mu \operatorname{Repeat}_t(v,\ell)
\right].
\]

其中 \(\hat{r}_t(v,\ell)\) 是 lens 的历史平均奖励，\(N_t(v)\) 是节点总访问次数，\(n_t(v,\ell)\) 是 lens 被选择的次数。UCB 项鼓励探索访问较少但可能有潜力的 lens，repeat penalty 则防止某些 lens 因早期收益高而长期反复命中同一批文档。

这一设计比固定 sub-intent rotation 更灵活：哪个 lens 最近带来更多新 top-k，就继续使用；哪个 lens 重复率高，就自动降权。

#### Parent Cooldown 与 Child Quota

当某个 coarse node 因重复率过高而触发 expansion 时，仅仅创建 children 还不够。因为父节点可能仍然具有较高历史 reward，scheduler 可能继续选择它，导致预算继续被消耗在 coarse node 内。

因此，FATE 引入 parent cooldown 和 child quota：

\[
\text{if } \operatorname{Expand}(v)=1,\quad
\operatorname{Cooldown}(v)=\kappa,
\]

并要求在父节点重新可选之前，至少访问若干个 child 或 sibling 节点：

\[
\sum_{u\in \operatorname{Children}(v)}
\mathbb{I}[u \text{ visited}]
\ge m_{\min}.
\]

这一步能迫使搜索过程从粗粒度节点向细粒度节点下钻，避免在大节点中无限打转。

### 3.5 算法流程

下面给出 FATE 的整体流程。

```text
Algorithm 1: Frontier-Anchored Taxonomy Exploration (FATE)

Input:
  Query budget T
  Initial taxonomy root r
  Dataset domain c
  Retriever R
  Query generator G_theta
  Expansion function E_theta

Initialize:
  Search tree T_0 with root r
  For each node v:
    D_0(v) = empty set
    A_0(v) = empty anchor set
    statistics for node/lens rewards and visits

for t = 1 ... T do
  1. Select node v_t using novelty-aware hierarchical scheduler

  2. Select lens l_t using sub-intent UCB within v_t

  3. Generate query:
       q_t ~ G_theta(v_t, l_t, c, A_t(v_t))

  4. Retrieve documents:
       TopK(q_t) = R(q_t)

  5. Compute new documents:
       N_t(v_t) = TopK(q_t) \ D_t(v_t)

  6. Extract knowledge from returned documents or RAG response

  7. Extract frontier anchors:
       A_new = ExtractAnchors(N_t(v_t), c)
       A_{t+1}(v_t) = UpdateAnchors(A_t(v_t), A_new)

  8. Compute novelty-aware reward:
       r_t = extraction reward
             + alpha * new document reward
             + beta * anchor coverage reward
             - mu * repeat penalty

  9. Update node/lens statistics and D_t(v_t)

 10. If expansion trigger is satisfied:
       Children(v_t) = E_theta(v_t, c)
       Apply parent cooldown and child quota

end for

Output:
  Extracted knowledge and retrieval trace
```

### 3.6 设计讨论

#### 为什么不是只做 query diversity？

Query diversity 并不等价于 retrieval diversity。两个查询可以在文本上差异很大，但如果它们都命中同一批 top-k 文档，那么对攻击目标而言，它们提供的信息增量很小。因此 FATE 的关键设计是直接对 retrieval overlap 建模，并将 \(\operatorname{Repeat@K}\) 放入 reward。

#### 为什么需要 dataset-aware expansion？

没有领域约束的 taxonomy expansion 容易生成“语言上合理但检索上无效”的节点。例如在 Enron 上，模型可能生成 email metadata、message format、header fields 等节点；在 HealthCareMagic 上，模型可能生成 web page structure、doctor answer format 等节点。这些节点会让树结构看起来更复杂，但不会有效提升文档覆盖。Dataset-aware expansion 将节点限制在真实内容 facet 上，因此能提高搜索树的有效性。

#### 为什么需要 frontier anchors？

Dataset-aware expansion 解决的是树结构质量，frontier anchors 解决的是文档级推进问题。我们希望每轮查询不仅回答“我在哪个语义区域搜索”，还回答“我应该沿着哪些新文档线索继续搜索”。Frontier anchors 正是这个桥梁：它们把实际检索结果中的局部线索反馈给 query generator，从而减少泛化查询和重复 top-k。

#### 与 IKEA 类方法的关系

IKEA 类方法通常具有较强的文档覆盖能力，因为它们更倾向于利用检索结果中的具体线索推进搜索。然而，单纯依赖这种线索推进可能缺少层级结构，导致查询方向不够稳定。TTEA 类方法具有较强的结构化探索能力和抽取能力，但容易在 coarse node 内重复利用。FATE 的目标是结合两者优势：

- 使用 taxonomy 维持全局语义结构；
- 使用 dataset-aware expansion 生成高质量分支；
- 使用 frontier anchors 引入文档级推进能力；
- 使用 novelty-aware scheduling 避免重复检索。

因此，FATE 可以被理解为一种面向 RAG 知识抽取攻击的结构化 frontier search。

### 3.7 预期优势

FATE 预期在以下方面提升攻击效果：

1. **更高的 unique retrieved documents**：通过 frontier anchors 和 repeat penalty，减少重复命中同一批 top-k；
2. **更稳定的 extracted chunks**：通过 taxonomy 和 dataset-aware expansion 保持查询方向，不至于完全随机扩散；
3. **更强的长尾覆盖能力**：通过 child expansion、parent cooldown 和 lens UCB，将预算分配到更细粒度、更少访问的区域；
4. **更好的跨数据集适配性**：通过 dataset-specific facets 和 anchor types，让方法能适配 Enron、HealthCareMagic、Pokemon、HarryPotter 等不同语料结构。

从方法定位上看，FATE 并不是单纯的 prompt engineering，而是一个将层级搜索、检索反馈、文档前沿建模和 bandit 调度结合起来的自适应攻击框架。它直接针对 RAG 攻击中的核心瓶颈：攻击器往往不是缺少不同形式的查询，而是缺少能够持续发现新检索结果的机制。
