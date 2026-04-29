# SmolVLA vs Paper: Divergence Summary

## Purpose

This note summarizes where our SmolVLA reproduction matches the paper
_Mechanistic Interpretability for Steering Vision-Language-Action Models_
and where it necessarily diverges because of architectural and evaluation differences.

## What We Preserved Faithfully

### 1. The core FFN-value-vector interpretation

We preserved the paper’s central mechanistic idea:

- FFN outputs are treated as weighted sums of value vectors.
- Each FFN neuron corresponds to one value vector, i.e. one column of `down_proj.weight`.
- We decode those value vectors into token space using the language-model head.

This means our interpretability pipeline follows the same basic logic:
- identify FFN value vectors,
- assign semantic meaning using token projections,
- group them into concept-like sets,
- intervene on the corresponding FFN activations.

### 2. The intervention target

We aligned the intervention point to the paper’s mechanism:

- the intervention is applied **before** `down_proj`,
- using a `forward_pre_hook` on `mlp.down_proj`,
- so selected FFN activation coordinates are overwritten before they are mixed into residual space.

This is an important point of fidelity. We are modifying FFN activations, not residual outputs.

### 3. Cluster-vs-random causal comparison

We also preserved the paper’s causal comparison logic:

- concept-aligned neuron set
- matched random neuron set
- compare whether the concept set changes behavior more than the random control

This is the backbone of the causal claim in our reproduction as well.

---

## Main Divergences

## 1. SmolVLA is a continuous-action policy, not an action-token policy

This is the largest divergence.

The paper’s clearest reproductions focus on VLA systems where the model directly predicts action tokens. In that setting, there is a natural bridge from decoded internal vectors to tokenized action behavior.

SmolVLA is different:

- it uses a SmolVLM-style backbone,
- coupled to an action expert,
- and produces continuous action chunks rather than discrete action tokens.

So our intervention still happens in the language/vision transformer, but the downstream behavior is mediated through a continuous-action control head rather than an action-token decoder.

### Consequence

Our strongest claims are about:
- causal changes in rollout behavior,
- motion statistics,
- transfer across visual contexts,

not about direct manipulation of action-token probabilities.

---

## 2. We interpret the VLM backbone, not the entire end-to-end action policy equally

The paper’s framing can be read as directly connecting internal semantic directions to VLA outputs.

In SmolVLA, the architecture is more layered:

- we extract and interpret value vectors from the VLM text backbone,
- then test whether those backbone interventions causally affect the downstream action expert.

So there is an extra indirection:

semantic FFN direction -> modified backbone state -> changed continuous actions

rather than:

semantic FFN direction -> changed action-token logits

### Consequence

Our results support the claim that semantic interventions in the backbone can affect downstream control, but not that the model internally represents actions in the same tokenized way as the paper’s main systems.

---

## 3. Our semantic labeling pipeline is approximate rather than fully manual

The paper includes a stronger human-interpretation component when identifying meaningful value vectors and semantic clusters.

Our pipeline is more automated:

- decode top LM-head tokens for each value vector,
- use keyword/pattern heuristics,
- build semantic embeddings,
- cluster them,
- rank clusters against concept anchors such as `fast` and `risk`

This is practical and scalable, but it is not a literal replication of a human curation pipeline.

### Consequence

Our concept labels should be treated as operational labels for intervention sets, not as perfect ground-truth semantic identities.

This is especially important for weaker concepts like `fast` and `safe`.

---

## 4. Our clearest effects are in motion statistics, not task success

The paper’s strongest claims are often framed around behaviorally meaningful steering in robotic settings.

In our current SmolVLA experiments:

- success rate remains low or zero in the short-horizon settings we used,
- the clearest effects appear in motion metrics such as mean end-effector displacement,
- and the strongest evidence comes from cluster-vs-random differences in those metrics.

### Consequence

Our reproduction currently supports:
- causal behavioral modulation,
- robustness of those modulations across contexts,

more strongly than:
- reliable task-success steering

So this is a mechanistic-control result first, and a task-performance result only weakly at this stage.

---

## 5. Our evaluation scope is narrower than the paper’s full benchmark scope

We have:

- a full value-vector extraction pipeline,
- clustering and steering infrastructure,
- alpha sweeps,
- init-state transfer tests,
- nuisance perturbation transfer tests

But the current experiments are still narrower than a maximal paper-style benchmark reproduction:

- mostly one core LIBERO task,
- limited rollout counts,
- no full all-task sweep yet,
- no exhaustive cluster-size sweep yet

### Consequence

Our results are best understood as:
- a strong mechanistic adaptation to SmolVLA,
- with meaningful causal and transfer evidence,
- but not yet a full-scale benchmark reproduction at the paper’s largest evaluation breadth.

---

## 6. Transfer analysis is a new emphasis in our project

One place where our project adds something beyond a narrow reproduction is the transfer question:

- fixed init-state context transfer,
- brightness transfer,
- occlusion transfer

This goes beyond merely showing that steering works in one setting.

### Consequence

Our project’s scientific center of gravity is:

“Do mechanistic steering effects transfer across visual contexts?”

This is fully compatible with the paper’s framework, but it is more specific than a direct one-to-one reproduction target.

---

## What We Can Claim Safely

Based on the current results, the strongest defensible claims are:

- We successfully adapted the paper’s FFN value-vector steering logic to SmolVLA.
- We identified concept-aligned clusters, especially `risk`, that produce stronger behavioral effects than matched random controls.
- Those effects transfer across paired init states and across the visual perturbations we tested.
- Therefore, at least some internal steering directions in SmolVLA behave like portable control features rather than purely scene-specific artifacts.

## What We Should Avoid Overclaiming

We should avoid claiming that:

- SmolVLA reproduces the paper’s action-token behavior exactly.
- Our concept labels are perfectly disentangled semantic ground truth.
- We have shown broad task-success improvement.
- We have already matched the paper’s evaluation scale in full.

---

## Compact Report Version

Our SmolVLA reproduction preserves the paper’s core FFN value-vector logic and the correct pre-`down_proj` intervention, but it differs in one major way: SmolVLA is a continuous-action VLM-plus-expert policy rather than an action-token VLA. As a result, our strongest evidence is not direct action-token steering, but causal changes in rollout behavior and their transfer across visual contexts. We also use a more automated semantic-labeling and clustering pipeline than a fully manual paper-style annotation workflow. The result is a faithful mechanistic adaptation, but not a literal one-to-one architectural reproduction.
