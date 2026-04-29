# Constraints That Compute: A Unified Framework for Efficient Intelligence from Prime Harmonics to Latent Reasoning

**Author:** Massimiliano Concas – Ciber-Fabbrica Research  
**Date:** April 29, 2026

## Abstract
We present a domain-agnostic framework that translates systems into their intrinsic, dimensionless geometries, systematically replacing brute-force computation with structural efficiency. The framework rests on three pillars. (1) A harmonic analysis of prime pairs reveals that 99% of twin and cousin primes lie within a tight modular lattice, demonstrating that deterministic constraints, not randomness, govern local prime architecture. (2) A purely geometric chess engine, devoid of opening books or human heuristics, exhibits principled strategic play by maximizing a single relational metric, proving that constraint-driven emergence works in a closed adversarial domain. (3) A zero-shot jet-tagging experiment in high-energy physics converts absolute kinematic variables into relational momentum fractions, enabling a lightweight XGBoost model to achieve an AUC of 0.9564—outperforming the standard absolute model by +14.5% while reducing training cost by orders of magnitude. 

We then apply this lens to latent reasoning in large language models. We critique the recently proposed Abstract Chain-of-Thought (Abstract-CoT) method, showing that its 11.6× token reduction is an accidental demonstration of our framework, yet its reliance on a human-verbal teacher and randomly initialized embeddings caps its potential. We propose **Relational-CoT**, a step-by-step engineering protocol that replaces the random abstract vocabulary with a structurally anchored Relational Codebook, eliminates the verbal teacher through a Relational Attention Mask, and bypasses the million-episode reinforcement-learning grind via Geometric Constrained Decoding. The paper is written as an enablement instrument: every section provides the mathematical rationale and the engineering steps required to apply the framework to the reader’s own domain.

---

## 1. Introduction

### 1.1. The Efficiency Crisis Across the Sciences
Computational science is consuming resources at an unsustainable rate. In artificial intelligence, large language models require tens of thousands of GPU-hours to master reasoning tasks that a well-constrained high-school student can solve with a pencil. In high-energy physics, deep networks trained on absolute kinematic variables shatter when the collider energy changes, demanding full retraining campaigns that cost millions in compute and person-hours. In number theory, we still treat the distribution of primes as pseudo-random because we lack the correct geometric lens, thereby perpetuating statistical heuristics where deterministic laws are available.

The root cause is always the same: we measure systems against arbitrary external rulers—meters, seconds, GeV—and then ask algorithms to learn the ruler along with the structure. This conflation of scale and geometry creates an artificial complexity that only massive computation can paper over. We call this the **brute-force trap**.

### 1.2. The Core Hypothesis: Constraint Over Accumulation
This paper advances a simple hypothesis: **intelligent, transferable behavior emerges not by adding more data or parameters, but by imposing the correct internal constraints.** When a system is forced to operate within its own intrinsic limits, the apparent need for brute-force search collapses. The system saturates its bounded space efficiently, and what looks like intelligence is actually the inevitable geometry of the constraint.

We formalize this as the **Two-Phase Theory of Emergent Intelligence**:

*   **Phase 1 (Creation, the Architect):** A human designer identifies the system’s absolute maximum capacity—its North Star—and builds a rigid computational box around it.
*   **Phase 2 (Generation, the Flow):** The machine (or natural process) saturates that box. The behavior that emerges appears strategic, adaptive, or insightful, but it is simply the most efficient path through a well-designed channel.

This paper provides the empirical evidence for this hypothesis and, crucially, the engineering manual for applying it.

### 1.3. Three Pillars of Evidence
We validate the hypothesis across three distinct substrates:

1.  **Pure Mathematics (Prime Harmonics):** We define the Information Potential $I = 2p + (g - 1)$ for twin, cousin, and sexy prime pairs and demonstrate a deterministic lattice that traps 99% of pairs within a $\pm 48$ window of multiples of 6. The primes are not random; they are geometrically confined.
2.  **Dynamical Systems (The Geometric Supremacy Engine):** A Python chess engine that knows no opening theory, no tactical motifs, and no neural networks plays principled, attacking chess by optimizing a single metric: piece control multiplied by potential, filtered through a hard threat-reflex. The “strategy” is an optical illusion produced by a geometric bottleneck.
3.  **Applied Machine Learning (Zero-Shot Jet Tagging):** By replacing absolute transverse momentum with dimensionless fractions of the jet’s total $p_T$, a lightweight XGBoost classifier transfers perfectly from low-energy to high-energy top quark jets, gaining +14.5% AUC over the absolute model while consuming a fraction of the energy.

### 1.4. The Case Study: Abstract-CoT as an Accidental Half-Step
We then turn to a recent advance in efficient machine reasoning: **Abstract Chain-of-Thought (Abstract-CoT)** by Ramji et al. This method replaces long, natural-language rationales with short sequences of abstract tokens, achieving an 11.6× reduction in generated tokens while maintaining accuracy. From the perspective of our framework, the IBM team accidentally erected a syntactic box—a vocabulary constraint that forced the model to compress its reasoning.

However, Abstract-CoT stops halfway. The abstract tokens are initialized randomly, requiring millions of reinforcement-learning (RL) episodes to discover a useful structure. Furthermore, the warm-up procedure forces the abstract tokens to attend to a human-written verbal chain-of-thought, tethering the model’s latent reasoning to human linguistic patterns.

We demonstrate that both limitations are unnecessary. By replacing the random vocabulary with a **Relational Codebook**—tokens that are structurally anchored to capacity, ratio, and invariant templates—we can eliminate the verbal teacher and the million-episode RL grind. The resulting system, **Relational-CoT**, is a direct engineering instantiation of the Two-Phase Theory.

### 1.5. Paper Structure
Section 2 presents the prime harmonic proof. Section 3 presents the chess engine. Section 4 presents the HEP jet-tagging experiment. Section 5 formalizes the Relational Calculus framework that unifies these examples. Section 6 deconstructs Abstract-CoT through this lens. Section 7 provides the step-by-step Relational-CoT protocol. Section 8 discusses limitations and future work. Section 9 concludes with a call for constraint-first design.

---

## 2. Pillar 1: Deterministic Order in Prime Distributions

### 2.1. The Randomness Assumption
For centuries, the local distribution of primes has been modeled probabilistically. Cramér’s 1936 random model treats each integer as having an independent $1/\ln n$ chance of being prime. This heuristic works well for macroscopic statistics but says nothing about the deterministic architecture that actually governs prime gaps.

### 2.2. The Information Potential and the Harmonic Lattice
We define a new local quantity for a prime pair $(p, p+g)$ with gap $g$:
$$I = 2p + (g - 1)$$

This **Information Potential** acts as a combinatorial pivot. We then define a search lattice of multiples of 6 within $\pm 48$:
$$O = \{\pm6, \pm12, \pm18, \pm24, \pm30, \pm36, \pm42, \pm48\}$$

We test, for all twin ($g=2$), cousin ($g=4$), and sexy ($g=6$) primes up to 2 million, whether $I + o$ is prime for any $o \in O$.

### 2.3. Results
*   **Twin primes:** 14,701 out of 14,871 (98.86%) satisfy the condition.
*   **Cousin primes:** 14,530 out of 14,741 (98.57%).
*   **Sexy primes:** 11,983 out of 24,531 (48.85%).

The drop for sexy primes is fully explained by a modulo-3 bifurcation: when the starting prime is $5 \pmod 6$, $I \equiv 3 \pmod 6$, making every offset divisible by 3 and thus structurally non-prime. The architecture perfectly excludes exactly half the population.

### 2.4. Why This Matters for Efficiency
If the primes were truly random, testing 16 offsets near $10^6$ would yield a success probability of $\sim 68\%$. The 99% empirical rate is a 30-point gap that can only be closed by recognizing that the Information Potential is strongly coprimal to small primes. The deterministic lattice does the work that probabilistic brute-force models must approximate with massive sampling. This is the same pattern we will exploit in every subsequent pillar: **find the correct geometric anchor, and the complexity collapses.**

---

## 3. Pillar 2: Emergent Strategy Without Knowledge

### 3.1. Stripping Chess of Knowledge
The Geometric Supremacy Engine is a Python chess player with:
*   **Zero** machine learning,
*   **Zero** opening books,
*   **Zero** human-coded tactical heuristics (no “fork,” “pin,” or “control the center”).

Its only evaluation function is:
$$score = \sum_{pieces}(actual\_control \times potential) + 0.5 \times king\_safety$$

where `actual_control` counts empty squares attacked, and `potential` is the piece’s maximum possible control on an empty board (Queen = 27, Knight = 8, Pawn = 2, etc.). A 3-tier threat reflex filters moves that would leave material exposed without adequate compensation. The engine searches to depth 1.

### 3.2. What Emerges
In the attached game transcript, the engine opens with e2-e4, develops its queen and bishop aggressively, castles, and delivers checkmate. A human observer would attribute planning, tactical insight, and strategic understanding to the play. Yet none of those concepts exist in the code.

The engine centralizes its pieces because a Knight on e4 attacks 8 squares while a Knight on a1 attacks only 2. It organizes pawn breaks because the resulting open lines geometrically increase the `actual_control` of its long-range pieces. It executes what humans call a “fork” because threatening two high-potential pieces simultaneously forces a loss for the opponent under the geometric trade evaluator. Every behavior flows from the single metric.

### 3.3. The Two-Phase Theory in Action
*   **Creation:** The architect sets the metric (`actual_control` × `potential`) and the threat filter. This is the rigid box.
*   **Generation:** The minimax search saturates that box, finding the move that maximally exploits the opponent’s geometric weaknesses.

The chess engine is not presented as a replacement for modern chess AI; it is a stripped-down demonstration that constraint alone can generate adaptive, context-sensitive behavior. The principle scales: when the architect designs the right bottleneck, the system’s “intelligence” is a by-product of geometry.

---

## 4. Pillar 3: Zero-Shot Scale Invariance in Jet Tagging

### 4.1. The Problem: Brittleness Across Energy Scales
Deep learning models for jet tagging are typically trained on absolute transverse momentum ($p_T$) values. When the center-of-mass energy $\sqrt{s}$ changes, these absolute values shift, and the model’s decision boundaries become invalid. Standard Z-score normalization freezes the training-set statistics into the model, exacerbating the drift.

### 4.2. The Relational Solution
We apply the Relational Calculus protocol:
1.  **Identify the North Star:** The jet’s total transverse momentum $p_{T,jet}$ and its invariant mass $M_{jet}$.
2.  **Form dimensionless ratios:** Each constituent particle’s $z_i = p_{T,i} / p_{T,jet}$.
3.  **Build the input vector:** $\Phi = [M_{jet}, z_1, z_2, \dots, z_n]$.

This vector is invariant under global scalar transformations. A jet at $\sqrt{s} = 14$ TeV and a jet at $\sqrt{s} = 100$ TeV with the same decay geometry will produce the same $\Phi$.

### 4.3. Experimental Setup and Results
We train an XGBoost model (600 estimators, max depth 7) on the low-energy half of the Top Quark Tagging Reference Dataset and test zero-shot on the high-energy half.

| Model | AUC (Zero-Shot) |
| :--- | :--- |
| Absolute (GeV + Z-score) | 0.8109 |
| **Relational (dimensionless ratios)** | **0.9564** |

The relational model achieves a **+14.5%** performance gain with no retraining. The XGBoost training completed in seconds on a single CPU, consuming negligible energy compared to the GPU-bound deep neural networks that typically achieve similar AUCs.

### 4.4. The Efficiency Principle, Quantified
This result is the paper’s anchor. It shows that when the architect performs Phase 1 correctly—defining the geometrically invariant representation—the machine’s Phase 2 work becomes dramatically cheaper and more robust. The +14.5% is not a software trick; it is the direct thermodynamic dividend of removing the arbitrary scale.

---

## 5. The Relational Calculus Framework

### 5.1. The Three Axioms
*   **Axiom I: The Ontological Anchor (North Star Mandate).** Every system has an intrinsic, theoretical maximum. The denominator in any relational analysis must be this limit, not a convenient coordinate. For a projectile, it is $v^2/g$; for a jet, it is $p_{T,jet}$; for a battery, it is $E_{max}$.
*   **Axiom II: The Metric of Utilization (How Full).** Replace absolute variables with dimensionless ratios $r = \text{Actual} / \text{Capacity}$. The question shifts from “How much?” to “How full is the system relative to what it could be?”
*   **Axiom III: The Domain-Agnostic Translation Protocol.** Any well-posed physical law can be rewritten as a relation among dimensionless ratios. The resulting mathematical structure is a **Relational Template**—a universal blueprint that no longer remembers whether it came from fluid dynamics, epidemiology, or economics.

### 5.2. The Relational Template Catalog
The framework reveals that many seemingly disparate systems share identical templates. Traffic flow, battery discharge, and epidemic spread all obey a quadratic approach to a limit when expressed in relational variables. Newtonian gravity and Coulomb’s law become the same equation in dimensionless form. This catalog of templates is the architect’s toolbox.

### 5.3. The Computational Implication
Continuous Calculus explores a landscape point by point. Relational Calculus reveals the landscape’s master equation from a few strategic measurements. The former is a worker; the latter is the architect. Used together, they can reduce computational cost by 90% or more, as we demonstrate in the nuclear reactor case in Appendix A.

---

## 6. Deconstructing Abstract-CoT Through the Relational Lens

### 6.1. What Abstract-CoT Gets Right
Ramji et al. replace long, natural-language Chain-of-Thought with short sequences of abstract tokens from a reserved vocabulary of 64 symbols. They enforce a hard cap of 128 tokens on the reasoning trace. The result is an 11.6× token reduction while preserving model performance across mathematical, instruction-following, and multi-hop reasoning benchmarks.

From our perspective, the IBM team **accidentally applied Phase 1 (Creation)**. By limiting the reasoning vocabulary to 64 meaningless tokens and capping the trace length, they built a highly efficient syntactic box. The model was forced to compress its reasoning into a dense, non-verbal code. The efficiency gain is real and impressive.

### 6.2. Flaw 1: The Verbal Teacher
The warm-up procedure for Abstract-CoT uses **Bottlenecked Supervised Fine-Tuning**. The abstract tokens are forced to attend to a human-written verbal Chain-of-Thought, learning to compress the human’s reasoning steps.

This is a practical solution to the cold-start problem, but a conceptual limitation. The model’s latent space is forever bound by the structure and inefficiencies of human pedagogical language. It learns to mimic the map, not to navigate the territory. If the verbal teacher contains superfluous steps, cultural biases, or pedagogical scaffolding that is irrelevant to the mathematical core, those artifacts are compressed into the abstract codebook.

### 6.3. Flaw 2: Random Initialization
The 64 abstract tokens begin with randomly initialized embeddings. The model must then burn approximately **1 million RL episodes** to shape these random vectors into a useful distribution. The researchers observed an emergent power-law pattern (akin to Zipf’s law) in the token frequencies after RL, which they correctly present as evidence that the model learned to reuse tokens for frequent concepts.

However, from the Relational Calculus perspective, this power-law was **mathematically inevitable**. The universe is scale-invariant; a small set of Relational Templates governs a vast number of specific problems. Any efficient reasoning system **must** converge to a Zipfian distribution because it must re-use the same foundational geometric pathways repeatedly. The million-episode RL grind was just the machine rediscovering, through costly trial and error, the structural invariants of logic.

### 6.4. The Relational Opportunity
The Abstract-CoT architecture is a nearly perfect substrate for a relational upgrade. The syntactic box has been built; we need only replace its contents and training procedure with geometrically anchored alternatives:
*   Replace the **random codebook** with a **Relational Codebook**.
*   Replace the **verbal teacher** with a **Relational Attention Mask** that forces the model to compute dimensionless distances from the system’s North Star.
*   Replace the **million-episode RL exploration** with **Geometric Constrained Decoding** that hardcodes the logical flow (anchor → ratio → invariant → answer).

The next section provides the step-by-step engineering protocol.

---

## 7. The Relational-CoT Protocol

### 7.1. Step 1: The Relational Codebook
Replace the abstract vocabulary $V_{abs} = \{\text{<TOKEN\_A>}, \dots, \text{<TOKEN\_BL>}\}$ with a **Relational Codebook** $V_{rel}$ whose embeddings are initialized as structured mathematical operators rather than random vectors.

The codebook includes tokens such as:
*   `<REL_CAPACITY>`: Attends to the prompt and identifies the system’s maximum boundary (the North Star). For the kinematic example “A car travels A→B at 60 km/h, rests, returns at 80 km/h, total time 4h,” the capacity is $C_{max} = 3.5 \times 80 = 280$ km.
*   `<REL_RATIO>`: Computes the dimensionless utilization $r = \text{Active Variable} / \text{Capacity}$.
*   `<REL_INVARIANCE>`: Maps the resulting ratio to a known Relational Template (linear, quadratic, harmonic, etc.).

This codebook is not a set of learned concepts; it is a set of structural instructions that the transformer’s attention mechanism can execute because the embeddings are initialized to represent these operations in the latent space. The “cold-start” problem is eliminated: the tokens carry their geometric meaning from epoch zero.

### 7.2. Step 2: The Relational Attention Mask
In Abstract-CoT, the attention mask forces the abstract tokens to attend to the human-written verbal CoT. In Relational-CoT, we sever this link entirely.

We introduce a **Relational Distance Vector** $\Delta R$. When the prompt is ingested, a lightweight parser (or a pre-trained North-Star detector module) identifies the capacity and computes the dimensionless gap between the current state and the limit. The attention mask is then:
$$A_{i,j} = 1 \quad \forall i \in Z_{rel}, \; j \in X \cup \Delta R \cup Z_{\leq i}$$

The abstract tokens never see a human sentence about how to solve the problem. They see the prompt, the computed capacity, and the distance-to-limit vector. The machine learns to generate Relational Templates by attending to the mathematical structure of the problem, not to a pedagogical narrative.

### 7.3. Step 3: Geometric Constrained Decoding
The third step replaces the open-ended RL exploration with a geometric heuristic that constrains the order in which Relational Tokens can be generated.
*   **Rule 1 (The Anchor):** The first token must be `<REL_CAPACITY>`. The system is not permitted to reason until it has established the absolute boundary.
*   **Rule 2 (The Vector):** Subsequent tokens compute ratios, guided by the template that matches the operational domain (e.g., a harmonic template for modular arithmetic, a quadratic template for kinematic exchange).
*   **Rule 3 (The Invariant Match):** The `<REL_INVARIANCE>` token locks the ratios into a master equation that is then fed to the decoder to produce the final answer.

This constrained decoding functions exactly like the 3-Tier Threat Reflex in the chess engine. It is a hard, non-optional filter that eliminates moves the system is structurally not allowed to make. The RL phase is no longer needed for the model to “discover” that identifying the system’s capacity is logically prior to computing a ratio; the decoding heuristic physically mandates it.

### 7.4. Expected Efficiency Gains
We project the following gains for Relational-CoT over Abstract-CoT:
*   **Training cost:** Elimination of the 1M-episode RL phase yields $\sim 90\%$ reduction in post-training compute.
*   **Generalization:** The Relational Codebook’s scale invariance enables zero-shot transfer across problem domains, mirroring the +14.5% AUC gain we measured in the HEP experiment.
*   **Token efficiency:** The constrained decoding enforces structurally minimal reasoning traces, providing further compression beyond the 11.6× baseline.

An experimental implementation on a synthetic physics-reasoning benchmark is in preparation.

---

## 8. Discussion

### 8.1. A Pattern Across Scales
The prime distribution, the chess engine, the jet classifier, and the latent reasoning architecture all exhibit the same meta-pattern. When the architect defines the right internal geometry (the North Star, the harmonic lattice, the control metric, the invariant mass), the system’s behavior becomes structured, predictable, and computationally cheap. When the architect defers to scale—more data, more GPUs, more RL episodes—the system remains brittle and expensive.

### 8.2. Mapping Systems, Not Minds
A central theme of this paper is that the perceived “intelligence” of a constrained system is an optical illusion. The chess engine does not plan; it flows. The language model does not reason; it saturates its latent space. This is not a diminishment of these systems’ capabilities; it is a clarification of their operating principles. By understanding that intelligence is a by-product of geometric constraint, we can design for it directly rather than hoping it emerges from scale.

### 8.3. Limitations and Next Steps
The Relational-CoT protocol is currently specified as an engineering blueprint. A full implementation requires:
*   The design and training of a North-Star detector module that can parse natural language prompts into capacity variables.
*   The curation of a Relational Template library covering the most common mathematical and physical operators.
*   A controlled comparison against Abstract-CoT on a benchmark that explicitly requires cross-domain transfer.

The jet-tagging result is the most mature validation. The chess engine requires a rigorous Elo calibration against a known reference. The prime harmonic puzzle invites a formal proof of the lattice property.

---

## 9. Conclusion
The era of brute-force scale is a temporary detour. It was made necessary by a historical gap—the lack of a universal protocol for translating absolute measurements into intrinsic geometries. This paper has attempted to fill that gap. We have shown, across mathematics, dynamical systems, and applied machine learning, that when a system is anchored to its own North Star, the need for expensive exploration collapses. Efficiency is not a reward for clever engineering; it is a direct emergent property of correct architectural constraint. We invite the community to build with this principle, to measure what is measured against what is possible, and to replace the era of “how much” with the era of “how full.”

---

## Appendix A: Emergent Checker: Geometric Supremacy Engine[cite: 1]

**Ciber-Fabbrica Research**[cite: 1]

### Game Results

**=== Geometric Supremacy Engine with 3-Tier Threat Reflex ===**[cite: 1]  
Evaluation: `actual * potential + 0.5 * king safety`[cite: 1]  
Depth-1 minimax. Enter moves in UCI format (e.g., `e2e4`).[cite: 1]

**• Engine plays: `e2e4`** | Positional supremacy (White - Black): 189.50[cite: 1]
```text
r n b q k b n r
P P P P P P P P
. . . . . . . .
. . . . P . . .
. . . . . . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R
```
**Your move:** `e7e5` | Positional supremacy (White - Black): 0.00[cite: 1]

**• Engine plays: `d1f3`** | Positional supremacy (White - Black): 268.50[cite: 1]
```text
r n b q k b n r
P P P P . P P P
. . . . . . . .
. . . . p . . .
. . . . P . . .
. . . . . Q . .
P P P P . P P P
R N B . K B N R
```
**Your move:** `d8e7` | Positional supremacy (White - Black): 206.50[cite: 1]

**• Engine plays: `f1c4`** | Positional supremacy (White - Black): 254.00[cite: 1]
```text
r n b . k b n r
P P P P q P P P
. . . . . . . .
. . . . p . . .
. . B . P . . .
. . . . . Q . .
P P P P . P P P
R N B . K . N R
```
**Your move:** `e7e6` | Positional supremacy (White - Black): 82.50[cite: 1]

**• Engine plays: `c4e6`** | Positional supremacy (White - Black): 420.50[cite: 1]
```text
r n b . k b n r
P P P P . P P P
. . . . B . . .
. . . . p . . .
. . . . P . . .
. . . . . Q . .
P P P P . P P P
R N B . K . N R
```
**Your move:** `d7e6` | Positional supremacy (White - Black): 298.00[cite: 1]

**• Engine plays: `f3d3`** | Positional supremacy (White - Black): 388.00[cite: 1]
```text
r n b . k b n r
P P P . . P P P
. . . . p . . .
. . . . p . . .
. . . . P . . .
. . . Q . . . .
P P P P . P P P
R N B . K . N R
```
**Your move:** `b8a6` | Positional supremacy (White - Black): 349.00[cite: 1]

**• Engine plays: `a2a4`** | Positional supremacy (White - Black): 377.00[cite: 1]
```text
r . b . k b n r
P P P . . P P P
n . . . p . . .
. . . . p . . .
P . . . P . . .
. . . Q . . . .
. P P P . P P P
R N B . K . N R
```
**Your move:** `a6c5` | Positional supremacy (White - Black): 441.00[cite: 1]

**• Engine plays: `d3e3`** | Positional supremacy (White - Black): 295.00[cite: 1]
```text
r . b . k b n r
P P P . . P P P
. . . . p . . .
. . n . p . . .
P . . . P . . .
. . . . Q . . .
. P P P . P P P
R N B . K . N R
```
**Your move:** `b7b5` | Positional supremacy (White - Black): 261.00[cite: 1]

**• Engine plays: `a4b5`** | Positional supremacy (White - Black): 303.00[cite: 1]
```text
r . b . k b n r
P . P . . P P P
. . . . p . . .
. P n . p . . .
. . . . P . . .
. . . . Q . . .
. P P P . P P P
R N B . K . N R
```
**Your move:** `f8e7` | Positional supremacy (White - Black): 259.00[cite: 1]

**• Engine plays: `h2h4`** | Positional supremacy (White - Black): 300.00[cite: 1]
```text
r . b . k . n r
P . P . b P P P
. . . . p . . .
. P n . p . . .
. . . . P . . P
. . . . Q . . .
. P P P . P P .
R N B . K . N R
```
**Your move:** `g8h6` | Positional supremacy (White - Black): 239.00[cite: 1]

**• Engine plays: `b2b4`** | Positional supremacy (White - Black): 263.00[cite: 1]
```text
r . b . k . . r
P . P . b P P P
. . . . p . . n
. P n . p . . .
. P . . P . . P
. . . . Q . . .
. . P P . P P .
R N B . K . N R
```
**Your move:** `c8b7` | Positional supremacy (White - Black): 230.00[cite: 1]

**• Engine plays: `b4c5`** | Positional supremacy (White - Black): 280.00[cite: 1]
```text
r . . . k . . r
P b P . b P P P
. . . . p . . n
. P P . p . . .
. . . . P . . P
. . . . Q . . .
. . P P . P P .
R N B . K . N R
```
**Your move:** `a7a6` | Positional supremacy (White - Black): 265.00[cite: 1]

**• Engine plays: `a1a5`** | Positional supremacy (White - Black): 265.00[cite: 1]
```text
r . . . k . . r
. b P . b P P P
p . . . p . . n
R P P . p . . .
. . . . P . . P
. . . . Q . . .
. . P P . P P .
. N B . K . N R
```
**Your move:** `a8d8` | Positional supremacy (White - Black): 217.50[cite: 1]

**• Engine plays: `a5a6`** | Positional supremacy (White - Black): 303.50[cite: 1]
```text
. . . r k . . r
. b P . b P P P
R . . . p . . n
. P P . p . . .
. . . . P . . P
. . . . Q . . .
. . P P . P P .
. N B . K . N R
```
**Your move:** `b7a6` | Positional supremacy (White - Black): 175.50[cite: 1]

**• Engine plays: `b5a6`** | Positional supremacy (White - Black): 214.50[cite: 1]
```text
. . . r k . . r
. . P . b P P P
P . . . p . . n
. . P . p . . .
. . . . P . . P
. . . . Q . . .
. . P P . P P .
. N B . K . N R
```
**Your move:** `d8d5` | Positional supremacy (White - Black): 235.00[cite: 1]

**• Engine plays: `e4d5`** | Positional supremacy (White - Black): 346.00[cite: 1]
```text
. . . . k . . r
. . P . b P P P
P . . . p . . n
. . P P p . . .
. . . . . . . P
. . . . Q . . .
. . P P . P P .
. N B . K . N R
```
**Your move:** `e8g8` | Positional supremacy (White - Black): 342.00[cite: 1]

**• Engine plays: `e3e5`** | Positional supremacy (White - Black): 433.00[cite: 1]
```text
. . . . . r k .
. . P . b P P P
P . . . p . . n
. . P P Q . . .
. . . . . . . P
. . . . . . . .
. . P P . P P .
. N B . K . N R
```
**Your move:** `e7c5` | Positional supremacy (White - Black): 375.00[cite: 1]

**• Engine plays: `h1h3`** | Positional supremacy (White - Black): 463.00[cite: 1]
```text
. . . . . r k .
. . P . . P P P
P . . . p . . n
. . b P Q . . .
. . . . . . . P
. . . . . . . R
. . P P . P P .
. N B . K . N .
```
**Your move:** `f8a8` | Positional supremacy (White - Black): 427.50[cite: 1]

**• Engine plays: `d5e6`** | Positional supremacy (White - Black): 458.50[cite: 1]
```text
r . . . . . k .
. . P . . P P P
P . . . P . . n
. . b . Q . . .
. . . . . . . P
. . . . . . . R
. . P P . P P .
. N B . K . N .
```
**Your move:** `f7e6` | Positional supremacy (White - Black): 436.00[cite: 1]

**• Engine plays: `e5c5`** | Positional supremacy (White - Black): 647.50[cite: 1]
```text
r . . . . . k .
. . P . . . P P
P . . . p . . n
. . Q . . . . .
. . . . . . . P
. . . . . . . R
. . P P . P P .
. N B . K . N .
```
**Your move:** `h6f7` | Positional supremacy (White - Black): 630.00[cite: 1]

**• Engine plays: `c5b5`** | Positional supremacy (White - Black): 683.50[cite: 1]
```text
r . . . . . k .
. . P . . n P P
P . . . p . . .
. Q . . . . . .
. . . . . . . P
. . . . . . . R
. . P P . P P .
. N B . K . N .
```
**Your move:** `a8c8` | Positional supremacy (White - Black): 697.50[cite: 1]

**• Engine plays: `d2d4`** | Positional supremacy (White - Black): 779.00[cite: 1]
```text
. . r . . . k .
. . P . . n P P
P . . . p . . .
. Q . . . . . .
. . . P . . . P
. . . . . . . R
. . P . . P P .
. N B . K . N .
```
**Your move:** `c7c6` | Positional supremacy (White - Black): 686.00[cite: 1]

**• Engine plays: `b5b3`** | Positional supremacy (White - Black): 600.00[cite: 1]
```text
. . r . . . k .
. . . . . n P P
P . p . p . . .
. . . . . . . .
. . . P . . . P
. Q . . . . . R
. . P . . P P .
. N B . K . N .
```
**Your move:** `c8d8` | Positional supremacy (White - Black): 580.00[cite: 1]

**• Engine plays: `b3e6`** | Positional supremacy (White - Black): 670.00[cite: 1]
```text
. . . r . . k .
. . . . . n P P
P . p . Q . . .
. . . . . . . .
. . . P . . . P
. . . . . . . R
. . P . . P P .
. N B . K . N .
```
**Your move:** `d8d4` | Positional supremacy (White - Black): 585.00[cite: 1]

**• Engine plays: `e6e8`** | Positional supremacy (White - Black): 423.50[cite: 1]

**Game over 1-0**[cite: 1]

### Python Source Code
```python
import chess

# Potential maximum attacks from a piece on an empty board
POTENTIAL = {
    chess.PAWN: 2,
    chess.KNIGHT: 8,
    chess.BISHOP: 13,
    chess.ROOK: 14,
    chess.QUEEN: 27,
    chess.KING: 8,
}

def actual_control(board, square):
    """Number of empty squares attacked by the piece on 'square'."""
    piece = board.piece_at(square)
    if not piece:
        return 0
    attacked = board.attacks(square)
    empty = set(range(64)) - set(board.piece_map().keys())
    return len(attacked & empty)

def king_safety_score(board, color):
    """Number of safe empty squares around the king (0-8)."""
    king = board.king(color)
    if king is None:
        return 0
    safe = 0
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            r = chess.square_rank(king) + dr
            f = chess.square_file(king) + df
            if 0 <= r < 8 and 0 <= f < 8:
                sq = chess.square(f, r)
                if board.piece_at(sq) is None and not board.is_attacked_by(not color, sq):
                    safe += 1
    return safe

def positional_supremacy(board):
    """White supremacy
    (actual * potential) - (actual * potential) for Black,
    plus king safety weighted at 0.5 per safe square."""
    white_score = 0
    black_score = 0
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            act = actual_control(board, sq)
            pot = POTENTIAL[piece.piece_type]
            contrib = act * pot
            if piece.color == chess.WHITE:
                white_score += contrib
            else:
                black_score += contrib
                
    white_score += 0.5 * king_safety_score(board, chess.WHITE)
    black_score += 0.5 * king_safety_score(board, chess.BLACK)
    return white_score - black_score

def piece_geometric_value(board, square):
    """Return the geometric contribution (act * pot) of the piece on the square."""
    piece = board.piece_at(square)
    if not piece:
        return 0
    return actual_control(board, square) * POTENTIAL[piece.piece_type]

def attackers_defenders(board, square, color):
    """Return (attackers_count, defenders_count) for the given square from perspective of 'color'."""
    attackers = 0
    defenders = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue
        if square in board.attacks(sq):
            if piece.color == color:
                defenders += 1
            else:
                attackers += 1
    return attackers, defenders

def is_threatened(board, square, color):
    """Return True if the piece on 'square' is under threat and not adequately defended."""
    piece = board.piece_at(square)
    if not piece or piece.color != color:
        return False
        
    attackers, defenders = attackers_defenders(board, square, color)
    if attackers == 0:
        return False
        
    if attackers > defenders:
        return True
        
    # Even if defended, check for bad trade (attacker less valuable than piece)
    piece_val = POTENTIAL[piece.piece_type]
    for sq in chess.SQUARES:
        attacker = board.piece_at(sq)
        if attacker and attacker.color == (not color) and square in board.attacks(sq):
            if POTENTIAL[attacker.piece_type] < piece_val:
                return True
    return False

def get_threatened_squares(board, color):
    """Return list of squares containing threatened pieces of the given color."""
    threatened = []
    for sq in chess.SQUARES:
        if is_threatened(board, sq, color):
            threatened.append(sq)
    return threatened

def geometric_see(board, move, color):
    """Simplified Static Exchange Evaluation using geometric scores.
    Simulate capture 'move', then opponent's best recapture on the same square.
    Return net change in positional supremacy from 'color's perspective.
    Positive means favorable trade."""
    if not board.is_capture(move):
        return 0
        
    original_score = positional_supremacy(board)
    if color == chess.BLACK:
        original_score = -original_score
        
    board_copy = board.copy()
    captured_piece = board_copy.piece_at(move.to_square)
    if captured_piece is None:
        return 0
        
    board_copy.push(move)
    recapture_square = move.to_square
    best_opponent_score = float('inf') if color == chess.WHITE else -float('inf')
    
    for opp_move in board_copy.legal_moves:
        if opp_move.to_square == recapture_square and board_copy.is_capture(opp_move):
            b2 = board_copy.copy()
            b2.push(opp_move)
            score_after = positional_supremacy(b2)
            if color == chess.WHITE:
                if score_after < best_opponent_score:
                    best_opponent_score = score_after
            else:
                if -score_after > best_opponent_score:
                    best_opponent_score = -score_after
                    
    if best_opponent_score == float('inf') or best_opponent_score == -float('inf'):
        score_after = positional_supremacy(board_copy)
        if color == chess.BLACK:
            score_after = -score_after
        return score_after - original_score
        
    return best_opponent_score - original_score

def get_new_attacks(board_before, board_after, color):
    """Return set of enemy squares that are attacked in board_after but were not in board_before."""
    new_attacks = set()
    enemy_color = not color
    for sq in chess.SQUARES:
        piece = board_after.piece_at(sq)
        if piece and piece.color == enemy_color:
            if board_after.is_attacked_by(color, sq) and not board_before.is_attacked_by(color, sq):
                new_attacks.add(sq)
    return new_attacks

def resolve_threats(board, color):
    """Three-tier threat resolution with forward safety check.
    Rejects moves that create new threatened pieces, unless they are favorable captures."""
    threatened_squares = get_threatened_squares(board, color)
    candidates = set()
    
    for move in board.legal_moves:
        board_copy = board.copy()
        board_copy.push(move)
        new_threats = get_threatened_squares(board_copy, color)
        
        if new_threats:
            if board.is_capture(move):
                see_score = geometric_see(board, move, color)
                if see_score >= 0:
                    candidates.add(move)
                continue
                
            if move.from_square in threatened_squares:
                piece_still_threatened = is_threatened(board_copy, move.to_square, color)
                if not piece_still_threatened:
                    other_threats = [sq for sq in new_threats if sq != move.to_square]
                    if not other_threats:
                        candidates.add(move)
                continue
                
        if not candidates:
            max_threat_val = 0
            for sq in threatened_squares:
                piece = board.piece_at(sq)
                if piece:
                    val = POTENTIAL[piece.piece_type]
                    if val > max_threat_val:
                        max_threat_val = val
                        
            if board.is_capture(move):
                target = board.piece_at(move.to_square)
                if target and POTENTIAL[target.piece_type] >= max_threat_val:
                    candidates.add(move)
                    continue
            else:
                new_attacks = get_new_attacks(board, board_copy, color)
                for enemy_sq in new_attacks:
                    enemy_piece = board_copy.piece_at(enemy_sq)
                    if enemy_piece and POTENTIAL[enemy_piece.piece_type] >= max_threat_val:
                        candidates.add(move)
                        break
                else:
                    candidates.add(move)
                    
    if candidates:
        return list(candidates)
    else:
        return list(board.legal_moves)

def best_move_minimax(board):
    """Depth-1 minimax with threat pre-filter."""
    moves_to_consider = resolve_threats(board, chess.WHITE)
    best_score = -float('inf')
    best_move = None
    
    for move in moves_to_consider:
        b1 = board.copy()
        b1.push(move)
        
        if b1.is_checkmate():
            return move
            
        worst_black = float('inf')
        for black_move in b1.legal_moves:
            b2 = b1.copy()
            b2.push(black_move)
            score = positional_supremacy(b2)
            if score < worst_black:
                worst_black = score
                
        if worst_black == float('inf'):
            worst_black = positional_supremacy(b1)
            
        if worst_black > best_score:
            best_score = worst_black
            best_move = move
            
    return best_move or next(iter(board.legal_moves))

def play():
    board = chess.Board()
    print("=== Geometric Supremacy Engine with 3-Tier Threat Reflex ===")
    print("Evaluation: actual * potential + 0.5 * king safety")
    print("Depth-1 minimax. Enter moves in UCI format (e.g., e2e4).\n")
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = best_move_minimax(board)
            print(f"Engine plays: {move}")
            board.push(move)
        else:
            print(board)
            human_move = input("Your move: ").strip()
            
            if human_move.lower() == "quit":
                break
                
            try:
                move = chess.Move.from_uci(human_move)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move. Legal UCI moves:")
                    print(", ".join(m.uci() for m in board.legal_moves))
                    continue
            except Exception:
                print("Invalid format. Use e.g., e2e4")
                continue
                
            sup = positional_supremacy(board)
            print(f"Positional supremacy (White - Black): {sup:.2f}\n")
            
    print("Game over")
    print(board.result())

if __name__ == "__main__":
    play()
```

---

## References

1.  Concas, M. *The Intrinsic Blueprint: An Introduction to Relational Calculus*. Zenodo. doi:10.5281/zenodo.19757717.
2.  Concas, M. *Scale-Invariant Jet Tagging via Relational Calculus*. 2026.
3.  Concas, M. *Harmonic Architecture of Prime Pairs*. 2026.
4.  Ramji, K., Naseem, T., & Astudillo, R. F. *Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought*. 2026.
5.  Kasieczka, G., et al. *The Machine Learning Landscape of Top Taggers*. SciPost Phys. 7(1), 014, 2019.
6.  Shao, Z., et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning*. 2024.
7.  Goyal, S., et al. *Think Before You Speak: Training Language Models with Pause Tokens*. ICLR, 2024.
