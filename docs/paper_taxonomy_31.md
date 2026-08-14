# Taxonomy of the Manually Included Transformer Verification Papers

## 1. Scope and reading protocol

This report classifies the 31 records in `manual_includes_cumulative_round1.csv` for the purpose of structuring a Transformer verification survey. The classification is based on each paper's research target, verified property, threat or specification, technical method, guarantee style, and relationship to the survey scope. Existing `research_track` labels were not reused because several are too coarse or incorrect for survey writing.

All 31 records were checked individually against their titles, abstracts, metadata, and stated contributions. The records without a stored abstract were checked against their official paper pages or full text. This is a structural survey classification; a later evidence-extraction pass should still record theorem assumptions, evaluated architectures, datasets, baselines, and quantitative results from the full papers.

## 2. Important bibliographic correction

The current audit file contains **31 records but only 30 unique papers**.

- `Robustness Verification for Attention Networks using Mixed Integer Programming` (2022 CoRR record)
- `Are Transformers More Robust? Towards Exact Robustness Verification for Transformers` (SAFECOMP 2023)

These are two versions of the same work, both associated with arXiv `2202.03932`. The second title is used as the canonical survey entry. The source audit files are intentionally left unchanged for provenance; the categorized CSV marks the first record as `duplicate_alias`.

## 3. Recommended top-level taxonomy

| Code | Primary category | Unique papers | Role in the survey |
|---|---|---:|---|
| A | Transformer core verification algorithms | 9 | Main technical chapter |
| B | Vision Transformer certification and certified defense | 5 | Main application chapter |
| C | Language and LLM input-robustness certification | 3 | Main application chapter |
| D | LLM behavior and semantic-property certification | 5 | Main emerging-properties chapter |
| E | Verification of LLM outputs and composed systems | 2 | Boundary chapter |
| F | Provable repair | 1 | Verification-derived downstream technique |
| G | Formal interpretability | 2 | Related downstream technique |
| H | Methodology, benchmarks, and toolchains | 2 | Survey methodology and evaluation chapter |
| I | Formal analysis of Transformer properties | 1 | Boundary or future-directions chapter |
|  | **Total** | **30** | 31 source records after counting one duplicate alias |

For writing and prioritization, these categories can be compressed into three scope tiers:

| Scope tier | Categories | Papers | Recommendation |
|---|---|---:|---|
| Core corpus | A, B, C, D, F | 23 | Read and compare in depth |
| Closely related | G, H | 4 | Retain as methodology or verification-derived work |
| Boundary work | E, I | 3 | Discuss separately; do not use to characterize model-level Transformer verification |

## 4. Category A: Transformer core verification algorithms

These papers contribute a verifier, an abstract domain, a relaxation, an exact encoding, or verification infrastructure that directly handles Transformer operations.

| Year | Paper | Main contribution | Guarantee interpretation |
|---:|---|---|---|
| 2020 | Robustness Verification for Transformers | First Transformer robustness verifier; develops bounds for self-attention cross-nonlinearity and cross-position dependence | Sound, deterministic, incomplete |
| 2021 | Fast and Precise Certification of Transformers | DeepT and Multi-norm Zonotopes for dot products, softmax, norm perturbations, and synonym substitutions | Sound abstract interpretation, incomplete |
| 2022 | Faith | GPU execution framework using semantic graph transformation, kernel fusion, specialized kernels, and autotuning | Accelerates an existing verifier and inherits its guarantee |
| 2023 | Are Transformers More Robust? | Encodes Sparsemax Transformers and maximum robustness as MIQCP with preprocessing heuristics | Exact or complete for the encoded model and property |
| 2024 | GaLileo | N-dimensional linear relaxation of softmax that retains dependencies among softmax inputs | Sound linear-relaxation certificate, incomplete |
| 2024 | A One-Layer Decoder-Only Transformer is a Two-Layer RNN | Reduces a decoder-only Transformer to an RNN to certify arbitrary, including length-changing, perturbation spaces | Sound certification through structural reduction |
| 2025 | Spy Inside | Verifies event-time-series Transformers using sampling, linear programming, and the extreme value theorem | Formal output bounds under its framework assumptions |
| 2026 | Parameterized Abstract Interpretation for Transformer Verification | Parameterized quadratic and affine abstract domains for self-attention inner products | Sound abstract interpretation, incomplete |
| 2026 | Vertex-Softmax | Exact score-box optimization for softmax via a vertex and threshold theorem, integrated into a CROWN-style verifier | Exact softmax primitive; the end-to-end verifier is still generally incomplete |

This category provides the strongest chronological backbone for the survey: early attention bound propagation, abstract interpretation, exact optimization, systems acceleration, tighter softmax and inner-product abstractions, and architectural reduction.

## 5. Category B: Vision Transformer certification and certified defense

These papers focus on certifying a vision task or constructing a ViT-based certified defense rather than proposing a general-purpose Transformer verifier.

| Year | Paper | Property or threat | Main method |
|---:|---|---|---|
| 2022 | Certified Patch Robustness via Smoothed Vision Transformers | One bounded contiguous adversarial patch | Derandomized smoothing, image ablation, and efficient token dropping |
| 2022 | ViP | Single- and dual-patch attacks; certified detection and recovery | ViT and masked-autoencoder-based detection and recovery |
| 2023 | CertViT | Norm-bounded adversarial robustness of pretrained ViTs | Proximal reduction of Lipschitz bounds plus accuracy-preserving projection |
| 2023 | PatchCensor | Natural, distribution-shift, or adversarial patches | Exhaustive masked-attention inference and voting at test time |
| 2024 | STR-Cert | Adversarial robustness of scene-text recognition pipelines | DeepPoly extensions and new polyhedral bounds for STR components |

The five papers should be compared along three independent axes: patch versus norm-bounded perturbations, classifier versus image-to-sequence tasks, and verification of a fixed model versus construction of a certifiably robust inference pipeline.

## 6. Category C: Language and LLM input-robustness certification

These papers retain the classical robustness question, but scale it to LLMs, universal text perturbations, or multimodal feature spaces.

| Year | Paper | Property or threat | Main method |
|---:|---|---|---|
| 2023 | Certified Robustness for Large Language Models with Self-Denoising | Prediction stability under noisy text inputs | Randomized smoothing with LLM self-denoising |
| 2024 | CR-UTP | Universal and input-specific text perturbations | Randomized smoothing, prompt search, and prompt ensembles |
| 2026 | Feature-Space Adversarial Robustness Certification for Multimodal Large Language Models | l2-bounded feature distortion in MLLMs | Feature-space Gaussian smoothing and a Gaussian Smoothness Booster |

All three provide probabilistic certificates and should not be mixed with deterministic bound-propagation verifiers without explicitly separating confidence level, sampling assumptions, and certified event.

## 7. Category D: LLM behavior and semantic-property certification

This category shows the field moving beyond local perturbation robustness toward properties that are specific to generative models.

| Year | Paper | Certified property | Guarantee and mechanism |
|---:|---|---|---|
| 2024 | QuaCer-C | Knowledge comprehension under a distribution of naturally noisy prompts | High-confidence probability bounds over knowledge-graph-derived specifications |
| 2025 | Shh, Don't Say That! | Domain adherence and out-of-domain generation | VALID gives adversarial probability bounds using a guide model |
| 2025 | Can AI Keep a Secret? | Contextual integrity and resistance to prompt-injection information flow | Deterministic token-level non-interference using provenance labels, a trust lattice, and hard attention masks |
| 2025 | Selective Risk Certification for LLM Outputs | Output risk under an abstention policy | Information-lift statistics and sub-gamma PAC-Bayes bounds |
| 2026 | Improving LLM Domain Certification with Pretrained Guide Models | Domain adherence and refusal behavior | PRISM improves VALID using pretrained, contrastively tuned guide models |

`Shh, Don't Say That!` and `Improving LLM Domain Certification with Pretrained Guide Models` form a direct method lineage and should be discussed together. `Can AI Keep a Secret?` is methodologically different because it enforces a deterministic architectural information-flow property rather than estimating a probability bound.

## 8. Category E: Verification of LLM outputs and composed systems

These papers are valuable, but the verified object is not the Transformer network in isolation.

| Year | Paper | Actual verified object | Scope warning |
|---:|---|---|---|
| 2025 | Safe LLM-Controlled Robots with Formal Guarantees via Reachability Analysis | Reachable states of the composed LLM-robot closed loop | System-level safety, not a direct property of Transformer internals |
| 2025 | Step-Wise Formal Verification for LLM-Based Mathematical Problem Solving | Individual steps in an LLM-generated mathematical solution | Post-hoc output checking; an LLM also participates in formalization |

The second paper is especially important for the survey's exclusion rule. It verifies LLM output, but also uses an LLM as part of the verification pipeline. It should remain a clearly marked boundary case rather than evidence about architecture-level LLM verification.

## 9. Category F: Provable repair

| Year | Paper | Repair object | Main guarantee and method |
|---:|---|---|---|
| 2024 | Provable Repair of Vision Transformers | Incorrect ViT classifications on a repair set | PRoViT edits the last layer through fine-tuning and a scalable LP formulation; returned repairs are sound, architecture-preserving, and guaranteed correct on the repair set, but the repair procedure is incomplete |

This paper deserves its own short section because repair changes the synthesis problem: verification asks whether a property holds, while provable repair searches for parameters that make the property hold and minimizes drawdown.

## 10. Category G: Formal interpretability

| Year | Paper | Explanation object | Verification contribution |
|---:|---|---|---|
| 2023 | Towards Formal XAI | Approximately minimal feature explanations for generic DNNs | Verification-guided search plus lower and upper bounds on explanation optimality |
| 2026 | Formal Mechanistic Interpretability | Internal circuits responsible for model behavior | Verifier-backed circuit discovery with domain robustness, robust patching, and minimality guarantees |

The first paper is architecture-agnostic and the second is closer to modern circuit analysis. Both are best presented as verification-enabled interpretability rather than as Transformer safety verification itself.

## 11. Category H: Methodology, benchmarks, and toolchains

| Year | Paper | Problem addressed | Contribution |
|---:|---|---|---|
| 2023 | ANTONIO | Existing numerical verifiers do not directly accept realistic NLP tasks | Generates and preprocesses NLP benchmarks for ERAN and Marabou |
| 2024 | NLP Verification: Towards a General Methodology for Certifying Robustness | Geometric perturbation sets may not preserve sentence semantics | Defines the embedding gap and proposes a more systematic NLP training-verification methodology |

These two papers should appear early in the survey because they explain what a valid NLP verification specification is and how experiments should be constructed. They do not merely belong in related work.

## 12. Category I: Formal analysis of Transformer properties

| Year | Paper | Formal object | Scope interpretation |
|---:|---|---|---|
| 2026 | Structural Sensitivity in Compressed Transformers | Layer-wise compression-error propagation and per-matrix norm inequalities | Lean 4 checks analytic bounds, but the work does not verify an end-to-end behavioral specification of a Transformer |

This is useful for a future-directions discussion on machine-checked Transformer analysis, but should not be counted as a core model-verification algorithm without a broader definition of verification.

## 13. Cross-cutting guarantee taxonomy

The papers should also be compared by guarantee style; otherwise deterministic verification and statistical certification will be conflated.

| Guarantee family | Representative papers | What is guaranteed |
|---|---|---|
| Sound, deterministic, incomplete bounds | Robustness Verification for Transformers; DeepT; GaLileo; Parameterized Abstract Interpretation | If the verifier succeeds, the property holds over the entire specified region; failure may be inconclusive |
| Exact or complete optimization | Are Transformers More Robust? | Exact maximum robustness for the encoded Sparsemax model and property |
| Exact local primitive inside an incomplete verifier | Vertex-Softmax | Exact softmax objective over score intervals, not exact end-to-end network verification |
| Certified defense by exhaustive or structural coverage | Smoothed ViT; PatchCensor; ViP | Robust prediction, detection, or recovery under a patch threat model |
| High-confidence statistical certificate | Self-Denoising; CR-UTP; QuaCer-C; Selective Risk; Feature-Space Smoothing | A probability, risk, or robust event under sampling and distributional assumptions |
| Architectural enforcement | Contextual Integrity Verification | Deterministic non-interference under the modified architecture and threat model |
| External formal checking of outputs or systems | LLM-controlled robots; MATH-VF | Reachability safety or correctness of a generated artifact, not a model-wide property |
| Machine-checked analytic theorem | Structural Sensitivity in Compressed Transformers | Formal mathematical inequalities about error propagation |

## 14. Recommended survey structure

1. Scope, terminology, and the distinction among verification, certification, certified defense, repair, and formal analysis.
2. Transformer-specific verification challenges: dot products, softmax, cross-position dependence, sequence length, and discrete semantics.
3. Core verification algorithms, organized chronologically and by solver family.
4. Specification validity, the NLP embedding gap, benchmarks, and toolchains.
5. Certified robustness by modality: NLP, vision, and multimodal models.
6. LLM-specific semantic and security properties: knowledge, domain adherence, non-interference, and selective risk.
7. Verification-derived techniques: provable repair and formal interpretability.
8. Output-level and system-level verification as a clearly separated boundary.
9. Open problems: scalability to full-size LLMs, meaningful specifications, deterministic versus statistical guarantees, sequence generation, and reproducible evaluation.

This structure gives the survey a coherent progression from **operator-level bounds**, through **model-level robustness**, to **LLM-specific behavioral properties** and finally **verification-derived downstream techniques**.

## 15. Machine-readable result

The complete per-record classification is stored in:

`data/runs/latest/processed/manual_includes_categorized_round1.csv`

It preserves all 31 source records and adds the following dimensions:

- primary category;
- verification target;
- verified property or threat;
- method family;
- guarantee style;
- survey role;
- canonical title and duplicate status;
- a short classification rationale.
