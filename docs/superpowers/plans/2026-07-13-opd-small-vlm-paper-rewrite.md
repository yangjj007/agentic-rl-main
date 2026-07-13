# OPD for Small VLM Reasoning Paper Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the Chinese abstract, introduction, and related work around the defensible contribution of systematically introducing OPD into sub-1B verifiable vision-language reasoning and demonstrating its complementarity with GRPO and fallback supervision.

**Architecture:** The paper will use a problem-first narrative modeled after DyME and VAR: establish why sparse RLVR and off-policy imitation both fail for sub-1B VLMs, introduce OPD as dense feedback on student-generated states, then explain verifier-confirmed three-route training and realized-route feedback as the mechanisms that make OPD usable. The wording will distinguish teacher-input gold access from routing-verifier reference access and will avoid unsupported “first VLM OPD” claims.

**Tech Stack:** Markdown paper draft, BibTeX seed library, arXiv primary-source metadata, repository training/eval artifacts.

---

### Task 1: Lock the Core Claim

**Files:**
- Modify: `docs/paper_reconstruction/claim_evidence_matrix.md`
- Modify: `docs/paper_reconstruction/related_paper_figure_review.md`

- [x] Record the allowed first claim as “first systematic study of OPD for sub-1B VLM RLVR,” with status `partial` until the literature audit and experiments support it.
- [x] Explicitly prohibit “first VLM OPD” because VOLD, Decomposed OPD, and VA-OPD precede this work.
- [x] Define the three evidence requirements: OPD-vs-no-OPD effectiveness, OPD/GRPO/SFT complementarity, and sub-1B-specific failure analysis.
- [x] Run `rg -n "first VLM OPD|首个 VLM OPD|sub-1B|互补" docs/paper_reconstruction` and verify no unsupported first claim remains.

### Task 2: Rewrite the Abstract

**Files:**
- Modify: `docs/paper_reconstruction/chinese_draft.md`

- [x] Replace the current implementation-heavy abstract with five moves: application need, sub-1B training failure, OPD insight, proposed framework, evidence status.
- [x] Keep EMA, all-reduce, route caps, environment names, and diagnostic rates out of the abstract.
- [x] State that the teacher does not see the answer in the gold-hidden setting while the RLVR verifier may use the reference.
- [x] Do not report unfinished CLRC accuracy or claim `>0.60`.

### Task 3: Rewrite the Introduction

**Files:**
- Modify: `docs/paper_reconstruction/chinese_draft.md`

- [x] Build paragraph 1 around the practical value and difficulty of reasoning in sub-1B VLMs.
- [x] Build paragraph 2 around the SFT/RLVR dilemma, citing multimodal CoT, SFT-or-RL, Visual-RFT, VLM-R1, and DyME.
- [x] Build paragraph 3 around the missing dense signal on student-generated error states and introduce OPD.
- [x] Build paragraph 4 around why generic OPD is insufficient: teacher reliability, sparse task success, and changing student autonomy.
- [x] Present the method in one reproducible sentence before introducing module names.
- [x] End with three contributions: sub-1B VLM OPD study, verifier-confirmed three-route complementarity, realized-route adaptive support and experiments.

### Task 4: Expand Related Work

**Files:**
- Modify: `docs/paper_reconstruction/chinese_draft.md`
- Modify: `docs/paper_reconstruction/references_seed.bib`

- [x] Expand Small VLM and multimodal CoT coverage with LLaVA, TinyLLaVA, MobileVLM, SmolVLM, Multimodal-CoT, LLaVA-CoT, Insight-V, Mulberry, and SFT-or-RL.
- [x] Expand VLM RLVR coverage with DeepSeekMath/GRPO background, DeepSeek-R1, Visual-RFT, VLM-R1, Reason-RFT, LMM-R1, MM-Eureka, Vision-R1, R1-VL, and OpenVLThinker.
- [x] Expand SFT/RL hybrid coverage with DyME, LUFFY, CHORD, SRFT, and KDRL.
- [x] Expand KD/OPD coverage with KD, sequence KD, MiniLLM, GKD, VOLD, Decomposed OPD, VA-OPD, IW-OPD, RG-OPD, DOPD, TA-OPD, PW-OPSD, SFD, and GateKD.
- [x] Expand curriculum coverage with TSCL, ALP-GMM, Auto-CEI, and self-evolving reasoning curriculum where relevant.
- [x] Ensure each paragraph states what prior work solved and the exact remaining gap; avoid bibliography dumping.

### Task 5: Audit Readability and Claims

**Files:**
- Verify: `docs/paper_reconstruction/chinese_draft.md`
- Verify: `docs/paper_reconstruction/claim_evidence_matrix.md`
- Verify: `docs/paper_reconstruction/references_seed.bib`

- [x] Count unique citation keys used in the draft and require at least 35. (`65` unique keys.)
- [x] Verify every cited key exists in `references_seed.bib`.
- [x] Search for forbidden claims: `first VLM OPD`, `完全无 gold`, `已超过 60`, and unqualified `no-gold`.
- [x] Check that the abstract can be summarized without implementation terminology in one sentence.
- [x] Check that Introduction presents the method before any controller implementation details.
- [x] Check that all unfinished effectiveness claims remain marked running or missing.
