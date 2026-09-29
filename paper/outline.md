# Berna R7 - Paper Outline

**Status:** draft
**Target:** arXiv / Zenodo
**Companion:** Berna R5 (DOI: 10.5281/zenodo.23015308)

## Title
Berna R7: A 1.06B Multilingual Language Model with a Modular Cell-Based Architecture

## Abstract (200 words target)
- LLMs are static; catastrophic forgetting; no localized correction.
- R5 proposed the DNA-Kernel Plexus framework (theory).
- R7 is the first 1.06B implementation, from scratch on one RTX 5090.
- Two hypotheses: H1 (saturation-triggered splitting), H2 (bounded forgetting).
- Baselines: vanilla, EWC, PackNet.
- Code + weights open.

## Sections
1. Introduction (1.5 pages)
2. Related Work (1 page)
3. Architecture (2 pages)
4. Experiments (3 pages)
   - Setup
   - H1: Saturation dynamics
   - H2: Bounded forgetting
   - Baselines
   - Ablations
5. Discussion (1 page)
6. Limitations (1 page)
7. Conclusion

## Appendices
- A. Reproducibility
- B. R5 compliance table
- C. Full experiment logs

## What We Do NOT Claim
- SOTA on any benchmark.
- Superiority over GPT-4-class models.
- Scalability beyond 1.5B.
- Theoretical completeness.

## Timeline
- Mother training: ~5 October 2026
- H1 + H2 experiments: ~10 October 2026
- Baselines: ~12 October 2026
- Draft complete: ~17 October 2026
- Submission: ~18 October 2026
