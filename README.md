# Berna R7 - Root Model

**A 1.06B multilingual language model with a modular cell-based architecture.**

Berna R7 implements the *DNA-Kernel Plexus* architecture from
[Berna R5](https://zenodo.org/records/23015308) (DOI: 10.5281/zenodo.23015308).

- Model: 1.06B parameters, trained from scratch
- Data: 2.66B tokens (English + Math + Code)
- Hardware: 1x RTX 5090 (32 GB), ~6.5 days
- License: Berna Research v1.0

## Status

| Component | Status |
|-----------|--------|
| Tokenizer (32k BPE) | complete |
| Data pipeline | complete (2.66B tokens) |
| Root model training | in progress (~5 October 2026) |
| Cell system (R5) | implemented |
| H1 experiment (splitting) | after Root |
| H2 experiment (forgetting) | after Root |
| Baseline comparison | after Root |
| Weights on HuggingFace | after training |

## Architecture

Berna R7 implements the six-layer structure from R5:

Input -> Sensors -> Spinal Cord -> Plexus -> Knowledge Cells -> DNA Kernel -> Registry

- Knowledge Cells: 112 cells (4 per MLP layer x 28 layers).
- 6D Knowledge Vector: K = [L, W, H, D, T, E], S = ||K||_omega.
- DNA Kernel: 100 chromosomes x 4 genes.
- Plexus: dynamic graph from co-activations.
- Registry: SQL catalog of models and cells.

See ARCHITECTURE.md and DNA.md for details.

## Project Structure

```
berna-r7/
  ARCHITECTURE.md      system design
  DNA.md               DNA Kernel spec
  README.md            this file
  LICENSE              Berna Research v1.0
  code/
    cells/             cell lifecycle (R5 implementation)
    baselines/         vanilla, EWC, PackNet
    train.py           training entry point
    dataset.py         data loader
    eval.py            perplexity
    h1_saturation_experiment.py
    h2_forgetting_experiment.py
  registry/
    schema.sql         SQL schema
    rules.json         router rules
  paper/
    outline.md         planned paper
```

## Compliance with R5

| R5 Component | R7 Status |
|---|---|
| Registry + UUIDs | implemented |
| connection_token | implemented |
| Federation | implemented |
| from-scratch training | yes |
| Knowledge Cells | 112 cells |
| 6D K vector | K1, K3, K5, K6 |
| DNA Kernel | 100x4 |
| S* = 0.85 | yes |
| Splitting (H1) | tested |
| Plexus | implemented |
| Router (Spinal Cord) | deferred |
| Bounded Forgetting (H2) | not yet tested |

## Citation

If you use Berna R7 in academic work:

```bibtex
@software{berna_r7_2026,
  author    = {Muhammed, Mohammed Kamil},
  title     = {Berna R7: A 1.06B Multilingual Language Model},
  year      = {2026},
  publisher = {Berna Labs},
  url       = {https://github.com/Berna-Labs/berna-r7}
}
```

For the underlying theory (R5):

```bibtex
@software{berna_r5_2026,
  author    = {Muhammed, Mohammed Kamil},
  title     = {DNA-Kernel Plexus},
  year      = {2026},
  doi       = {10.5281/zenodo.23015308}
}
```

## License

Berna Research License v1.0 - see LICENSE.

Research use is free. Commercial use requires a separate license.
Contact: info@bernalabs.com
