# PowerInfer: Consumer-GPU LLM Serving (UCF project fork)

This repository is a fork of **[Tiiny-AI/PowerInfer](https://github.com/Tiiny-AI/PowerInfer)** for our **UCF CAP 6614** project work. For full upstream build instructions, model support, and engine documentation, see **[`README-PowerInfer.md`](README-PowerInfer.md)** in this repo, and the original projects linked below.

## Project team

| Name | Email |
| --- | --- |
| Cuong Dang | [cuong.dang@ucf.edu](mailto:cuong.dang@ucf.edu) |
| Hai Nguyen | [hai.nguyen@ucf.edu](mailto:hai.nguyen@ucf.edu) |
| Joshua Lowe | [jo201760@ucf.edu](mailto:jo201760@ucf.edu) |
| Lam Nguyen | [lam.nguyen@ucf.edu](mailto:lam.nguyen@ucf.edu) |

## System paper: PowerInfer

| | |
| --- | --- |
| **Title** | PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU |
| **Authors** | Xue, Y., *et al.* (SJTU IPADS) |
| **Venue** | SOSP 2024 |

**Short description:** PowerInfer exploits *locality* in LLM inference: for any given input, only a small set of neurons are *hot* (frequently activated). The system pre-loads hot neurons on the GPU while computing cold neurons on the CPU, reducing memory pressure and data movement. Reported results include up to about **11×** speedup over [llama.cpp](https://github.com/ggerganov/llama.cpp) on a single **NVIDIA RTX 4090** for large models (e.g. **175B**-class).

**Upstream code (reference):** [github.com/SJTU-IPADS/PowerInfer](https://github.com/SJTU-IPADS/PowerInfer)

## Our project angle

We deploy **PowerInfer** on a **consumer GPU** with an **OPT** or **LLaMA**-family model, then:

1. **Profile** the hot/cold neuron distribution across layers and runs.
2. **Compare** end-to-end inference speed against **llama.cpp** and **vLLM** under controlled settings.
3. **Analyze** how the hot-neuron set varies with **different input types** (e.g. topic, length, or task format).

Project artifacts (scripts, plots, and result tables) live under [`Reports/`](Reports/) in this repository.

## Links

| Resource | URL |
| --- | --- |
| This fork (base) | [Tiiny-AI/PowerInfer](https://github.com/Tiiny-AI/PowerInfer) |
| Original PowerInfer (SJTU) | [SJTU-IPADS/PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) |

## License and attribution

This project inherits the upstream project’s terms; see the license files in the tree and `README-PowerInfer.md` for full notices. When you cite the system in academic work, use the **SOSP 2024** PowerInfer paper and the upstream repository as appropriate to your style guide.
