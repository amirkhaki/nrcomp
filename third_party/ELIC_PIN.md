# Pinned ELIC Reimplementation Submodule

- **Upstream Repository:** https://github.com/VincentChandelier/ELiC-ReImplemetation
- **Pinned Commit SHA:** `92a9ece1e6e1a188a12dfd7a58f9b51c554f9f2d`
- **Pin Date:** 2026-11-15
- **Notes:** Community reimplementation of ELIC (He et al., CVPR 2022) built on CompressAI.
  - Latent dimension $M = 320$, hyperprior dimension $N = 192$.
  - Unevenly grouped channel slices: `groups = [0, 32, 48, 64, 80, 96]` summing to 320.
  - Per instructions in `design.md` Section 6.1, this submodule must not be edited or updated.
    All wrappers, patches, and integrations reside in `nrcomp/codec/`.
