# Joint LiDAR–3DGS compression: a systematic survey and reproducibility guide

**No existing paper jointly compresses LiDAR point clouds while optimizing for downstream 3D Gaussian Splatting rendering quality — this is a confirmed, wide-open research gap.** Two mature but disconnected communities operate in parallel: learned point cloud compression (LPCC) optimizes for D1/D2 PSNR and downstream detection, while 3DGS compression optimizes for rendered image PSNR/SSIM/LPIPS. The bridge paper closest to this intersection — Bits-to-Photon (NYU, 2024) — decodes compressed point clouds directly into renderable Gaussians, but targets dense RGB point clouds, not sparse LiDAR in driving scenes. A system that compresses raw LiDAR scans while preserving 3DGS reconstruction fidelity at the receiver would be the first of its kind, sitting at a convergence point of three active CVPR/ECCV/NeurIPS publication streams.

This survey covers **40+ papers** across six research areas, audits **13 code repositories** for reproducibility, and provides a concrete experimental starting point on an RTX 5090 / RTX 6000 Ada class GPU.

---

## 1. Direct baselines: methods you must reproduce

These papers form the experimental backbone of any project at this intersection. Each has verified open-source code runnable on a 48 GB GPU.

### 1A. LiDAR-initialized 3DGS for autonomous driving

| Method | Venue | LiDAR use | Datasets | GitHub | Train time |
|--------|-------|-----------|----------|--------|------------|
| **SplatAD** | CVPR 2025 | Init + LiDAR rendering | PandaSet, Argoverse2, nuScenes | `carlinds/splatad` + `georghess/neurad-studio` | Minutes–hours on A100 |
| **Street Gaussians** | ECCV 2024 | Init + depth supervision | KITTI, Waymo | `zju3dv/street_gaussians` | ~30 min / scene on RTX 3090 |
| **OmniRe** | ICLR 2025 Spotlight | Init + depth supervision | Waymo, PandaSet, Argoverse2, KITTI, nuScenes, NuPlan | `ziyc/drivestudio` | Variable; multi-method framework |

**SplatAD** is the single most important baseline. It is the first 3DGS method that renders both camera images and LiDAR data (depth, intensity, ray-drop) from a unified Gaussian representation, using custom CUDA kernels with rolling-shutter compensation. It initializes Gaussians from accumulated multi-frame LiDAR scans, filters dynamic objects, and applies MCMC densification. Built on the neurad-studio framework (itself built on Nerfstudio), it provides Docker/Apptainer recipes and unified dataparsers for five driving datasets. Your key experiment: train SplatAD on PandaSet, then systematically degrade the input LiDAR (simulate compression artifacts via quantization/downsampling) and measure how rendering quality degrades — this establishes the "sensitivity curve" that motivates your joint optimization.

**OmniRe / drivestudio** deserves special attention as a unified codebase. It implements seven methods (3DGS, Deformable-3DGS, 4DGaussians, Street Gaussians, PVG, EmerNeRF, OmniRe) across six datasets with a single training command. This lets you run controlled ablations across methods without reimplementing data pipelines. Configuration is via YAML; a typical command is `python tools/train.py --config_file configs/omnire.yaml dataset=waymo/3cams`.

**Street Gaussians** remains the most lightweight baseline, achieving **135 FPS at 1066×1600** with ~30-minute training on a single RTX 3090. Its simplicity makes it ideal for fast iteration during method development.

### 1B. LiDAR point cloud compression

| Method | Venue | Technique | BD-Rate vs G-PCC | Code | Downstream eval |
|--------|-------|-----------|-------------------|------|-----------------|
| **RENO** | CVPR 2025 | Multiscale sparse tensors | −12.25% vs G-PCCv23 | `NJUVISION/RENO` | 3D detection (PointPillars) |
| **G-PCC v23** | MPEG standard | Octree + RAHT | Anchor | `mpeg-pcc-tmc13` | D1/D2 PSNR only |

**RENO** is the state-of-the-art real-time neural LiDAR codec: **10 fps encode/decode on RTX 3090**, model size just **1 MB**, and 14-bit geometry precision. It uses multiscale sparse tensors (MinkowskiEngine) and achieves 48% bitrate savings over Draco. Critically, RENO evaluates only D1/D2 PSNR and downstream PointPillars detection — **it never evaluates rendering quality**. Your contribution: add a 3DGS rendering quality evaluation loop after RENO decoding, measuring PSNR/SSIM/LPIPS of images rendered from Gaussians initialized with RENO-decoded point clouds vs. uncompressed point clouds.

### 1C. 3DGS compression

| Method | Venue | Ratio vs vanilla 3DGS | MipNeRF360 PSNR | Size (MB) | Code |
|--------|-------|----------------------|-----------------|-----------|------|
| **HAC++** | TPAMI 2025 | >100× | 27.82 | 19.4 | `YihangChen-ee/HAC-plus` |
| **ContextGS** | NeurIPS 2024 | >100× | 27.75 | 19.3 | `wyf0912/ContextGS` |
| **HAC** | ECCV 2024 | ~75× | 27.77 | 22.9 | `YihangChen-ee/HAC` |
| **RDO-Gaussian** | ECCV 2024 | ~40× | 27.05 | 23.5 | `USTC-IMCL/RDO-Gaussian` |
| **LightGaussian** | NeurIPS 2024 | >15× | 27.28 | 42.0 | `VITA-Group/LightGaussian` |

**HAC/HAC++** achieves the best compression–quality tradeoff through binary hash grids for spatial context modeling of Scaffold-GS anchor points, adaptive quantization modules, and entropy coding with a custom CUDA-based codec (10× faster than torchac). It requires CUDA 11.8, builds on Scaffold-GS, and needs MPEG's `tmc3` for geometry coding. Tested on an NVIDIA L40s (48 GB) — identical tier to your RTX 6000 Ada.

**RDO-Gaussian** is methodologically the most relevant compression paper because it formulates 3DGS compression as explicit **rate-distortion optimization** where distortion equals rendering quality. It uses dynamic pruning with learnable Gaussian/SH masks plus entropy-constrained vector quantization (ECVQ). This R-D framework is directly transferable to your joint LiDAR+3DGS system.

**Critical finding: no 3DGS compression method evaluates geometric fidelity metrics** (D1/D2 PSNR, Chamfer distance, Hausdorff distance). All use exclusively image-based metrics. This means no one has verified whether compressed Gaussians preserve the underlying point cloud geometry — another gap you can exploit.

---

## 2. Technically adjacent work: components to reuse

These papers are not direct baselines but contain architectural components, loss functions, or training recipes that transfer directly to a joint LiDAR–3DGS compression system.

### Rendering-oriented point cloud compression

**Bits-to-Photon (B2P)** from NYU (arXiv 2406.05915) is the archetype paper for this intersection. It proposes an end-to-end learned PCC scheme that decodes bitstreams directly into renderable 3D Gaussians via differentiable Gaussian splatting. The encoder uses geometry-invariant 3D sparse convolutions (MinkowskiEngine); the decoder produces Gaussian positions, covariances, and colors; and the entire pipeline is jointly optimized for bitrate + rendering quality. Code is at `huzi96/gaussian-pcloud-render` (tested on RTX 4080 Super). The limitation: it targets dense RGB point clouds (8iVFB, THuman), not sparse LiDAR. **Reuse: the end-to-end architecture and joint R-D loss function.**

**RO-PCAC** (IEEE TCSVT 2025, arXiv 2411.07899) integrates point cloud attribute compression with differentiable rendering, optimizing for rendered multiview image quality instead of point-to-point distortion. It uses a sparse tensor-based transformer (SP-Trans). Code was promised at `net-F/RO-PCAC` but **is not yet publicly available**. **Reuse: the rendering-oriented loss formulation and evaluation protocol.**

### PCC tools applied to Gaussian point clouds

**GausPcgc** (arXiv 2505.18197) treats optimized 3DGS Gaussians as point clouds and applies AI-based PCC with specialized handling for the globally sparse, locally dense distribution of Gaussians. It creates the GausPcc-1K dataset and achieves 8.2% BD-Rate gain over G-PCCv23 while evaluating both PCC metrics and rendering PSNR. Code promised but **not yet released**. **Reuse: the insight that Gaussian distributions require specialized PCC, not off-the-shelf codecs.**

**HybridGS** (arXiv 2505.01938) uses canonical PCC tools like RAHT (from G-PCC) to compress 3DGS data in a dual-channel sparse representation. **Voxel-GS** (arXiv 2512.17528) generates integer-type Gaussian point clouds and compresses via octree + run-length coding, compatible with standardized PCC codecs. Both contribute to the emerging MPEG I-3DGS standardization effort.

### Streaming and progressive codecs for 3DGS

**ProGS** (arXiv 2603.09703, March 2026) is the first streaming-friendly progressive codec for 3DGS, using octree-structured multi-LoD representations with context-based coding. It achieves **45× storage reduction** and supports variable-bandwidth streaming — directly relevant to your autonomous driving data transmission scenario. No public code yet.

**CodecGS** (Fraunhofer HHI, ICCV 2025) achieves **146× compression** by converting 3DGS to progressive tri-plane feature maps, then encoding them with standard HEVC/VVC video codecs. The key insight: DCT entropy loss optimizes feature planes for compatibility with non-differentiable video codecs. Code marked "coming soon."

**VideoRF** (CVPR 2024) serializes 4D radiance fields into 2D feature video streams, enabling real-time mobile streaming by exploiting hardware video accelerators. **Reuse: the paradigm of leveraging existing video codec infrastructure for neural scene transmission.**

### Collaborative perception and V2X

No V2X paper evaluates neural rendering as a downstream task — all optimize for BEV detection/segmentation. However, **Distributed NeRF Learning for Multi-Robot Perception** (arXiv 2409.20289) uses NeRF model weights as the shared representation between agents, achieving implicit compression of sensor data into compact neural parameters. **PreSight** (arXiv 2403.09079) compresses historical traversal data into city-scale NeRF representations for perception priors. Both validate the concept of neural representations as bandwidth-efficient alternatives to raw point cloud transmission.

### Additional 3DGS compression methods with code

| Method | Key technique | Code |
|--------|--------------|------|
| Compact3D | K-means VQ + RLE | `UCDvision/compact3d` |
| EAGLES | Quantized latent embeddings + MLP decoder | `Sharath-girish/efficientgaussian` |
| Scaffold-GS | Anchor-based structured representation | `city-super/Scaffold-GS` |
| Compact-3DGS | Learnable masks + hash-grid color + VQ | `maincold2/Compact-3DGS` |
| C3DGS | Sensitivity-aware VQ + WebGPU | `KeKsBoTer/c3dgs` |
| Mini-Splatting | Gaussian count reduction (compaction) | `fatPeter/mini-splatting` |
| SOG | 2D grid organization + image compression | `fraunhoferhhi/Self-Organizing-Gaussians` |
| SUNDAE | Spectral graph pruning + CNN compensation | `RunyiYang/SUNDAE` |
| Reduced-3DGS | SH band adaptation + codebook VQ | `graphdeco-inria/reduced-3dgs` |

The comprehensive survey **3DGS.zip** (Eurographics 2025 STAR, `w-m.github.io/3dgs-compression-survey/`) provides tabulated comparisons across all methods on four standard benchmarks with PSNR/SSIM/LPIPS/size metrics.

---

## 3. The research gap and how to exploit it

### Three disconnected communities, one missing link

The literature reveals a striking structural gap. Three active research streams produce top-venue papers independently but never intersect:

- **Stream A — LPCC**: RENO, suLPCC, G-PCC optimize for D1/D2 PSNR and detection metrics. They never render images from decoded point clouds.
- **Stream B — 3DGS for driving**: SplatAD, Street Gaussians, OmniRe initialize Gaussians from raw LiDAR. They never consider what happens when LiDAR is compressed before transmission.
- **Stream C — 3DGS compression**: HAC++, ContextGS, RDO-Gaussian compress already-optimized Gaussians. They never handle raw sensor data or evaluate geometric fidelity.

**Your contribution sits at the intersection of all three.** The missing system: compress raw LiDAR point clouds (Stream A) in a way that preserves downstream 3DGS reconstruction and rendering quality (Streams B+C) for bandwidth-constrained autonomous driving data transmission.

### Concrete open problems

**Problem 1: Rendering-aware LiDAR compression.** Modify RENO's loss function to include a differentiable 3DGS rendering term. Currently RENO minimizes point-to-point distance; add an auxiliary loss where decoded points initialize Gaussians, render novel views, and backpropagate rendering quality through the entire pipeline. The technical challenge is making the LiDAR→Gaussian initialization differentiable.

**Problem 2: Joint rate-distortion optimization across modalities.** Extend RDO-Gaussian's R-D framework to operate on raw LiDAR input rather than already-optimized Gaussians. The rate term should measure LiDAR bitstream size; the distortion term should measure rendered image quality. This requires a differentiable path from compressed LiDAR → Gaussian parameters → rendered images.

**Problem 3: Geometric fidelity vs. rendering fidelity tradeoff.** No paper has studied whether D1/D2 PSNR correlates with rendered image PSNR when LiDAR is used as 3DGS initialization. Aggressive geometric compression might destroy fine structures that matter for rendering but not for D1 metrics, or vice versa. A systematic study of this correlation would be independently publishable.

**Problem 4: V2X neural scene transmission.** In cooperative driving, vehicles could transmit compressed LiDAR that a receiver uses for 3DGS reconstruction, rather than transmitting rendered images or BEV features. This reframes V2X bandwidth allocation as a neural rendering rate-distortion problem — completely unexplored territory.

---

## 4. Datasets ranked for this specific research agenda

Your experiments need datasets with **synchronized LiDAR + surround cameras + ego-motion + 3D annotations** and existing 3DGS baselines for comparison.

| Rank | Dataset | LiDAR | Cameras | 3DGS papers using it | Download | License |
|------|---------|-------|---------|----------------------|----------|---------|
| **1** | **Waymo Open** | 5 LiDARs, ~177–300K pts/frame | 5 cameras, 10 Hz | Most popular (Street Gaussians, OmniRe, S³Gaussian, IDSplat) | ~1.5 TB | Non-commercial |
| **2** | **nuScenes** | 32-beam, ~34K pts/sweep | 6 cameras, 360° surround | SplatAD, DrivingGaussian, NeuRAD | ~350 GB | Non-commercial |
| **3** | **PandaSet** | Pandar64 + PandarGT solid-state | 6 cameras, 360° | SplatAD, NeuRAD, UniSim, AutoSplat | ~80 GB | **CC BY 4.0** (commercial OK) |
| **4** | **Argoverse 2** | 2×VLP-32C (64 beams total) | 7 ring + 2 stereo = 9 cameras | SplatAD, Street Gaussians | ~1 TB | CC BY-NC-SA 4.0 |

**Recommended starting point: PandaSet.** It has the most permissive license (CC BY 4.0), manageable download size (80 GB), dual LiDAR systems (spinning + solid-state), 6 surround cameras, and is supported by both SplatAD and NeuRAD out of the box. Waymo's NOTR subset (dynamic32 + static32 splits) is the de facto benchmark for final evaluation, but its ~1.5 TB size and restrictive license make it less practical for initial development.

**Standard training protocol for 3DGS on driving data:**
1. Accumulate multi-frame LiDAR in world coordinates using provided ego-poses
2. Filter dynamic objects using 3D bounding box annotations
3. Initialize Gaussian positions from the accumulated static point cloud
4. Train for **30,000 iterations** with Adam optimizer (position LR: 1.6e-4 with exponential decay; opacity LR: 0.05; SH LR: 0.0025)
5. Loss: L1 + 0.2 × D-SSIM + LiDAR depth supervision
6. Densification from iteration 500 to 15,000, every 100 steps
7. Evaluate on held-out frames (every 10th frame) using PSNR, SSIM, LPIPS

---

## 5. Reproducibility audit summary

All entries below have been verified against actual GitHub repositories.

| Method | GitHub | VRAM needed | LiDAR? | Key deps | Status |
|--------|--------|------------|--------|----------|--------|
| SplatAD | `georghess/neurad-studio` | ≥24 GB | Required | Nerfstudio, gsplat (custom fork), CUDA | ✅ Runnable |
| Street Gaussians | `zju3dv/street_gaussians` | ≥24 GB | Required | PyTorch 1.13, CUDA 11.6 | ✅ Runnable |
| OmniRe | `ziyc/drivestudio` | ≥24 GB | Required | gsplat 1.3, pytorch3d, nvdiffrast, smplx | ✅ Runnable |
| RENO | `NJUVISION/RENO` | ~8 GB | LiDAR-only | MinkowskiEngine, PyTorch | ✅ Runnable |
| B2P | `huzi96/gaussian-pcloud-render` | ~16 GB | Dense PC (not LiDAR) | MinkowskiEngine, PyTorch 1.12, CUDA 11.3 | ✅ Runnable |
| HAC / HAC++ | `YihangChen-ee/HAC` / `HAC-plus` | ~48 GB | No | Scaffold-GS, tmc3 (G-PCC), CUDA 11.8 | ✅ Runnable |
| RDO-Gaussian | `USTC-IMCL/RDO-Gaussian` | ~24 GB | No | 3DGS base | ✅ Runnable |
| ContextGS | `wyf0912/ContextGS` | ~24 GB | No | Scaffold-GS, CUDA 11.8 | ✅ Runnable |
| LightGaussian | `VITA-Group/LightGaussian` | ~24 GB | No | 3DGS base, conda | ✅ Runnable |
| GausPcgc | Not released | Unknown | No | MinkowskiEngine (expected) | ⏳ Promised |
| RO-PCAC | Not released | Unknown | No | Unknown | ⏳ Promised |
| CodecGS | Not released | Unknown | No | HEVC/VVC | ⏳ Promised |

All verified repos are feasible on your **RTX 5090 / RTX 6000 Ada (48 GB VRAM)**. HAC was explicitly tested on an L40s (48 GB, same tier). SplatAD and Street Gaussians run on RTX 3090 (24 GB), so 48 GB gives comfortable headroom for multi-camera driving scenes.

---

## 6. Recommended experimental pipeline

Based on this survey, here is a concrete, reproducible starting recipe:

**Phase 1 — Establish baselines (Week 1–2).** Install neurad-studio and train SplatAD on PandaSet (10 scenes). Record per-scene PSNR/SSIM/LPIPS for camera rendering and depth/intensity/ray-drop metrics for LiDAR rendering. This gives you the "uncompressed LiDAR → 3DGS" upper bound. Simultaneously, install RENO and compress the same PandaSet LiDAR sequences at multiple bitrates. Record D1/D2 PSNR at each rate point.

**Phase 2 — Measure the sensitivity gap (Week 3–4).** Feed RENO-decoded LiDAR (at each bitrate) into SplatAD as initialization instead of raw LiDAR. Train 3DGS and measure rendering quality degradation. Plot the curve: LiDAR bitrate → D1 PSNR vs. LiDAR bitrate → rendered image PSNR. **If these curves diverge — and they almost certainly will — you have your core contribution.** Traditional LPCC metrics (D1/D2) do not predict rendering quality, proving the need for rendering-aware compression.

**Phase 3 — Build the joint system (Week 5–8).** Fork RENO's encoder-decoder and add a differentiable 3DGS rendering branch. The decoder outputs point positions that are used to initialize Gaussians; a lightweight Gaussian parameter predictor (inspired by B2P's architecture) estimates covariance/color; differentiable splatting (via gsplat) renders images; and the rendering loss backpropagates through the entire pipeline. The total loss becomes λ_rate × bitrate + λ_geo × D1_distortion + λ_render × (1 − SSIM). Apply HAC-style entropy coding to the Gaussian parameters for additional compression.

**Phase 4 — Evaluate and ablate (Week 9–10).** Compare against: (a) G-PCC + SplatAD, (b) RENO + SplatAD, (c) uncompressed LiDAR + SplatAD, (d) HAC/HAC++ applied to already-trained Gaussians. Report both traditional PCC metrics (D1/D2 PSNR, Chamfer distance) and rendering metrics (PSNR, SSIM, LPIPS). Scale to Waymo NOTR for the final benchmark.

---

## Conclusion: where the opportunity lies

This survey confirms that the intersection of LiDAR compression and 3DGS rendering quality is **genuinely unexplored**. The closest work — B2P from NYU — bridges PCC and Gaussian splatting for dense RGB point clouds but not for sparse LiDAR in driving contexts. RENO delivers state-of-the-art real-time LiDAR compression but evaluates only geometry metrics. SplatAD achieves state-of-the-art LiDAR-initialized 3DGS but assumes uncompressed input. The gap between these three papers is precisely your research contribution.

The most impactful novelty is the **sensitivity analysis**: demonstrating empirically that D1/D2 PSNR is a poor proxy for rendering quality when compressed LiDAR initializes 3DGS. This finding alone would justify a rendering-aware compression loss. The technical infrastructure exists — RENO, SplatAD, HAC, and gsplat are all open-source, all run on your GPU, and all use compatible frameworks (PyTorch + CUDA + MinkowskiEngine). The dataset pipeline is solved by PandaSet (permissive license, dual LiDAR, supported by neurad-studio). What remains is connecting these components with a differentiable path from compressed LiDAR to rendered pixels — a tractable engineering challenge with high publication potential at CVPR, ECCV, or NeurIPS.