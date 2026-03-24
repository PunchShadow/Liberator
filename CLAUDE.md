# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Liberator is a GPU out-of-memory graph processing framework that partitions graphs too large for GPU memory and streams data between host and device. It supports BFS, CC, SSSP, and PageRank. The binary is named `ptGraph`. The project is published in IEEE TPDS 2023.

## Build

Requires CMake 3.17+, CUDA 11.4+, and a C++14-capable compiler. The CUDA compiler path is hardcoded to `/usr/local/cuda/bin/nvcc` in CMakeLists.txt — update if your installation differs.

```bash
mkdir -p build && cd build
cmake ..
make
```

The `CUDA_ARCHITECTURES` property in CMakeLists.txt is set to `"86"` (Ampere). Change this to match your GPU (e.g., `"75"` for Turing, `"80"` for A100).

The converter (text edge list to binary CSR) is built separately:
```bash
cd converter
g++ -o converter converter.cpp
```

## Running

```bash
./ptGraph \
  --input <path-to-graph> \
  --type <bfs|cc|sssp|pr> \
  --source <node-id>       # BFS/SSSP only, default 0
  --model <0|7>            # 0 = Ascetic (baseline), 7 = Liberator
  --testTime <n>           # number of runs, default 1
```

## Input Formats

- `.bcsr` — binary CSR for BFS and CC
- `.bwcsr` — binary weighted CSR for SSSP
- `.bcsc` — binary CSC for PageRank

Use the `converter/` tool to convert text edge lists to these binary formats.

## Architecture

All source files are in the project root (flat structure, no `src/` directory).

- **`main.cu`** — Entry point. Selects algorithm and model, dispatches to `*_opt` or `new*_opt` functions.
- **`CalculateOpt.cu/.cuh`** — Algorithm orchestration. Contains both baseline (`bfs_opt`, `cc_opt`, `sssp_opt`, `pr_opt`) and Liberator (`newbfs_opt`, `newcc_opt`, `newsssp_opt`, `newpr_opt`) implementations. The `new*` variants use data reuse optimizations.
- **`NewCalculateOpt.cu`** — Additional Liberator-model algorithm implementations.
- **`New_CC_opt.cuh`** — Liberator-specific CC implementation (`New_CC_opt` function).
- **`GraphMeta.cu/.cuh`** — Core `GraphMeta<EdgeType>` template class managing graph data, GPU memory allocation, partition management, and host-device transfers.
- **`gpu_kernels.cu/.cuh`** — CUDA kernels for each algorithm (`bfs_kernel`, `cc_kernel`, `sssp_kernel`, `pr_kernel`) plus overload/subgraph variants.
- **`common.cu/.cuh`** — Partitioning helpers, degree calculation, edge list overload management, and data recording utilities.
- **`globals.cuh`** — Type definitions (`SIZE_TYPE`, `EDGE_POINTER_TYPE`, `EdgeWithWeight`, `FragmentData`) and algorithm enum.
- **`ArgumentParser.cu/.cuh`** — CLI argument parsing (key-value pairs with `--` prefix).
- **`TimeRecord.cu/.cuh`** — GPU timing via CUDA events.
- **`range.cuh`** — Python-like range utility used in GPU grid-stride loops.
- **`constants.cuh`** — Hardcoded dataset paths (for original authors' environment; not used at runtime when `--input` is specified).

## Key Design Concepts

- **Model 0 (Ascetic)**: Baseline partitioning — transfers all needed partitions each iteration.
- **Model 7 (Liberator)**: Data reuse optimization — tracks which partitions are already on GPU and avoids redundant transfers. Uses `FragmentData` structs to track partition residency.
- **Graph partitioning**: The edge list is divided into fixed-size partitions. Each iteration identifies which partitions contain edges for active vertices and transfers only those not already resident on the GPU.
- **Overload mechanism**: When active vertices' edges exceed GPU memory, an "overload" edge list is assembled on the CPU (using multithreaded `fillDynamic`) and streamed to GPU in sub-partitions.

## Testing

No automated test suite. Validate by running algorithms on small graphs with known results and comparing output.
