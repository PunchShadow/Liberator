# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat CUDA/C++ project: most production sources live at the top level rather than under `src/`. Use `main.cu` as the CLI entry point, `CalculateOpt*.cu` and `New_CC_opt.cuh` for algorithm orchestration, `GraphMeta.*` for graph storage and partition management, and `gpu_kernels.*` for CUDA kernels. Algorithm-specific helpers live in `bfs.*`, `cc.*`, and `sssp.*`. The standalone format converter is in `converter/`. Treat `build/` as generated CMake output and keep it untracked.

## Build, Test, and Development Commands
Configure and build the main binary with:

```bash
cmake -S . -B build
cmake --build build -j
```

This produces `build/ptGraph`. Run it with explicit flags, for example:

```bash
./build/ptGraph --input graph.bcsr --type bfs --source 0 --model 7 --testTime 1
```

Build the converter separately when needed:

```bash
g++ -O2 -std=c++14 -o converter/converter converter/converter.cpp
```

## Coding Style & Naming Conventions
Match the style already present in the file you touch; this codebase mixes older tab-indented blocks with newer 4-space sections. Keep opening braces on the same line, prefer quoted local includes, and keep diffs narrow instead of reformatting unrelated code. Follow existing naming patterns: PascalCase for types such as `ArgumentParser` and `GraphMeta`, and repository-local function names such as `newbfs_opt` or `convertTxtToByte`. No formatter or linter is configured here.

## Testing Guidelines
There is no automated test suite or coverage gate. Validate changes by rebuilding and running the affected algorithm on a small known graph, then record the exact command and result. If you touch parsing, formats, or memory-limit logic, test both the main binary and the `converter/` utility. `test.cpp` is a local experiment file, not part of the default build.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects such as `Add ...`, `Support ...`, and `Update ...`. Keep each commit focused on one logical change. PRs should state the problem, list touched algorithms or core modules, include the commands used for build and validation, and note any CUDA or GPU assumptions. For this CLI project, sample terminal output is more useful than screenshots.

## Configuration Tips
`CMakeLists.txt` hardcodes `/usr/local/cuda/bin/nvcc` and `CUDA_ARCHITECTURES "86"`; adjust locally for your environment. Prefer `--input` over editing dataset paths in `constants.cuh`.
