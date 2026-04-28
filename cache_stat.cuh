#ifndef CACHE_STAT_CUH
#define CACHE_STAT_CUH

#ifdef CACHE_STAT

#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <vector>

struct ChunkInfo {
  int chunk_id;
  uint64_t chunk_bytes;
  uint64_t num_total_edges;
  uint32_t start_vertex;  // inclusive
  uint32_t end_vertex;    // exclusive
};

class CacheStatRecorder {
 public:
  CacheStatRecorder(const std::string& csv_path,
                    const std::string& framework,
                    const std::string& algo,
                    const std::string& dataset,
                    int source,
                    int run_id);
  ~CacheStatRecorder();

  // Initialise chunk metadata. `vertex_to_chunk_host` length == num_vertices.
  // Use -1 for vertices that are not in any chunk (e.g., overload vertices).
  void init_chunks(const std::vector<ChunkInfo>& chunks,
                   const std::vector<int>& vertex_to_chunk_host,
                   uint32_t num_vertices);

  void begin_iter(int iter);
  void mark_admission(int chunk_id);
  // Launch the active-counting kernel asynchronously on `stream`.
  // `d_active_mask` is a bool array length == num_vertices.
  // `d_degree` is a uint64 array length == num_vertices.
  void record_demand(const bool* d_active_mask,
                     const uint64_t* d_degree,
                     cudaStream_t stream = 0);
  void end_iter();
  void finalize();

  int num_chunks() const { return static_cast<int>(chunks_.size()); }

 private:
  std::string csv_path_;
  std::string framework_;
  std::string algo_;
  std::string dataset_;
  int source_;
  int run_id_;

  std::ofstream csv_;
  std::vector<ChunkInfo> chunks_;
  std::vector<int> is_resident_;
  std::vector<int> admission_pending_;
  uint32_t num_vertices_ = 0;
  int current_iter_ = -1;

  int* d_vertex_to_chunk_ = nullptr;   // length = num_vertices
  uint32_t* d_active_v_count_ = nullptr; // length = num_chunks
  uint64_t* d_active_e_count_ = nullptr; // length = num_chunks
};

// Public CUDA kernel — defined in cache_stat.cu
__global__ void count_active_per_chunk_kernel(
    const bool* active_mask,
    uint32_t num_vertices,
    const int* vertex_to_chunk,
    const uint64_t* degree,
    int num_chunks,
    uint32_t* d_v_counts,
    uint64_t* d_e_counts);

#endif  // CACHE_STAT
#endif  // CACHE_STAT_CUH
