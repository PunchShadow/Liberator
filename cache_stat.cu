#ifdef CACHE_STAT

#include "cache_stat.cuh"

#include <iostream>

__global__ void count_active_per_chunk_kernel(
    const bool* active_mask,
    uint32_t num_vertices,
    const int* vertex_to_chunk,
    const uint64_t* degree,
    int num_chunks,
    uint32_t* d_v_counts,
    uint64_t* d_e_counts) {
  uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= num_vertices) return;
  int c = vertex_to_chunk[v];
  if (c < 0 || c >= num_chunks) return;
  if (!active_mask[v]) return;
  atomicAdd(&d_v_counts[c], 1u);
  atomicAdd(reinterpret_cast<unsigned long long*>(&d_e_counts[c]),
            static_cast<unsigned long long>(degree[v]));
}

CacheStatRecorder::CacheStatRecorder(const std::string& csv_path,
                                     const std::string& framework,
                                     const std::string& algo,
                                     const std::string& dataset,
                                     int source,
                                     int run_id)
    : csv_path_(csv_path),
      framework_(framework),
      algo_(algo),
      dataset_(dataset),
      source_(source),
      run_id_(run_id) {}

CacheStatRecorder::~CacheStatRecorder() {
  if (csv_.is_open() || d_vertex_to_chunk_ != nullptr) {
    finalize();
  }
}

void CacheStatRecorder::init_chunks(
    const std::vector<ChunkInfo>& chunks,
    const std::vector<int>& vertex_to_chunk_host,
    uint32_t num_vertices) {
  chunks_ = chunks;
  num_vertices_ = num_vertices;
  is_resident_.assign(chunks_.size(), 0);
  admission_pending_.assign(chunks_.size(), 0);

  csv_.open(csv_path_);
  if (!csv_.is_open()) {
    std::cerr << "[CACHE_STAT] failed to open " << csv_path_ << " for writing"
              << std::endl;
    return;
  }
  csv_ << "framework,algo,dataset,source,run_id,iter,chunk_id,chunk_bytes,"
          "num_total_edges,is_resident,num_active_vertices,num_active_edges,"
          "is_hit,is_admission_event,is_eviction_event\n";

  cudaMalloc(&d_vertex_to_chunk_, num_vertices * sizeof(int));
  cudaMemcpy(d_vertex_to_chunk_, vertex_to_chunk_host.data(),
             num_vertices * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&d_active_v_count_, chunks_.size() * sizeof(uint32_t));
  cudaMalloc(&d_active_e_count_, chunks_.size() * sizeof(uint64_t));

  current_iter_ = -1;
  std::cerr << "[CACHE_STAT] initialised " << chunks_.size()
            << " chunks; csv=" << csv_path_ << std::endl;
}

void CacheStatRecorder::begin_iter(int iter) {
  current_iter_ = iter;
  cudaMemset(d_active_v_count_, 0, chunks_.size() * sizeof(uint32_t));
  cudaMemset(d_active_e_count_, 0, chunks_.size() * sizeof(uint64_t));
  std::fill(admission_pending_.begin(), admission_pending_.end(), 0);
}

void CacheStatRecorder::mark_admission(int chunk_id) {
  if (chunk_id < 0 || chunk_id >= static_cast<int>(chunks_.size())) return;
  if (is_resident_[chunk_id] == 0) {
    is_resident_[chunk_id] = 1;
    admission_pending_[chunk_id] = 1;
  }
}

void CacheStatRecorder::record_demand(const bool* d_active_mask,
                                       const uint64_t* d_degree,
                                       cudaStream_t stream) {
  if (num_vertices_ == 0) return;
  const int threads = 256;
  const int blocks = (num_vertices_ + threads - 1) / threads;
  count_active_per_chunk_kernel<<<blocks, threads, 0, stream>>>(
      d_active_mask, num_vertices_, d_vertex_to_chunk_, d_degree,
      static_cast<int>(chunks_.size()), d_active_v_count_, d_active_e_count_);
}

void CacheStatRecorder::end_iter() {
  const size_t C = chunks_.size();
  std::vector<uint32_t> v_counts(C);
  std::vector<uint64_t> e_counts(C);
  cudaMemcpy(v_counts.data(), d_active_v_count_, C * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(e_counts.data(), d_active_e_count_, C * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);
  for (size_t c = 0; c < C; ++c) {
    int is_hit = (is_resident_[c] && v_counts[c] > 0) ? 1 : 0;
    csv_ << framework_ << ',' << algo_ << ',' << dataset_ << ',' << source_
         << ',' << run_id_ << ',' << current_iter_ << ',' << c << ','
         << chunks_[c].chunk_bytes << ',' << chunks_[c].num_total_edges << ','
         << is_resident_[c] << ',' << v_counts[c] << ',' << e_counts[c] << ','
         << is_hit << ',' << admission_pending_[c] << ',' << 0 << '\n';
  }
}

void CacheStatRecorder::finalize() {
  if (csv_.is_open()) {
    csv_.flush();
    csv_.close();
  }
  if (d_vertex_to_chunk_) {
    cudaFree(d_vertex_to_chunk_);
    d_vertex_to_chunk_ = nullptr;
  }
  if (d_active_v_count_) {
    cudaFree(d_active_v_count_);
    d_active_v_count_ = nullptr;
  }
  if (d_active_e_count_) {
    cudaFree(d_active_e_count_);
    d_active_e_count_ = nullptr;
  }
}

#endif  // CACHE_STAT
