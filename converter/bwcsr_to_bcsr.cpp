// Convert .bwcsr (weighted binary CSR) to .bcsr (unweighted binary CSR)
// by stripping the weight field from each edge.
//
// .bwcsr format: [num_nodes:u32] [num_edges:u32] [nodePointers:u32*N] [edges: (toNode:u32, weight:u32)*E]
// .bcsr  format: [num_nodes:u32] [num_edges:u32] [nodePointers:u32*N] [edges: toNode:u32*E]
//
// Usage: ./bwcsr_to_bcsr input.bwcsr output.bcsr

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input.bwcsr output.bcsr\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "rb");
    if (!fin) { perror("open input"); return 1; }

    FILE *fout = fopen(argv[2], "wb");
    if (!fout) { perror("open output"); fclose(fin); return 1; }

    uint32_t num_nodes, num_edges;
    fread(&num_nodes, sizeof(uint32_t), 1, fin);
    fread(&num_edges, sizeof(uint32_t), 1, fin);
    printf("num_nodes = %u, num_edges = %u\n", num_nodes, num_edges);

    // Write header
    fwrite(&num_nodes, sizeof(uint32_t), 1, fout);
    fwrite(&num_edges, sizeof(uint32_t), 1, fout);

    // Copy nodePointers as-is
    const size_t BUF_ELEMS = 1 << 20; // 4 MB buffer
    uint32_t *buf = (uint32_t *)malloc(BUF_ELEMS * sizeof(uint32_t));
    if (!buf) { perror("malloc"); return 1; }

    size_t remaining = num_nodes;
    while (remaining > 0) {
        size_t chunk = remaining < BUF_ELEMS ? remaining : BUF_ELEMS;
        size_t r = fread(buf, sizeof(uint32_t), chunk, fin);
        if (r != chunk) { fprintf(stderr, "short read on nodePointers\n"); return 1; }
        fwrite(buf, sizeof(uint32_t), chunk, fout);
        remaining -= chunk;
    }
    printf("nodePointers copied.\n");

    // Read EdgeWithWeight (toNode, weight) pairs, write only toNode
    // Each edge in .bwcsr is 8 bytes: (u32 toNode, u32 weight)
    // We read 2 uint32 per edge, write 1
    const size_t EDGE_BUF = 1 << 20; // process 1M edges at a time
    uint32_t *ebuf = (uint32_t *)malloc(EDGE_BUF * 2 * sizeof(uint32_t));
    uint32_t *obuf = (uint32_t *)malloc(EDGE_BUF * sizeof(uint32_t));
    if (!ebuf || !obuf) { perror("malloc"); return 1; }

    uint64_t edges_done = 0;
    remaining = num_edges;
    while (remaining > 0) {
        size_t chunk = remaining < EDGE_BUF ? remaining : EDGE_BUF;
        size_t r = fread(ebuf, sizeof(uint32_t) * 2, chunk, fin);
        if (r != chunk) { fprintf(stderr, "short read on edges at %lu\n", (unsigned long)edges_done); return 1; }
        for (size_t i = 0; i < chunk; i++) {
            obuf[i] = ebuf[i * 2]; // toNode only, skip weight
        }
        fwrite(obuf, sizeof(uint32_t), chunk, fout);
        edges_done += chunk;
        remaining -= chunk;
        if (edges_done % (100000000ULL) == 0) {
            printf("  %lu / %u edges processed (%.1f%%)\n",
                   (unsigned long)edges_done, num_edges,
                   100.0 * edges_done / num_edges);
        }
    }
    printf("Done: %lu edges converted.\n", (unsigned long)edges_done);

    free(buf);
    free(ebuf);
    free(obuf);
    fclose(fin);
    fclose(fout);
    return 0;
}
