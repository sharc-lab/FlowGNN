#ifndef __HOST_H__
#define __HOST_H__

#include "dcl.h"
#include "xcl2.hpp"
#include "dataset.hpp"

constexpr int NUM_TRIALS = 25;

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

extern aligned_vector<WT_TYPE> node_embedding_weight_fixed;
extern aligned_vector<WT_TYPE> node_conv_weights_fixed;
extern aligned_vector<WT_TYPE> node_conv_bias_fixed;
extern aligned_vector<WT_TYPE> graph_mlp_1_weights_fixed;
extern aligned_vector<WT_TYPE> graph_mlp_1_bias_fixed;
extern aligned_vector<WT_TYPE> graph_mlp_2_weights_fixed;
extern aligned_vector<WT_TYPE> graph_mlp_2_bias_fixed;
extern aligned_vector<WT_TYPE> graph_mlp_3_weights_fixed;
extern aligned_vector<WT_TYPE> graph_mlp_3_bias_fixed;
extern aligned_vector<WT_TYPE> avg_deg_fixed;

void load_weights();
void fetch_one_graph(
    int g,
    char* graph_name,
    aligned_vector<node_feature_t>& node_feature,
    aligned_vector<edge_t>& edge_list,
    int num_of_nodes,
    int num_of_edges
);

#endif
