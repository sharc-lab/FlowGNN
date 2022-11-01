#ifndef __HOST_H__
#define __HOST_H__

#include "dcl.h"
#include "xcl2.hpp"
#include "dataset.hpp"

constexpr int NUM_TRIALS = 25;

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

extern aligned_vector<WT_TYPE> embedding_h_atom_embedding_list_weights;
extern aligned_vector<WT_TYPE> layers_posttrans_fully_connected_0_linear_weight_in;
extern aligned_vector<WT_TYPE> layers_posttrans_fully_connected_0_linear_bias_in;
extern aligned_vector<WT_TYPE> MLP_layer_FC_layers_0_weight_in;
extern aligned_vector<WT_TYPE> MLP_layer_FC_layers_0_bias_in;
extern aligned_vector<WT_TYPE> MLP_layer_FC_layers_1_weight_in;
extern aligned_vector<WT_TYPE> MLP_layer_FC_layers_1_bias_in;
extern aligned_vector<WT_TYPE> MLP_layer_FC_layers_2_weight_in;
extern aligned_vector<WT_TYPE> MLP_layer_FC_layers_2_bias_in;

void load_weights();
void fetch_one_graph(
    int g,
    char* graph_name,
    aligned_vector<node_feature_t>& node_feature,
    aligned_vector<node_eigen_t>& node_eigen,
    aligned_vector<edge_t>& edge_list,
    aligned_vector<int>& edge_attr,
    int num_of_nodes,
    int num_of_edges
);

#endif
