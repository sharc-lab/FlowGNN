#ifndef __HOST_H__
#define __HOST_H__

#include <array>
#include <ap_fixed.h>
#include "xcl2.hpp"

// #region Model Parameters
constexpr int MAX_EDGE = 500;
constexpr int MAX_NODE = 500;
constexpr int ND_FEATURE = 9;
constexpr int EDGE_ATTR = 3;
constexpr int EMB_DIM = 100;
constexpr int NUM_TASK = 1;
constexpr int L_IN = 200;
constexpr int L_OUT = 100;
constexpr int NUM_LAYERS = 4;
// #endregion

// #region Data Types
typedef ap_fixed<16, 3> FM_TYPE;
typedef ap_fixed<16, 3> WT_TYPE;

typedef struct {
    int u;
    int v;
} edge_t;

typedef std::array<int, ND_FEATURE> node_feature_t;
typedef std::array<WT_TYPE, 4> node_eigen_t;
// #endregion

constexpr int NUM_GRAPHS = 4113;
constexpr int NUM_TRIALS = 100;

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
