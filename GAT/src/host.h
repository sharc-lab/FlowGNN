#ifndef __HOST_H__
#define __HOST_H__

#include <array>
#include <ap_fixed.h>
#include "xcl2.hpp"
#include "dataset.hpp"

// #region Model Parameters
constexpr int MAX_EDGE = 5500;
constexpr int MAX_NODE = 500;
constexpr int ND_FEATURE = 9;
constexpr int ND_FEATURE_TOTAL = 173;
constexpr int EDGE_ATTR = 3;
constexpr int ED_FEATURE_PER_LAYER = 13;
constexpr int EMB_DIM = 16;
constexpr int NUM_HEADS = 4;
constexpr int NUM_LAYERS = 5;
constexpr int NUM_TASK = 1;
// #endregion

// #region Hardware Parameters
constexpr int GATHER_PARALLEL = 8; // how many dimensions of EMB_DIM should a message passing PE process each cycle?
constexpr int APPLY_PARALLEL = 1; // how many dimensions of EMB_DIM should the node embedding PE process each cycle?
constexpr int NODE_PARALLEL = 2; // how many nodes should the node embedding PE process simultaneously?
constexpr int EDGE_PARALLEL = 4; // how many message passing PEs are there?
constexpr int MLP_PARALLEL = 2;
// #endregion

// #region Data Types
typedef ap_fixed<16, 6> FM_TYPE;
typedef ap_fixed<16, 6> WT_TYPE;

typedef std::array<int, ND_FEATURE> node_feature_t;
typedef std::array<int, EDGE_ATTR> edge_attr_t;

typedef struct {
    int u;
    int v;
} edge_t;
// #endregion

constexpr int NUM_TRIALS = 25;

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

extern aligned_vector<WT_TYPE> scoring_fn_target_fixed;
extern aligned_vector<WT_TYPE> scoring_fn_source_fixed;
extern aligned_vector<WT_TYPE> linear_proj_weights_fixed;
extern aligned_vector<WT_TYPE> skip_proj_weights_fixed;
extern aligned_vector<WT_TYPE> graph_pred_weights_fixed;
extern aligned_vector<WT_TYPE> graph_pred_bias_fixed;

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
