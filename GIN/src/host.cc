#include "host.h"
#include "xcl2.hpp"

aligned_vector<WT_TYPE> node_mlp_1_weights_fixed(NUM_LAYERS * MLP_1_OUT * EMB_DIM);
aligned_vector<WT_TYPE> node_mlp_1_bias_fixed(NUM_LAYERS * MLP_1_OUT);
aligned_vector<WT_TYPE> node_mlp_2_weights_fixed(NUM_LAYERS * EMB_DIM * MLP_1_OUT);
aligned_vector<WT_TYPE> node_mlp_2_bias_fixed(NUM_LAYERS * EMB_DIM);
aligned_vector<WT_TYPE> node_embedding_table_fixed(ND_FEATURE_TOTAL * EMB_DIM);
aligned_vector<WT_TYPE> edge_embedding_table_fixed(NUM_LAYERS * ED_FEATURE_PER_LAYER * EMB_DIM);
aligned_vector<WT_TYPE> graph_pred_linear_weight_fixed(NUM_TASK * EMB_DIM);
aligned_vector<WT_TYPE> graph_pred_linear_bias_fixed(NUM_TASK);
aligned_vector<WT_TYPE> eps_fixed(NUM_LAYERS);

static const char* GRAPH_INFO_FORMAT = "../../../graphs/graph_info/g%d_info.txt";
static const char* GRAPH_NAME_FORMAT = "../../../graphs/graph_bin/g%d";

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_GIN_compute_graphs;
    cl::CommandQueue q;
    
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err,
                  q = cl::CommandQueue(
                      context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i
                      << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_GIN_compute_graphs = cl::Kernel(program, "GIN_compute_graphs", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
	    std::cout << "Failed to program any device found, exit!\n";
	    exit(EXIT_FAILURE);
    }


    printf("\n******* This is the HLS for GIN model *******\n");

    load_weights();
    printf("\n******* Weights loading done *******\n");

    OCL_CHECK(err, cl::Buffer node_mlp_1_weights_buf(context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * MLP_1_OUT * EMB_DIM * sizeof(WT_TYPE),
        node_mlp_1_weights_fixed.data(),
        &err));
    OCL_CHECK(err, cl::Buffer node_mlp_1_bias_buf(context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * MLP_1_OUT * sizeof(WT_TYPE),
        node_mlp_1_bias_fixed.data(),
        &err));
    OCL_CHECK(err, cl::Buffer node_mlp_2_weights_buf(context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * EMB_DIM * MLP_1_OUT * sizeof(WT_TYPE),
        node_mlp_2_weights_fixed.data(),
        &err));
    OCL_CHECK(err, cl::Buffer node_mlp_2_bias_buf(context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * EMB_DIM * sizeof(WT_TYPE),
        node_mlp_2_bias_fixed.data(),
        &err));
    OCL_CHECK(err, cl::Buffer node_embedding_table_buf(context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        ND_FEATURE_TOTAL * EMB_DIM * sizeof(WT_TYPE),
        node_embedding_table_fixed.data(),
        &err));
    OCL_CHECK(err, cl::Buffer edge_embedding_table_buf(context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * ED_FEATURE_PER_LAYER * EMB_DIM * sizeof(WT_TYPE),
        edge_embedding_table_fixed.data(),
        &err));
    OCL_CHECK(err, cl::Buffer graph_pred_linear_weight_buf(context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_TASK * EMB_DIM * sizeof(WT_TYPE),
        graph_pred_linear_weight_fixed.data(),
        &err));
    OCL_CHECK(err, cl::Buffer graph_pred_linear_bias_buf(context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_TASK * sizeof(WT_TYPE),
        graph_pred_linear_bias_fixed.data(),
        &err));
    OCL_CHECK(err, cl::Buffer eps_buf(context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * sizeof(WT_TYPE),
        eps_fixed.data(),
        &err));

    int num_of_graphs = NUM_GRAPHS;
    aligned_vector<int> nums_of_nodes(NUM_GRAPHS);
    aligned_vector<int> nums_of_edges(NUM_GRAPHS);
    aligned_vector<int> reload_weights(NUM_GRAPHS);
    aligned_vector<FM_TYPE> result(NUM_GRAPHS * NUM_TASK);
    aligned_vector<node_feature_t> node_feature;
    aligned_vector<edge_t> edge_list;
    aligned_vector<edge_attr_t> edge_attr;

    for (int g = 1; g <= NUM_GRAPHS; g++)
    {
        char graph_name[128];
        char info_file[128];
        int num_of_nodes;
        int num_of_edges;

        sprintf(info_file, GRAPH_INFO_FORMAT, g);
        sprintf(graph_name, GRAPH_NAME_FORMAT, g);

        FILE* f_info = fopen(info_file, "r");
        fscanf(f_info, "%d\n%d", &num_of_nodes, &num_of_edges);
        fclose(f_info);

        nums_of_nodes[g - 1] = num_of_nodes;
        nums_of_edges[g - 1] = num_of_edges;
        reload_weights[g - 1] = g == 1;

        fetch_one_graph(g, graph_name, node_feature, edge_list, edge_attr, num_of_nodes, num_of_edges);
    }

    OCL_CHECK(err, cl::Buffer nums_of_nodes_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_GRAPHS * sizeof(int),
        nums_of_nodes.data(),
        &err));
    OCL_CHECK(err, cl::Buffer nums_of_edges_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_GRAPHS * sizeof(int),
        nums_of_edges.data(),
        &err));
    OCL_CHECK(err, cl::Buffer reload_weights_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_GRAPHS * sizeof(int),
        reload_weights.data(),
        &err));
    OCL_CHECK(err, cl::Buffer result_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
        NUM_GRAPHS * NUM_TASK * sizeof(FM_TYPE),
        result.data(),
        &err));
    OCL_CHECK(err, cl::Buffer node_feature_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        node_feature.size() * sizeof(node_feature_t),
        node_feature.data(),
        &err));
    OCL_CHECK(err, cl::Buffer edge_list_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        edge_list.size() * sizeof(edge_t),
        edge_list.data(),
        &err));
    OCL_CHECK(err, cl::Buffer edge_attr_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        edge_attr.size() * sizeof(edge_attr_t),
        edge_attr.data(),
        &err));

    int idx = 0;
    krnl_GIN_compute_graphs.setArg(idx++, num_of_graphs);
    krnl_GIN_compute_graphs.setArg(idx++, nums_of_nodes_buf);
    krnl_GIN_compute_graphs.setArg(idx++, nums_of_edges_buf);
    krnl_GIN_compute_graphs.setArg(idx++, reload_weights_buf);
    krnl_GIN_compute_graphs.setArg(idx++, result_buf);
    krnl_GIN_compute_graphs.setArg(idx++, node_feature_buf);
    krnl_GIN_compute_graphs.setArg(idx++, edge_list_buf);
    krnl_GIN_compute_graphs.setArg(idx++, edge_attr_buf);
    krnl_GIN_compute_graphs.setArg(idx++, node_embedding_table_buf);
    krnl_GIN_compute_graphs.setArg(idx++, edge_embedding_table_buf);
    krnl_GIN_compute_graphs.setArg(idx++, node_mlp_1_weights_buf);
    krnl_GIN_compute_graphs.setArg(idx++, node_mlp_1_bias_buf);
    krnl_GIN_compute_graphs.setArg(idx++, node_mlp_2_weights_buf);
    krnl_GIN_compute_graphs.setArg(idx++, node_mlp_2_bias_buf);
    krnl_GIN_compute_graphs.setArg(idx++, graph_pred_linear_weight_buf);
    krnl_GIN_compute_graphs.setArg(idx++, graph_pred_linear_bias_buf);

    for (int i = 0; i < NUM_TRIALS; i++)
    {
        printf("(%d/%d) Computing GIN ...\n", i + 1, NUM_TRIALS);
        OCL_CHECK(err, err = q.enqueueTask(krnl_GIN_compute_graphs));
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({result_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
        OCL_CHECK(err, err = q.finish());
    }

    FILE* c_output = fopen("HLS_output.txt", "w+");
    for (int g = 1; g <= NUM_GRAPHS; g++) {
        int num_of_nodes = nums_of_nodes[g - 1];
        int num_of_edges = nums_of_edges[g - 1];
        char graph_name[128];
        sprintf(graph_name, GRAPH_NAME_FORMAT, g);

        printf("********** Graph %s *************\n", graph_name);
        printf("# of nodes: %d, # of edges: %d\n", num_of_nodes, num_of_edges);
        for (int t = 0; t < NUM_TASK; t++) {
            printf("%.8f\n", float(result[(g - 1) * NUM_TASK + t]));
            fprintf(c_output, "g%d: %.8f\n", g, float(result[(g - 1) * NUM_TASK + t]));
        }
        printf("GIN computation done.\n");
    }

    return 0;
}
