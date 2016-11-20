//
// Created by xixuan on 10/10/16.
//

#include "base_learner.h"
#include "tools.h"

#include <cassert>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>

namespace dbm {

    template
    class Tree_node<float>;

    template
    class Tree_node<double>;

    template
    class Linear_regression<double>;

    template
    class Linear_regression<float>;

    template
    class Global_mean<float>;

    template
    class Global_mean<double>;

    template
    class Neural_network<float>;

    template
    class Neural_network<double>;

}

namespace dbm {

    template
    class Tree_info<double>;

    template
    class Tree_info<float>;

}

namespace dbm {

    template <typename T>
    Global_mean<T>::Global_mean() : Base_learner<T>('m') {}

    template <typename T>
    Global_mean<T>::~Global_mean() {};

    template <typename T>
    T Global_mean<T>::predict_for_row(const Matrix<T> &data, int row_ind) {
        return mean;
    }

    template <typename T>
    void Global_mean<T>::predict(const Matrix<T> &data_x,
                                 Matrix<T> &prediction,
                                 const T shrinkage,
                                 const int *row_inds,
                                 int n_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #if _DEBUG_BASE_LEARNER
            assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) + shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #if _DEBUG_BASE_LEARNER
            assert(n_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < n_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) + shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }

}

namespace dbm {

    template <typename T>
    Neural_network<T>::Neural_network(int n_predictor, int n_hidden_neuron, char loss_type) :
            n_predictor(n_predictor), n_hidden_neuron(n_hidden_neuron),
            loss_type(loss_type), Base_learner<T>('n') {

        col_inds = new int[n_predictor];

        input_weight = new Matrix<T>(n_hidden_neuron, n_predictor + 1);
        hidden_weight = new Matrix<T>(1, n_hidden_neuron + 1);

        input_output = new Matrix<T>(n_predictor + 1, 1, 0);
        hidden_output = new Matrix<T>(n_hidden_neuron + 1, 1, 0);

    }

    template <typename T>
    Neural_network<T>::~Neural_network<T>() {
        delete input_weight, hidden_weight, input_output, hidden_output, col_inds;
        input_weight = nullptr, hidden_weight = nullptr;
        input_output = nullptr, hidden_output = nullptr, col_inds = nullptr;
    }

    template <typename T>
    inline T Neural_network<T>::activation(const T &input) {
        return 1 / (1 + std::exp( - input));
    }

    template <typename T>
    void Neural_network<T>::forward() {
        T ip = 0;
        for(int i = 0; i < n_hidden_neuron; ++i) {
            for(int j = 0; j < n_predictor + 1; ++j)
                ip += input_weight->get(i, j) * input_output->get(j, 0);
            hidden_output->assign(i, 0, activation(ip));
        }
        hidden_output->assign(n_hidden_neuron, 0, 1);

        output_output = 0;
        for(int j = 0; j < n_hidden_neuron + 1; ++j)
            output_output += hidden_weight->get(0, j) * hidden_output->get(j, 0);
    }

    template <typename T>
    T Neural_network<T>::predict_for_row(const Matrix<T> &data, int row_ind) {
        for(int i = 0; i < n_predictor; ++i)
            input_output->assign(i, 0, data.get(row_ind, col_inds[i]));
        input_output->assign(n_predictor, 0, 1);
        forward();
        switch (loss_type) {
            case 'n':
                return output_output;
            case 'p':
                return std::log(output_output <= 0.0001 ? 0.0001 : output_output);
            case 'b':
                return output_output;
            case 't':
                return std::log(output_output <= 0.0001 ? 0.0001 : output_output);
            default:
                throw std::invalid_argument("Specified distribution does not exist.");
        }
    }

    template <typename T>
    void Neural_network<T>::predict(const Matrix<T> &data_x,
                                    Matrix<T> &prediction,
                                    const T shrinkage,
                                    const int *row_inds,
                                    int n_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #if _DEBUG_BASE_LEARNER
            assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) + shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #if _DEBUG_BASE_LEARNER
            assert(n_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < n_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) + shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }

}

namespace dbm {

    template <typename T>
    Linear_regression<T>::Linear_regression(int n_predictor, char loss_type) :
            n_predictor(n_predictor), loss_type(loss_type), Base_learner<T>('l') {
        col_inds = new int[n_predictor];
        coefs_no_intercept = new T[n_predictor];
    }

    template <typename T>
    Linear_regression<T>::~Linear_regression() {
        delete col_inds;
        delete coefs_no_intercept;
        col_inds = nullptr, coefs_no_intercept = nullptr;
    };

    template <typename T>
    T Linear_regression<T>::predict_for_row(const Matrix<T> &data, int row_ind) {
        T result = 0;
        for(int i = 0; i < n_predictor; ++i) {
            result += data.get(row_ind, col_inds[i]) * coefs_no_intercept[i];
        }
        result += intercept;
        switch (loss_type) {
            case 'n':
                return result;
            case 'p':
                return std::log(result <= 0.0001 ? 0.0001 : result);
            case 'b':
                return result;
            case 't':
                return std::log(result <= 0.0001 ? 0.0001 : result);
            default:
                throw std::invalid_argument("Specified distribution does not exist.");
        }
    }

    template <typename T>
    void Linear_regression<T>::predict(const Matrix<T> &data_x,
                                       Matrix<T> &prediction,
                                       const T shrinkage,
                                       const int *row_inds,
                                       int n_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #if _DEBUG_BASE_LEARNER
            assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) + shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #if _DEBUG_BASE_LEARNER
            assert(n_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < n_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) + shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }

}

namespace dbm {

    template<typename T>
    Tree_node<T>::Tree_node(int depth) : larger(nullptr), smaller(nullptr), column(-1),
                                         split_value(0), loss(std::numeric_limits<T>::max()),
                                         depth(depth), last_node(false), prediction(0),
                                         no_training_samples(0), Base_learner<T>('t') {}

    template<typename T>
    Tree_node<T>::Tree_node(int depth, int column, bool last_node,
                            T split_value, T loss, T prediction,
                            int no_tr_samples) : larger(nullptr), smaller(nullptr), column(column),
                                                 split_value(split_value), loss(loss),
                                                 depth(depth), last_node(last_node), prediction(prediction),
                                                 no_training_samples(no_tr_samples), Base_learner<T>('t') {}

    template<typename T>
    Tree_node<T>::~Tree_node() {

        if (this == nullptr) return;

        delete larger;
        larger = nullptr;

        delete smaller;
        smaller = nullptr;

    }

    template<typename T>
    T Tree_node<T>::predict_for_row(const Matrix<T> &data_x, int row_ind) {
        if (last_node)
            return prediction;
        if (data_x.get(row_ind, column) > split_value) {
            #if _DEBUG_BASE_LEARNER
            assert(larger != NULL);
            #endif
            return larger->predict_for_row(data_x, row_ind);
        } else {
            #if _DEBUG_BASE_LEARNER
            assert(larger != NULL);
            #endif
            return smaller->predict_for_row(data_x, row_ind);
        }
    }

    template<typename T>
    void Tree_node<T>::predict(const Matrix<T> &data_x,
                               Matrix<T> &prediction,
                               const T shrinkage,
                               const int *row_inds,
                               int n_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #if _DEBUG_BASE_LEARNER
            assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) + shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #if _DEBUG_BASE_LEARNER
            assert(n_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < n_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) + shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }
}






/* ========================
 *
 * Tools for base learners
 *
 * ========================
 */

// for global means
namespace dbm {

    template <typename T>
    void save_global_mean(const Global_mean<T> *mean, std::ofstream &out) {
        out << mean->mean << std::endl;
    }

    template <typename T>
    void load_global_mean(std::ifstream &in, Global_mean<T> *&mean) {
        std::string line;
        std::getline(in, line);

        std::string words[100];
        int count = split_into_words(line, words);

        #if _DEBUG_BASE_LEARNER
        assert(count == 1);
        #endif
        mean = new Global_mean<T>;
        mean->mean = T(std::stod(words[0]));
    }

}

//for neural networks
namespace dbm {
    
    template <typename T>
    void save_neural_network(const Neural_network<T> *neural_network, std::ofstream &out) {

        out << neural_network->n_predictor << ' '
            << neural_network->n_hidden_neuron << ' '
            << neural_network->loss_type << std::endl;

        for(int i = 0; i < neural_network->n_predictor; ++i)
            out << neural_network->col_inds[i] << ' ';
        out << std::endl;

        for(int i = 0; i < neural_network->n_hidden_neuron; ++i) {
            for(int j = 0; j < neural_network->n_predictor + 1; ++j)
                out << neural_network->input_weight->get(i, j) << ' ';
            out << std::endl;
        }

        for(int i = 0; i < neural_network->n_hidden_neuron + 1; ++i)
            out << neural_network->hidden_weight->get(0, i) << ' ';

        out << std::endl;

    }
    
    template <typename T>
    void load_neural_network(std::ifstream &in, Neural_network<T> *&neural_network) {
        std::string line;
        std::string words[500];

        std::getline(in, line);
        int count = split_into_words(line, words);
        #if _DEBUG_BASE_LEARNER
        assert(count == 3);
        #endif
        neural_network = new Neural_network<T>(std::stoi(words[0]), std::stoi(words[1]), words[2].front());

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #if _DEBUG_BASE_LEARNER
        assert(count == neural_network->n_predictor);
        #endif
        for(int i = 0; i < count; ++i)
            neural_network->col_inds[i] = std::stoi(words[i]);

        for(int i = 0; i < neural_network->n_hidden_neuron; ++i) {

            line.clear();
            std::getline(in, line);
            count = split_into_words(line, words);
            #if _DEBUG_BASE_LEARNER
            assert(count == neural_network->n_predictor + 1);
            #endif
            for(int j = 0; j < count; ++j)
                neural_network->input_weight->assign(i, j, T(std::stod(words[j])));

        }

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #if _DEBUG_BASE_LEARNER
        assert(count == neural_network->n_hidden_neuron + 1);
        #endif
        for(int i = 0; i < count; ++i)
            neural_network->hidden_weight->assign(0, i, T(std::stod(words[i])));

    }
    
}

// for linear regression
namespace dbm {

    template <typename T>
    void save_linear_regression(const Linear_regression<T> *linear_regression,
                                std::ofstream &out) {
        out << linear_regression->n_predictor << ' '
            << linear_regression->loss_type << std::endl;
        for(int i = 0; i < linear_regression->n_predictor; ++i)
            out << linear_regression->col_inds[i] << ' ';
        out << std::endl;
        for(int i = 0; i < linear_regression->n_predictor; ++i)
            out << linear_regression->coefs_no_intercept[i] << ' ';
        out << std::endl;
        out << linear_regression->intercept << std::endl;
    }

    template <typename T>
    void load_linear_regression(std::ifstream &in, Linear_regression<T> *&linear_regression) {
        std::string line;
        std::string words[500];

        std::getline(in, line);
        int count = split_into_words(line, words);
        #if _DEBUG_BASE_LEARNER
        assert(count == 2);
        #endif
        linear_regression = new Linear_regression<T>(std::stoi(words[0]), words[1].front());

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #if _DEBUG_BASE_LEARNER
        assert(count == linear_regression->n_predictor);
        #endif
        for(int i = 0; i < count; ++i)
            linear_regression->col_inds[i] = std::stoi(words[i]);

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #if _DEBUG_BASE_LEARNER
        assert(count == linear_regression->n_predictor);
        #endif
        for(int i = 0; i < count; ++i)
            linear_regression->coefs_no_intercept[i] = T(std::stod(words[i]));

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
            #if _DEBUG_BASE_LEARNER
        assert(count == 1);
            #endif
        linear_regression->intercept = T(std::stod(words[0]));
    }

}

// for trees

namespace dbm {

    template<typename T>
    void save_tree_node(const Tree_node<T> *node, std::ofstream &out) {
        if (node == nullptr) {
            out << "#" << '\n';
        } else {
            out << node->depth << ' '
                << node->column << ' '
                << node->last_node << ' '
                << node->split_value << ' '
                << node->loss << ' '
                << node->prediction << ' '
                << node->no_training_samples << ' '
                << std::endl;
            save_tree_node(node->larger, out);
            save_tree_node(node->smaller, out);
        }
    }

    template<typename T>
    bool readNextToken(int &depth,
                       int &column,
                       bool &last_node,
                       T &split_value,
                       T &loss,
                       T &prediction,
                       int &no_tr_samples,
                       std::istream &in,
                       bool &isNumber) {

        isNumber = false;

        if (in.eof()) return false;

        std::string line;
        std::getline(in, line);

        std::string words[100];
        int count = split_into_words(line, words);

        if (!count || words[0] == "==")
            return false;

        if (words[0] != "#") {
            isNumber = true;
            depth = std::stoi(words[0]);
            column = std::stoi(words[1]);
            last_node = bool(std::stoi(words[2]));
            split_value = T(std::stod(words[3]));
            loss = T(std::stod(words[4]));
            prediction = T(std::stod(words[5]));
            no_tr_samples = std::stoi(words[6]);
        }
        return true;
    }

    template<typename T>
    void load_tree_node(std::ifstream &in, Tree_node<T> *&node) {
        int depth, column, no_tr_samples;
        T split_value, loss, prediction;
        bool last_node = false;
        bool isNumber;
        if (!readNextToken(depth,
                           column,
                           last_node,
                           split_value,
                           loss,
                           prediction,
                           no_tr_samples,
                           in,
                           isNumber))
            return;
        if (isNumber) {
            node = new Tree_node<T>(depth,
                                    column,
                                    last_node,
                                    split_value,
                                    loss,
                                    prediction,
                                    no_tr_samples);
            load_tree_node(in, node->larger);
            load_tree_node(in, node->smaller);
        }
    }

    template<typename T>
    void delete_tree(Tree_node<T> *tree) {
        delete tree;
        tree = nullptr;
    }

    template<typename T>
    void print_tree_info(const dbm::Tree_node<T> *tree) {
        if (tree->last_node) {
            std::cout << "depth: " << tree->depth << ' '
                      << "column: " << tree->column << ' '
                      << "split_value: " << tree->split_value << ' '
                      << "loss: " << tree->loss << ' '
                      << "last_node: " << tree->last_node << ' '
                      << "prediction: " << tree->prediction << ' '
                      << "no_training_sample: " << tree->no_training_samples
                      << std::endl;
            std::cout << "==========" << std::endl;
            return;
        }
        std::cout << "depth: " << tree->depth << ' '
                  << "column: " << tree->column << ' '
                  << "split_value: " << tree->split_value << ' '
                  << "loss: " << tree->loss << ' '
                  << "last_node: " << tree->last_node << ' '
                  << "prediction: " << tree->prediction << ' '
                  << "no_training_sample: " << tree->no_training_samples
                  << std::endl;
        std::cout << "==========" << std::endl;
        print_tree_info(tree->larger);
        print_tree_info(tree->smaller);
    }

    template<typename T>
    void Tree_info<T>::get_depth(const dbm::Tree_node<T> *tree) {
        if (tree->last_node) {
            depth = std::max(depth, tree->depth);
            return;
        }
        get_depth(tree->larger);
        get_depth(tree->smaller);
    }

    template<typename T>
    void Tree_info<T>::fill(const dbm::Tree_node<T> *tree, int h) {

        std::ostringstream temporary;
        temporary << "(" << std::to_string(tree->depth) << ")";


        if (tree->last_node) {
            temporary << " " << tree->prediction;
            tree_nodes[h][tree->depth] = temporary.str();
            return;
        }

        temporary << " l:" << tree->loss
                  << " c:" << tree->column
                  << " v:" << tree->split_value;
        tree_nodes[h][tree->depth] = temporary.str();
        int next_higher = h - std::max(1, int(height / std::pow(2, tree->depth + 2))),
                next_lower = h + int(height / std::pow(2, tree->depth + 2));
        fill(tree->larger, next_higher);
        fill(tree->smaller, next_lower);
    }

    template<typename T>
    Tree_info<T>::Tree_info(const dbm::Tree_node<T> *tree) {

        get_depth(tree);

        height = std::pow(2, depth);

        tree_nodes = new std::string *[height];
        for (int i = 0; i < height; ++i) {
            tree_nodes[i] = new std::string[depth + 1];
        }

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < depth + 1; ++j)
                tree_nodes[i][j] = "";
        }

        fill(tree, height / 2);

    }

    template<typename T>
    Tree_info<T>::~Tree_info() {

        for (int i = 0; i < height; ++i) {
            delete[] tree_nodes[i];
        }
        delete[] tree_nodes;

    }

    template<typename T>
    void Tree_info<T>::print() const {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < depth + 1; ++j) {
                std::cout << tree_nodes[i][j] << "\t\t";
            }
            std::cout << std::endl;
        }
    }

    template<typename T>
    void Tree_info<T>::print_to_file(const std::string &file_name, const int & number) const {
        std::ofstream file(file_name.c_str(), std::ios_base::app);
        file << std::endl;
        file << "=======================  Tree "
             << number
             << "  ======================="
             << std::endl;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < depth + 1; ++j) {
                file << tree_nodes[i][j] << "\t\t";
            }
            file << std::endl;
        }
        file.close();
    }

}

namespace dbm {

    template void save_global_mean<double>(const Global_mean<double> *mean, std::ofstream &out);

    template void save_global_mean<float>(const Global_mean<float> *mean, std::ofstream &out);

    template void load_global_mean<float>(std::ifstream &in, Global_mean<float> *&mean);

    template void load_global_mean<double>(std::ifstream &in, Global_mean<double> *&mean);


    template void save_neural_network<double>(const Neural_network<double> *neural_network, std::ofstream &out);

    template void save_neural_network<float>(const Neural_network<float> *neural_network, std::ofstream &out);

    template void load_neural_network<double>(std::ifstream &in, Neural_network<double> *&neural_network);

    template void load_neural_network<float>(std::ifstream &in, Neural_network<float> *&neural_network);


    template void save_linear_regression<double>(const Linear_regression<double> *linear_regression, std::ofstream &out);

    template void save_linear_regression<float>(const Linear_regression<float> *linear_regression, std::ofstream &out);

    template void load_linear_regression<double>(std::ifstream &in, Linear_regression<double> *&linear_regression);

    template void load_linear_regression<float>(std::ifstream &in, Linear_regression<float> *&linear_regression);


    template void save_tree_node<double>(const Tree_node<double> *node, std::ofstream &out);

    template void save_tree_node<float>(const Tree_node<float> *node, std::ofstream &out);

    template void load_tree_node<double>(std::ifstream &in, Tree_node<double> *&node);

    template void load_tree_node<float>(std::ifstream &in, Tree_node<float> *&node);

    template void delete_tree<double>(Tree_node<double> *tree);

    template void delete_tree<float>(Tree_node<float> *tree);

    template void print_tree_info<double>(const dbm::Tree_node<double> *tree);

    template void print_tree_info<float>(const dbm::Tree_node<float> *tree);

}