//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_BASE_LEARNER_H
#define DBM_CODE_BASE_LEARNER_H

#ifndef _DEBUG_BASE_LEARNER
#define _DEBUG_BASE_LEARNER 1
#endif

#include "matrix.h"

#include <string>

namespace dbm {

    template<typename T>
    class Tree_node;

    template<typename T>
    class Tree_trainer;

    template<typename T>
    class Tree_info;

    template<typename T>
    void save_tree_node(const Tree_node<T> *node, std::ofstream &out);

    template<typename T>
    void load_tree_node(std::ifstream &in, Tree_node<T> *&tree);

    template<typename T>
    void delete_tree(Tree_node<T> *tree);

    template<typename T>
    void print_tree_info(const dbm::Tree_node<T> *tree);

    template<typename T>
    class Base_learner {
    protected:
        char learner_type;

        virtual T predict_for_row(const Matrix<T> &data, int row_ind) = 0;

    public:
        Base_learner(const char &type) : learner_type(type) {};

        char get_type() const { return learner_type; };

        virtual void
        predict(const Matrix<T> &data, Matrix<T> &prediction, const int *row_inds = NULL, int n_rows = 0) = 0;
    };

    template<typename T>
    class Tree_node : public Base_learner<T> {
    private:

        Tree_node *larger;
        Tree_node *smaller;

        int depth;

        int column;
        T split_value;
        T loss;

        bool last_node;
        T prediction;

        int no_training_samples;

        T predict_for_row(const Matrix<T> &data_x, int row_ind);

    public:

        Tree_node(int depth);

        Tree_node(int depth, int column, bool last_node, T split_value,
                  T loss, T prediction, int no_tr_samples);

        ~Tree_node();

        void predict(const Matrix<T> &data_x, Matrix<T> &prediction, const int *row_inds = NULL, int n_rows = 0);

        friend void save_tree_node<>(const Tree_node<T> *node, std::ofstream &out);

        friend void load_tree_node<>(std::ifstream &in, Tree_node<T> *&tree);

        friend void delete_tree<>(Tree_node<T> *tree);

        friend void print_tree_info<>(const Tree_node<T> *tree);

        friend class Tree_trainer<T>;

        friend class Tree_info<T>;

    };

}

#endif //DBM_CODE_BASE_LEARNER_H
