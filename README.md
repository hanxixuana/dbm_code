# dbm_code
This is a C++ implementation of Delta Boosting Machines.

Note on adding new base learners:
1. prototype in base_learner.h
2. implementation in base_learner.cpp
3. instantiation of base learner class in base_learner.cpp
4. prototypes of load_base_learner and save_base_learner in base_learner.h and tool.h
5. implementation in tools.cpp
6. instantiation of two functions in base_learner.cpp
