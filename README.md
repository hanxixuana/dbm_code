# dbm_code
This is a C++ implementation of Delta Boosting Machines.

Note on adding new base learners:

1. prototype in base_learner.h

2. implementation in base_learner.cpp

3. instantiation of base learner class in base_learner.cpp

4. prototypes of load_base_learner and save_base_learner in base_learner.h and tool.h

5. implementation in tools.cpp

6. instantiation of two functions in base_learner.cpp

#Simon Edit:
a.) output required: marginal influence, variable importance
b.) considered exporting tree in XML format to avoid creating thousands of files.
c.) Include at least 3-4 distributions, (normal, poisson, bernoulli, gamma, tweedie in sequence)
d.) Include at least 3-4 base learners excluding intercepts (tree, spline, svr, spline, straight line)
e.) Compare the results of the results over original DBM, GBM and GLM.


Programs to be done:

1. computing engine

2. client-server connection
#Simon Edit:
a.) allow scoring only one observation
b.) allow scoring a batch
c.) One-click process

3. client user interface
#Simon Edit:
a.) Analytics interface for data scientistics : displaying diagnostics, modeling anlayses, marginal influence, (refer to Simon DBM paper).
b.) Business Intelligence interface for actuaries: Allow investigation on a certain segmentation (e.g only age 30-35 years old analysis) 

Plan for Simon
1. To provide inputs/refinement on both modeling and interface on a weekly basis.
2. To provide mathematics derivation for DBM processing of loss functions


Plan for Xixuan in Nov

1. before Nov. 15, complete Issue 2, 4, 5, 6

2. before Nov. 30, complete Issue 1, 3

Plan for Yi in Nov

1. before Nov. 10, figure out technology choices and choose packages for status transimission (TCP?) and data file transmission (FTP?)between clients and the server

2. before Nov. 15, complete a demo Python package for client and a Python deamon for server
	
	- the client package realizes functions like starting connection, sending commands to server, receiving feedbacks from server, transmiting large files to server, receiving large files from server
	- probably multiple threads or processes are needed for status transmission and data file transmission (one thread for keep sending and receiving commands and feedbacks and another thread for transmiting large files)
	- both the client and the server sides verify the integrity of files by checking hash values
	- both sides also have the capability of and storing organizing received files

3. before Nov. 30, complete a well-functioning client-server framework

References for networking programming in Python:

1. https://pymotw.com/2/socket/tcp.html

2. https://docs.python.org/3/howto/sockets.html

3. https://www.tutorialspoint.com/python/python_networking.htm

4. http://stackoverflow.com/questions/18389076/python-client-server-send-big-size-files

5. http://stackoverflow.com/questions/20007319/how-to-do-a-large-text-file-transfer-in-python

6. http://stackoverflow.com/questions/19990974/how-to-read-large-file-socket-programming-and-python

7. http://www.bogotobogo.com/python/python_network_programming_server_client_file_transfer.php

8. http://www.jb51.net/article/48791.htm

9. http://blog.csdn.net/rebelqsp/article/details/22183981

10. http://www.linuxidc.com/Linux/2013-08/89299.htm

11. http://blog.csdn.net/yongjian_luo/article/details/37816915

12. http://blog.csdn.net/taiyang1987912/article/details/44850999

13. http://www.cnblogs.com/sunada2005/archive/2013/06/19/3144275.html


Plan for Xixuan in Dec

1. before Dec 7, complete a Python interface for the C++ computing engine

2. before Dec 11, complete the whole server side by puting the server program and the computing engine together through the interface

3. before Dec 14, join Yi and complete the client user interface

4. before Dec 20, complete testing

Plan for Yi in Dec

1. before Dec 7, choose and test a UI framework and make a demo

2. before Dec 14, complete the client user interface

3. before Dec 20, complete testing





