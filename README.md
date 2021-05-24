# NTM
**Implement TSP using NTM**

**Abstract**
Neural Networks have been used to solve a lot of complex problems in today’s era. Following the prevalent trend, I used recurrent neural networks that was engaged in Neural Turing Machines as my neural program learning architecture to solve certain algorithmic task. In my project, the algorithmic task that I tried to solve was traveling salesman problem. TSP is a np-hard problem which is a class of the most complex problems in the algorithm world. I used supervised learning along with the Neural Turing Machines which involves training the network by feeding it input and output and train the network along the way. An NTM is a differentiable computer trainable by gradient descent, yielding a practical mechanism for learning programs. It is an efficient way to solve this algorithmic task as it has very close resemblance to models of working memory because its architecture has an attentional process to read and write to memory. In comparison to other models of working memory, NTM architecture can learn to use its working memory instead of deploying a fixed set of procedures over data. Additionally, TSP is a really interesting problem to work on and solve. The link for the Github repository is included in the report.

**Requirements**
Tensorflow v2
Numpy

**Inspiration**
The implementation is inspired by the implementation of snowkylin/ntm(https://github.com/snowkylin/ntm) and Mark Collier's implementation of NTM(https://github.com/MarkPKCollier/NeuralTuringMachine). 

**Steps to run**
1. Clone the directory.

2. Modify the Controller parameters in config.json(RNN size and RNN layers)

3. Modify Max Sequence length, test Sequence length and vector dim to the sequence length for which you want to train the application.

4. Run Task.py with 'train' as mode in config to train the data.

5. Run Task.py with 'test' as mode in config to test the data.
