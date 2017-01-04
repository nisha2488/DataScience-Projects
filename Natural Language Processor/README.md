Implemented part-of-speech tagging with the goal of associating every word in a sentence with its part of speech. Modelled the part-of-speech problem into three different Bayesian Networks as shown below. 

![Alt text](Models.png?raw=true "Title")

Implemented the above using three different techniques - Naive Bayes, Viterbi algorithm and Variable elimination

label.py, is the main program, pos scorer.py has the scoring code and pos solver.py contains the actual part-of-speech estimation code. The main code for the above three algorithms is included in this code file, pos_solver.py

Command to run the program:
python label.py bc.train bc.test