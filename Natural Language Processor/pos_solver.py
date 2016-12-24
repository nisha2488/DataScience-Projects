###################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
'''
1. Abstraction:
For this problem, we have taken -
(i) Words (W) in the sentence as observed states
(ii) Corresponding part of speech (S) as the hidden states
For each of the three models, we have then used the training data to calculate-
(i) The initial probability: How likely a sentence is to start with a given part of speech
Calculated as:
Initial_probability(pos), P(S_0) = Number of times a sentence starts with that POS / Total sentences
(ii) The transition probability: How likely a part of speech is to be followed by a given part of speech
Calculated as:
Transition_probability(pos1)(pos2), P(S_1|S_0) = Number of times S_0 is followed by S_1 / Total occurrences of S_0
(iii) The emission probability: Given a part of speech, how likely is the value for a given word, for example, how likely is it for the word 'air' to be a noun.
Calculated as:
Emission_probability(pos)(word), P(W|S) = Number of times given word, W occurs as POS, S / Total occurrences of the POS, S

2. Brief description of how the program works:
Step 1:
(i) In this step, we calculate the initial, transition and emission probability (as defined above) using the training data.
We also calculate the probabilities for each suffix up to length 4, i.e., how like a word is to end with a given suffix.
We do this to handle the foreign words in the test data, i.e., words not encountered in the training data.
For each word in the training data, we create 4 suffix (or less depending on the length of the word) and store it's corresponding POS.
After all the words are done, we calculate the emission probability for each unique suffix similar to the way we calculate it for the words in the training dataset.
Thus, if an unknown word is encountered in the test data, we can look up the suffix of this word and get it's corresponding emission probability.

(ii) We also calculate the probability for a given POS, for example, the probability of NOUN is simply the total occurrences of NOUN / Total words
This is required for model 2.

(iii) Also, for model 3, we need to calculate the transition probability to a state based on the previous two states.
We accomplish the same in step 1, where we check how likely a pair of states is to be followed by a given state.

Step 2:
In step 2, we implement the model given in 1b.
In this step, for a given sentence, we do the following for each word:
a. Get the emission probability for a word for a given part of speech
b. Get the probability for a given POS (as explained in point (ii) in step 1)
c. Get the product, step (a) * step (b) for all possible POS
d. Return the POS which gives the maximum product in (c)

Using the above, we label the given sentence.

Step 3:
In step 3, we implement Viterbi algorithm for the HMM given in figure 1a.
In this step, for a given sentence , we do the following-
a. We first find the initial probability for a given word for each POS
b. We then take the product of each of the above initial probability with the transition probability for all possible POS.
c. We then take the maximum of the values returned by step (b) and multiply it with the emission probability for the given word with respect to every possible POS
Step (c) will thus give us 12 values (since 12 possible POS) for a given word.
d. We then calculate the product of each of the value obtained in step c with the transition probability of the previous state to every possible POS.
We then take the maximum from these values and multiply it with the emission probability for the given word with respect to every possible POS.
e. In this way, we get results for all words as mentioned in step (d).
f. Once we have calculated possible state values for the last word, we backtrack.
g. For each state, we backtrack to see which POS resulted in the maximum value for the product described in step (d).
h. We thus return the most likely sequence at the end of backtracking.

Step 4:
In step 4, we implement the Variable elimination algorithm for the model given in figure 1c.
In this step, for a given sentence, we do the following-
a. We start with the last word and go on eliminating words from the start of the chain, say if we have 5 words, we start with S5 and start eliminating S1, S2, S3 and S4 in the given order.
Since, for the last word, we need to eliminate the words in front of it, we call the forward elimination function written in the code.
b. We store all the intermediate results obtained from eliminating each of the above variables.
c. We then multiply the emission probability for the given word with the results obtained after eliminating the remaining variables.
d. The maximum value obtained in step c determines the POS for that word.
e. We then move to the second last word, S_n-1.
We now have all the states ahead of S_n-1 to be eliminated and 1 state after S_n-1, i.e., the last state which is to be eliminated.
However, we don't actually need to call forward elimination here since we already have the results of eliminating those variables stored as intermediate results while finding the last state.
We thus can re-use these results and only call the backward elimination algorithm to eliminate the last state.
We save the results of eliminating this last state using backward elimination.
f. We then do the calculation similar to that explained in steps c and d above.
g. In this way, we go on finding the most probable POS for each word from end to the beginning of the sentence.
For each word, we re-use the tables that were cached previously by the forward elimination and the backward elimination.
We keep a track of the intermediate results separately for each variable that is being eliminated by the forward elimination and backward elimination.
Say, If a variable, S4 was eliminated while forward elimination, we keep a track of it ('fS4' dictionary).
If the same variable, S4 is encountered while backward elimination for the first time, we cache its results as well ('bS4' dictionary).
Thus, if next time we need to eliminate this variable for finding some other state, depending on the whether is eliminated using forward elimination (fS4) or backward elimination (bS4), we use the corresponding stored values for S4.
For example, if we need to eliminate S4 in a sentence with 8 words, we will use the results from forward elimination for states S1, S2 and S3.
We then use the results from backward elimination obtained previously by eliminating the states S8, S7 and S6.
We then merge the two and eliminate the last variable S5 and compute result for S4.

The above model takes a lot of time for computation since there are many boundary conditions that were handled in forward and backward elimination such as sentences with 2 words.
Separate calculation was required for calculating the probabilities for the first two states and the last twp states as evident from the dependencies in the model.
Also, there are lot of intermediate results generated when getting the POS for the last word.
All this increased the execution time for the program. Whereas the code runs pretty fast for the remaining two models.


3. Design decisions made:
a. For model 1b, a given word depends only on the current state and also each state is independent of each other.
For this, we have taken the probability of any given state (POS) as the total occurrences of that POS / total words in the training dataset.
Since the interpretation of this probability calculation was up to the team, we decided to use this method of calculating the probability for any given POS.

b. The calculation of the log of the posterior probability for all 3 models and the ground truth is based on the model 1a.
We have done the factorization based on this model as-
P(Initial state) * (Product of transition probabilities P(S_i+1|S_i)) * (Product of emission probabilities, P(W|S))

4. Accuracy for each technique
The code returns pretty good results for HMM, as good as 100% for the tiny data set and 96.11% for words for the larger dataset, bc.test and 61.55% for sentences.
The second best results are returned by Simplified model, with 94.21% for words and 61.55% for sentences.
Surprisingly, complex model does not return that good results with 90.53% accuracy for words and 35.50% for sentences.

Thus, as per our analysis it seems that Viterbi works best for labelling a given sentence.
Since the complex model is not a HMM, we have used variable elimination (forward-backward algorithm) which is not returning that good results.
However, it seems that implementing it with something similar to Viterbi which takes previous two states into consideration would have given better results.

'''

####

import random
import math
import operator
import string

debug = 0

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

# train_data = read_data("bc.train.txt")
# tot_sentences = len(train_data)

class Solver:

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling

    def __init__(self):
        self.initial_prob = {}
        self.pos_prob = {}
        self.pos_tot_prob = {}
        self.suff_pos_prob ={}
        self.emission_prob = {}
        self.suff_emission_prob = {}
        self.trans_deno_prob = {}
        self.transition_prob = {}
        self.long_trans_deno_prob = {}
        self.long_transition_prob = {}
        self.tot_sentences = 0
        self.trained_words = {}
        self.tables = {}

    def posterior(self, sentence, label):
        # print sentence
        # print label[0], label[1]
        # print self.initial_prob[label[0]]
        initial_prob = math.log(self.initial_prob[label[0]])
        transition_sum = 0
        emission_sum = 0
        for i in range(0, (len(sentence) - 1)):
            if label[i+1] in self.transition_prob[label[i]]:
                transition_sum += math.log(self.transition_prob[label[i]][label[i+1]])
        for i in range(0, len(sentence)):
            emission_prob = self.get_emission_probability(sentence, i, label[i])
            if emission_prob != 0:
                emission_sum += math.log(emission_prob)

        # print initial_prob + transition_sum + emission_sum
        return initial_prob + transition_sum + emission_sum

    # Do the training!
    #
    def train(self, data):
        # pass
        self.tot_sentences = len(data)
        for s in data:
            # print len(s[0])
            words = s[0]
            pos = s[1]
            new_words = []
            new_pos = []

            # Storing all the words found in training data
            for word in words:
                if word not in self.trained_words:
                    self.trained_words[word] = word

            # Calculating the initial probability for each part of speech
            if pos[0] not in self.initial_prob:
                self.initial_prob[pos[0]] = 1
            else:
                self.initial_prob[pos[0]] += 1

            # Calculating each POS's probability
            for col in range(0, len(pos)):
                if pos[col] not in self.pos_prob:
                    self.pos_prob[pos[col]] = 1
                else:
                    self.pos_prob[pos[col]] += 1


            # Calculating emission probability
            for col in range(0, len(pos)):
                if pos[col] not in self.emission_prob:
                    self.emission_prob[pos[col]] = {}
                if words[col] not in self.emission_prob[pos[col]]:
                    self.emission_prob[pos[col]][words[col]] = 1
                else:
                    self.emission_prob[pos[col]][words[col]] += 1

            # Calculating the transition probabilities
            for col in range(0, len(pos) - 1):
                # For transition prob, last word is not counted for denominator
                if pos[col] not in self.trans_deno_prob:
                    self.trans_deno_prob[pos[col]] = 1
                else:
                    self.trans_deno_prob[pos[col]] += 1
                next_col = col + 1
                if pos[col] not in self.transition_prob:
                    self.transition_prob[pos[col]] = {}
                if pos[next_col] not in self.transition_prob[pos[col]]:
                    self.transition_prob[pos[col]][pos[next_col]] = 1
                else:
                    self.transition_prob[pos[col]][pos[next_col]] += 1

            # Calculating the longer transition probabilities
            for col in range(0, len(pos) - 2):
                next_col1 = col + 1
                next_col2 = col + 2
                # For longer transition prob, last two words are not counted for denominator
                if pos[col] not in self.long_trans_deno_prob:
                    self.long_trans_deno_prob[pos[col]] = {}
                if pos[next_col1] not in self.long_trans_deno_prob[pos[col]]:
                    self.long_trans_deno_prob[pos[col]][pos[next_col1]] = 1
                else:
                    self.long_trans_deno_prob[pos[col]][pos[next_col1]] += 1

                if pos[col] not in self.long_transition_prob:
                    self.long_transition_prob[pos[col]] = {}
                if pos[next_col1] not in self.long_transition_prob[pos[col]]:
                    self.long_transition_prob[pos[col]][pos[next_col1]] = {}
                if pos[next_col2] not in self.long_transition_prob[pos[col]][pos[next_col1]]:
                    self.long_transition_prob[pos[col]][pos[next_col1]][pos[next_col2]] = 1
                else:
                    self.long_transition_prob[pos[col]][pos[next_col1]][pos[next_col2]] += 1

            # Creating new words and their POS from given word's suffixes
            for col in range(0, len(pos)):
                # print "Word:" + str(words[col])
                # print "POS:" + str(pos[col])
                n = len(words[col]) - 1
                i = n - 1
                w = words[col][n]
                count = 4
                while i >= 0 and count > 0:
                    w = words[col][i] + w
                    new_words.append(w)
                    new_pos.append(pos[col])
                    i -= 1
                    count -= 1
                    # print "New word:" + w
                    # print "New pos:" + pos[col]

            # Calculating each POS's probability for the suffixes
            for col in range(0, len(new_pos)):
                if new_pos[col] not in self.suff_pos_prob:
                    self.suff_pos_prob[new_pos[col]] = 1
                else:
                    self.suff_pos_prob[new_pos[col]] += 1

            # # Calculating emission probability for suffixes and POS
            # for col in range(0, len(new_pos)):
            #     if new_pos[col] not in self.suff_emission_prob:
            #         self.suff_emission_prob[new_pos[col]] = {}
            #     if new_words[col] not in self.suff_emission_prob[new_pos[col]]:
            #         self.suff_emission_prob[new_pos[col]][new_words[col]] = 1
            #     else:
            #         self.suff_emission_prob[new_pos[col]][new_words[col]] += 1

            # Calculating emission probability for suffixes and POS
            for col in range(0, len(new_words)):
                if new_words[col] not in self.suff_emission_prob:
                    self.suff_emission_prob[new_words[col]] = {}
                if new_pos[col] not in self.suff_emission_prob[new_words[col]]:
                    self.suff_emission_prob[new_words[col]][new_pos[col]] = 1
                else:
                    self.suff_emission_prob[new_words[col]][new_pos[col]] += 1

        for pos in self.initial_prob:
            self.initial_prob[pos] = self.initial_prob[pos] / float(self.tot_sentences)

        for state in self.pos_prob:
            for next_state in self.pos_prob:
                # print state, next_state
                if next_state in self.transition_prob[state]:
                    self.transition_prob[state][next_state] /= float(self.trans_deno_prob[state])

        for state in self.pos_prob:
            for next_state1 in self.pos_prob:
                if next_state1 in self.long_transition_prob[state]:
                    for next_state2 in self.pos_prob:
                        if next_state2 in self.long_transition_prob[state][next_state1]:
                            self.long_transition_prob[state][next_state1][next_state2] /= float(self.long_trans_deno_prob[state][next_state1])

        for state in self.emission_prob:
            for word in self.emission_prob[state]:
                self.emission_prob[state][word] /= float(self.pos_prob[state])

        total_pos = sum(self.pos_prob.itervalues())
        for state in self.pos_prob:
            self.pos_tot_prob[state] = (self.pos_prob[state])/float(total_pos)

        # for state in self.suff_emission_prob:
        #     for word in self.suff_emission_prob[state]:
        #         self.suff_emission_prob[state][word] /= float(self.suff_pos_prob[state])

        for word in self.suff_emission_prob:
            total_occ = sum(self.suff_emission_prob[word].itervalues())
            for state in self.suff_emission_prob[word]:
                self.suff_emission_prob[word][state] /= float(total_occ)

    # Functions for deriving the emission probability for trained and missing words
    def get_emission_probability(self, sentence, i, curr_pos):
        if sentence[i] in self.emission_prob[curr_pos]:
            return self.emission_prob[curr_pos][sentence[i]]

            # elif sentence[i][0] == sentence[i][0].upper() and curr_pos == 'noun':
            # word_pos_prob[idx][curr_pos] = max_prev_prob * 1

        elif sentence[i] not in self.trained_words:
            n = len(sentence[i]) - 1
            prev_letter = n - 1
            new_suff = sentence[i][n]
            count = 4
            max_prob_suff = 0
            while prev_letter >= 0 and count > 0:
                new_suff = sentence[i][prev_letter] + new_suff
                if new_suff in self.suff_emission_prob and curr_pos in self.suff_emission_prob[new_suff]:
                    new_prob_suff = self.suff_emission_prob[new_suff][curr_pos]
                else:
                    new_prob_suff = 0
                if new_prob_suff >= max_prob_suff:
                    max_prob_suff = new_prob_suff

                prev_letter -= 1
                count -= 1

            return max_prob_suff
        else:
            return 0


    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        result_pos = []
        result_marginal_prob = []

        for i in range(0, len(sentence)):
            max_prob = 0
            max_pos = ''
            for pos in self.initial_prob:
                new_prob = self.pos_tot_prob[pos] * self.get_emission_probability(sentence, i, pos)
                if new_prob >= max_prob:
                    max_prob = new_prob
                    max_pos = pos
            result_pos.append(max_pos)
            result_marginal_prob.append(round(max_prob,2))

        return [[result_pos], [result_marginal_prob]]
        # return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]

    def hmm(self, sentence):
        # print "Words in sentence in hmm: "
        # print len(sentence)
        max_prob_pos = {}
        word_pos_prob = {}
        word_pos_prob['w0'] = {}
        if debug == 1:
            print sentence

        # Calculate the probability for the first word of the sentence
        for pos in self.initial_prob:
            word_pos_prob['w0'][pos] = self.initial_prob[pos] * self.get_emission_probability(sentence, 0, pos)

        # Calculate the probability for the remaining words of the sentence
        for i in range(1, len(sentence)):
            idx = 'w'+str(i)
            prev_idx = 'w' + str(i-1)
            if debug == 1:
                print "Idx:" + str(idx)
                print "Prev_idx:" + str(prev_idx)
            word_pos_prob[idx] = {}
            max_prob_pos[idx] = {}
            # Checking the word for each POS
            for curr_pos in self.initial_prob:
                max_prev_prob = 0
                max_prev_pos = ''

                # Checking transition and previous probability for above selected POS
                for prev_pos in self.initial_prob:
                    if curr_pos in self.transition_prob[prev_pos]:
                        new_prob = word_pos_prob[prev_idx][prev_pos] * self.transition_prob[prev_pos][curr_pos]
                    else:
                        new_prob = 0
                    if new_prob >= max_prev_prob:
                        max_prev_prob = new_prob
                        max_prev_pos = prev_pos

                word_pos_prob[idx][curr_pos] = max_prev_prob * self.get_emission_probability(sentence, i, curr_pos)

                if debug == 1:
                    print "For word " + str(sentence[i]) + " and curr_pos:" + str(curr_pos)
                    print max_prev_pos
                max_prob_pos[idx][curr_pos] = max_prev_pos

        # Backtracking to find the MAP inference
        if debug == 1:
            print max_prob_pos
        map_inference = []
        # last_pos = []
        idx = 'w' + str((len(sentence) - 1))
        pos = max(word_pos_prob[idx].iteritems(), key=operator.itemgetter(1))[0]
        map_inference.append(pos)
        for i in reversed(range(len(sentence) - 1)):
            prev_pos = []
            if debug == 1:
                print idx, pos
            prev_pos.append(max_prob_pos[idx][pos])
            map_inference = prev_pos + map_inference
            idx = 'w' + str(i)
            pos = prev_pos[0]

        return [[map_inference], []]
        # return [ [ [ "noun" ] * len(sentence)], [] ]

    def forward_elimination(self, sentence, word):
        # print "In forward elimination: " + str(sentence[word])
        max_prob = 0
        max_pos = ''
        fwd_w = 0

        while fwd_w < word:
            # fwd_w is the state to be eliminated

            if fwd_w == 0:
                for s2 in self.initial_prob:
                    for s1 in self.initial_prob:
                        prob_sum = 0
                        for s0 in self.initial_prob:
                            if s1 in self.transition_prob[s0] and s1 in self.long_transition_prob[s0] and s2 in \
                                    self.long_transition_prob[s0][s1]:
                                prob_sum += self.initial_prob[s0] * self.get_emission_probability(sentence, fwd_w, s0) * \
                                            self.transition_prob[s0][s1] * self.long_transition_prob[s0][s1][s2]
                        # Create intermediate result table for the variable to be eliminated
                        idx = 'fs0'
                        if idx not in self.tables:
                            self.tables[idx] = {}
                        if s1 not in self.tables[idx]:
                            self.tables[idx][s1] = {}
                        if s2 not in self.tables[idx][s1]:
                            self.tables[idx][s1][s2] = prob_sum
                        else:
                            self.tables[idx][s1][s2] = prob_sum
                        # print "For word " + str(sentence[fwd_w]) + " prob_sum = " + str(
                        #         prob_sum) + " for s1: " + str(s1) + " for s2:" + str(s2)
                # print "for word 0:" + str(self.tables['fs0'])

            elif fwd_w > 0 and fwd_w < (len(sentence) - 2):
                prev_idx = 'fs' + str(fwd_w - 1)
                # print "Previous index" + str(prev_idx)
                # print "Prev idx:" + str(prev_idx)
                for s2 in self.initial_prob:
                    for s1 in self.initial_prob:
                        prob_sum = 0
                        for s0 in self.initial_prob:
                            if s1 in self.long_transition_prob[s0] and s2 in self.long_transition_prob[s0][s1]:
                                prob_sum += self.get_emission_probability(sentence, fwd_w, s0) * \
                                            self.long_transition_prob[s0][s1][s2] * self.tables[prev_idx][s0][s1]
                        # idx = 's' + str(fwd_w - 1) + 's' + str(fwd_w + 2)
                        idx = 'fs' + str(fwd_w)
                        # print idx
                        if idx not in self.tables:
                            self.tables[idx] = {}
                        if s1 not in self.tables[idx]:
                            self.tables[idx][s1] = {}
                        if s2 not in self.tables[idx][s1]:
                            self.tables[idx][s1][s2] = prob_sum
                        else:
                            self.tables[idx][s1][s2] = prob_sum
                        # print "For word " + str(sentence[fwd_w]) + " prob_sum = " + str(prob_sum) + " for s1: " + str(s1) + " for s2:"+ str(s2)
                        # print "For idx: " + str(idx)
                        # print str(self.tables[idx])

            elif fwd_w == (len(sentence) - 2):
                prev_idx = 'fs' + str(fwd_w - 1)
                # print "Prev idx:" + str(prev_idx)
                for s1 in self.initial_prob:
                    prob_sum = 0
                    for s0 in self.initial_prob:
                        prob_sum += self.get_emission_probability(sentence, fwd_w, s0) * self.tables[prev_idx][s0][s1]
                    new_prob = self.get_emission_probability(sentence, word, s1) * prob_sum
                    if new_prob >= max_prob:
                        max_prob = new_prob
                        max_pos = s1
                # print "For word " + str(sentence[fwd_w]) + " prob_sum = " + str(
                #         prob_sum) + " for s1: " + str(s1) + " for s2:" + str(s2)

            fwd_w += 1

        # return final_tables
        return max_prob, max_pos

    def backward_elimination(self, sentence, word):
        # print "In backward elimination: " + str(sentence[word])
        if debug == 1:
            print "For word:" + str(sentence[word])
        # final_tables = {}
        # final_tables = []
        max_prob = 0
        max_pos = ''
        bckd_w = len(sentence) - 1
        if debug == 1:
            print "Word and backward word index:" + str(word) + ' ' + str(bckd_w)
        while bckd_w > word:
            # bck_switch = 1
            # bckd_w is the state to be eliminated
            if bckd_w == len(sentence) - 1 and bckd_w - word != 1:
                if debug == 1:
                    print "In first if"
                idx = 'bs' + str(bckd_w)
                if idx not in self.tables:
                    for s5 in self.initial_prob:
                        for s6 in self.initial_prob:
                            prob_sum = 0
                            for s7 in self.initial_prob:
                                if s6 in self.long_transition_prob[s5] and s7 in self.long_transition_prob[s5][s6]:
                                    prob_sum += self.get_emission_probability(sentence, bckd_w, s7) * \
                                                self.long_transition_prob[s5][s6][s7]
                            if idx not in self.tables:
                                self.tables[idx] = {}
                            if s6 not in self.tables[idx]:
                                self.tables[idx][s6] = {}
                            if s5 not in self.tables[idx][s6]:
                                self.tables[idx][s6][s5] = prob_sum
                            else:
                                self.tables[idx][s6][s5] = prob_sum

            elif bckd_w > (word + 1):
                if debug == 1:
                    print "In second if"
                idx = 'bs' + str(bckd_w)
                next_idx = 'bs' + str(bckd_w + 1)
                if idx not in self.tables:
                    for s5 in self.initial_prob:
                        for s6 in self.initial_prob:
                            prob_sum = 0
                            for s7 in self.initial_prob:
                                if s6 in self.long_transition_prob[s5] and s7 in self.long_transition_prob[s5][s6]:
                                    prob_sum += self.get_emission_probability(sentence, bckd_w, s7) * \
                                                self.long_transition_prob[s5][s6][s7] * self.tables[next_idx][s7][s6]
                            # idx = 's' + str(fwd_w - 1) + 's' + str(fwd_w + 2)
                            if idx not in self.tables:
                                self.tables[idx] = {}
                            if s6 not in self.tables[idx]:
                                self.tables[idx][s6] = {}
                            if s5 not in self.tables[idx][s6]:
                                self.tables[idx][s6][s5] = prob_sum
                            else:
                                self.tables[idx][s6][s5] = prob_sum

            elif bckd_w - word == 1:
                if debug == 1:
                    print "In third if"
                next_idx = 'bs' + str(bckd_w + 1)
                prev_idx = 'fs' + str(word - 1)
                if debug == 1:
                    print "In backward, prev and new idx:" + str(prev_idx) + ' ' + str(next_idx)
                for s5 in self.initial_prob:
                    prob_sum = 0
                    for s4 in self.initial_prob:
                        if (word + 1) == len(sentence) - 1:
                            prob_sum += self.get_emission_probability(sentence, bckd_w, s4) * self.tables[prev_idx][s4][s5]
                        elif (word - 1) > -1:
                            prob_sum += self.get_emission_probability(sentence, bckd_w, s4) * self.tables[next_idx][s5][s4] * \
                                        self.tables[prev_idx][s4][s5]
                        elif s4 in self.transition_prob[s5]:
                            prob_sum += self.get_emission_probability(sentence, bckd_w, s4) * self.tables[next_idx][s5][s4] * \
                                        self.transition_prob[s5][s4] # * self.initial_prob[s5]
                        if word == 0:
                            new_prob = self.initial_prob[s5] * self.get_emission_probability(sentence, word, s5) * prob_sum
                        else:
                            new_prob = self.get_emission_probability(sentence, word, s5) * prob_sum

                        if new_prob >= max_prob:
                            max_prob = new_prob
                            max_pos = s5
            bckd_w -= 1

        return max_prob, max_pos
        # return final_tables

    def complex(self, sentence):
        self.tables = {}
        result_pos = []
        result_marginal_prob = []

        if len(sentence) <= 2:
            return self.hmm(sentence)
        else:
            for word in reversed(range(len(sentence))):
                state_prob = []
                state_pos = []
                if word == len(sentence) - 1: # or len(sentence) == 2:
                    # print word
                    max_prob, max_pos = self.forward_elimination(sentence, word)
                    # print "first if: " + str(self.tables)
                else:
                    # print word
                    max_prob, max_pos = self.backward_elimination(sentence, word)

                state_prob.append(round(max_prob,2))
                state_pos.append(max_pos)
                # print word, max_pos

                result_marginal_prob = state_prob + result_marginal_prob
                result_pos = state_pos + result_pos
                # print result_pos, result_marginal_prob

            return [[result_pos], [result_marginal_prob]]
        # return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labellings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"
