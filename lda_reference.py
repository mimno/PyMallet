"""

Topic models look for groups of words that occur frequenty together. We can often recognize these clusters as specific themes that appear in the collection -- thus the term "topic" model.

Our example corpus today is a collection of Viking sagas. Start python like this:

    python -i topicmodel.py sagas_en.txt 20

We will work at the python prompt ">>>".

Today we'll be working with the simplest and most reliable topic model algorithm, Gibbs sampling.
Gibb sampling is a way to take a very complicted optimization problem and break it into little problems that are individually easy.

First, we need to have a way of describing probability distributions.
A discrete distribution is a vector of numbers that are >= 0.0 and sum to 1.0.
One function is called *entropy*. Entropy takes a distribution and returns a number.

1. Run `entropy(np.array([0.7, 0.1, 0.2]))`. What is the value?

[Response here]

2. Run `entropy(np.array([7, 1, 2]))`. Does the value change? Why or why not?

[Response here]

3. Try different (non-negative) values of the three numbers. What is the largest value you can get, and what is the smallest?

[Response here]

4. Now try different (non-negative) values of *four* numbers. Can you get a larger or smaller entropy than with three?

[Response here]

5. Describe in your own words what entropy is measuring.

[Response here]

The Gibbs sampling algorithm proceeds in multiple iterations. In each iteration, 
we look at all the word tokens in all the documents, one after another.
For each word, we erase its current topic assignment and sample a new topic 
assignment given all the other word tokens' topic assignments.

Now look at the lines below the "SAMPLING DISTRIBUTION" comment. These define two vectors:
* The probability of each topic in the current document
* The probability of the current word in each topic

We'll look at a particular dramatic moment in Njal's saga. Define these variables:

    document = documents[1160]
    doc_topic_counts = document["topic_counts"]
    word = "sword"
    word_topic_counts = word_topics[word]

Use this command to suppress scientific notation:

    np.set_printoptions(suppress=True)

6. Calculate the entropy of `doc_topic_counts`

7. Calculate the entropy of `(doc_topic_counts + doc_smoothing)`. Should this be larger or smaller than the previous value?

8. Calculate the entropy of `(word_topic_counts + word_smoothing) / (topic_totals + smoothing_times_vocab_size)`

9. Calculate the entropy of `(doc_topic_counts + doc_smoothing) * (word_topic_counts + word_smoothing) / (topic_totals + smoothing_times_vocab_size)`

These values are random initializations. Let's run the algorithm
over the documents a few times and see what happens. Run:

    sample(25)

Use `print_all_topics()` to get a view of the current state of the topics.

10. This function prints the number of tokens in each topic for the sample doc. Describe how (if at all) they change.

11. Recalculate the four entropies we calculated above for the sampling distribution. How are they different?

12. What is the value of `word_smoothing`? Previously we added 1.0 in this situation. Why are we using a different value now? Use the concept of entropy in your answer.

[Response here]

13. What are Norse sagas about, from the perspective of the model?

[Response here]

14. I'm removing a list of frequent words, words that are too short, and
words whose first letter is capitalized. Why does removing capitalized words
help? What happens if you remove that check? Is this a good idea?

[Response here]

"""

import re, sys, random, math
import numpy as np
from collections import Counter
from timeit import default_timer as timer

word_pattern = re.compile("\w[\w\-\']*\w|\w")

if len(sys.argv) != 3:
    print("Usage: topicmodel.py [docs file] [num topics]")
    sys.exit()

num_topics = int(sys.argv[2])
doc_smoothing = 0.5
word_smoothing = 0.01

stoplist = set()
with open("stoplist.txt", encoding="utf-8") as stop_reader:
    for line in stop_reader:
        line = line.rstrip()
        stoplist.add(line)

word_counts = Counter()

documents = []
word_topics = {}
topic_totals = np.zeros(num_topics)

for line in open(sys.argv[1], encoding="utf-8"):
    #line = line.lower()
    
    tokens = word_pattern.findall(line)
    
    ## remove stopwords, short words, and upper-cased words
    tokens = [w for w in tokens if not w in stoplist and len(w) >= 3 and not w[0].isupper()]
    word_counts.update(tokens)
    
    doc_topic_counts = np.zeros(num_topics)
    token_topics = []
    
    for w in tokens:
        
        ## Generate a topic randomly
        topic = random.randrange(num_topics)
        token_topics.append({ "word": w, "topic": topic })
        
        ## If we haven't seen this word before, initialize it
        if not w in word_topics:
            word_topics[w] = np.zeros(num_topics)
        
        ## Update counts: 
        word_topics[w][topic] += 1
        topic_totals[topic] += 1
        doc_topic_counts[topic] += 1
    
    documents.append({ "original": line, "token_topics": token_topics, "topic_counts": doc_topic_counts })

## Now that we're done reading from disk, we can count the total
##  number of words.
vocabulary = list(word_counts.keys())
vocabulary_size = len(vocabulary)

smoothing_times_vocab_size = word_smoothing * vocabulary_size

def sample(num_iterations):
    for iteration in range(num_iterations):
        
        start = timer()
        
        for document in documents:
            
            doc_topic_counts = document["topic_counts"]
            token_topics = document["token_topics"]
            doc_length = len(token_topics)
            for token_topic in token_topics:
                
                w = token_topic["word"]
                old_topic = token_topic["topic"]
                word_topic_counts = word_topics[w]
                
                ## erase the effect of this token
                word_topic_counts[old_topic] -= 1
                topic_totals[old_topic] -= 1
                doc_topic_counts[old_topic] -= 1
                
                ###
                ### SAMPLING DISTRIBUTION
                ###
                
                ## Does this topic occur often in the document?
                topic_probs = (doc_topic_counts + doc_smoothing) / (doc_length + num_topics * doc_smoothing)
                ## Does this word occur often in the topic?
                topic_probs *= (word_topic_counts + word_smoothing) / (topic_totals + smoothing_times_vocab_size)
                
                ## sample from an array that doesn't sum to 1.0
                sample = random.uniform(0, np.sum(topic_probs))
                
                new_topic = 0
                while sample > topic_probs[new_topic]:
                    sample -= topic_probs[new_topic]
                    new_topic += 1
                
                ## add back in the effect of this token
                word_topic_counts[new_topic] += 1
                topic_totals[new_topic] += 1
                doc_topic_counts[new_topic] += 1
                
                token_topic["topic"] = new_topic
        end = timer()
        print(end - start)
                

def entropy(p):
    ## make sure the vector is a valid probability distribution
    p = p / np.sum(p)
    
    result = 0.0
    for x in p:
        if x > 0.0:
            result += -x * math.log2(x)
            
    return result

def print_topic(topic):
    sorted_words = sorted(vocabulary, key=lambda w: word_topics[w][topic], reverse=True)
    
    for i in range(20):
        w = sorted_words[i]
        print("{}\t{}".format(word_topics[w][topic], w))

def print_all_topics():
    for topic in range(num_topics):
        sorted_words = sorted(vocabulary, key=lambda w: word_topics[w][topic], reverse=True)
        print(" ".join(sorted_words[:20]))

sample(100)