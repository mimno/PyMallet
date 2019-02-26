import re, sys, random, math
import numpy as np
from collections import Counter
import topicmodel
from timeit import default_timer as timer

import pstats, cProfile
import pyximport
pyximport.install()

word_pattern = re.compile("\w[\w\-\']*\w|\w")

if len(sys.argv) != 3:
    print("Usage: topicmodel.py [docs file] [num topics]")
    sys.exit()

num_topics = int(sys.argv[2])
doc_smoothing = 0.5
word_smoothing = 0.01

stoplist = set()
with open("stoplists/en.txt", encoding="utf-8") as stop_reader:
    for line in stop_reader:
        line = line.rstrip()
        stoplist.add(line)

word_counts = Counter()

documents = []
word_topics = {}
topic_totals = np.zeros(num_topics, dtype=int)

for line in open(sys.argv[1], encoding="utf-8"):
    #line = line.lower()
    
    tokens = word_pattern.findall(line)
    
    ## remove stopwords, short words, and upper-cased words
    tokens = [w for w in tokens if not w in stoplist and len(w) >= 3 and not w[0].isupper()]
    word_counts.update(tokens)
    
    doc_topic_counts = np.zeros(num_topics, dtype=int)
    
    documents.append({ "original": line, "token_strings": tokens, "topic_counts": doc_topic_counts })

## Now that we're done reading from disk, we can count the total
##  number of words.

vocabulary = list(word_counts.keys())
vocabulary_size = len(vocabulary)
word_ids = { w: i for (i, w) in enumerate(vocabulary) }
smoothing_times_vocab_size = word_smoothing * vocabulary_size

word_topics = np.zeros((len(vocabulary), num_topics), dtype=int)

for document in documents:
    tokens = document["token_strings"]
    doc_topic_counts = document["topic_counts"]
    
    doc_tokens = np.ndarray(len(tokens), dtype=int)
    doc_topics = np.ndarray(len(tokens), dtype=int)
    topic_changes = np.zeros(len(tokens), dtype=int)
    
    for i, w in enumerate(tokens):
        word_id = word_ids[w]
        topic = random.randrange(num_topics)
        
        doc_tokens[i] = word_id
        doc_topics[i] = topic
        
        ## Update counts: 
        word_topics[word_id][topic] += 1
        topic_totals[topic] += 1
        doc_topic_counts[topic] += 1
    
    document["doc_tokens"] = doc_tokens
    document["doc_topics"] = doc_topics
    document["topic_changes"] = topic_changes

sampling_dist = np.zeros(num_topics, dtype=float)
topic_normalizers = np.zeros(num_topics, dtype=float)
for topic in range(num_topics):
    topic_normalizers[topic] = 1.0 / (topic_totals[topic] + smoothing_times_vocab_size)

def profile():

    model = topicmodel.TopicModel(50, len(vocabulary), doc_smoothing, word_smoothing)
    document = documents[0]

    for document in documents:
        c_doc = topicmodel.Document(document["doc_tokens"], document["doc_topics"], document["topic_changes"], document["topic_counts"])
        model.add_document(c_doc)

    #model.sample(10)
    
    #cProfile.runctx("topicmodel.sample_doc(doc_tokens, doc_topics, topic_changes, doc_topic_counts, word_topics, topic_totals, sampling_dist, topic_normalizers, doc_smoothing, word_smoothing, smoothing_times_vocab_size, num_topics)", globals(), locals(), "topics.prof")
    cProfile.runctx("model.sample(10)", globals(), locals(), "topics.prof")
    
    stats = pstats.Stats("topics.prof")
    stats.strip_dirs().sort_stats("time").print_stats()
    
    

def sample(num_iterations):
    start = timer()
    
    for iteration in range(num_iterations):
        
        for document in documents:
            
            doc_topic_counts = document["topic_counts"]
            doc_tokens = document["doc_tokens"]
            doc_topics = document["doc_topics"]
            topic_changes = document["topic_changes"]
            
            # Pass the document to the fast C code
            topicmodel.sample_doc(doc_tokens, doc_topics, topic_changes, doc_topic_counts, word_topics, topic_totals, sampling_dist, topic_normalizers, doc_smoothing, word_smoothing, smoothing_times_vocab_size, num_topics)
            
        if iteration % 10 == 0:
            end = timer()
            print(end - start)
            start = timer()

def entropy(p):
    ## make sure the vector is a valid probability distribution
    p = p / np.sum(p)
    
    result = 0.0
    for x in p:
        if x > 0.0:
            result += -x * math.log2(x)
            
    return result

def print_topic(topic):
    sorted_words = sorted(zip(word_topics[:,topic], vocabulary), reverse=True)
    
    for i in range(20):
        w = sorted_words[i]
        print("{}\t{}".format(w[0], w[1]))

def print_all_topics():
    for topic in range(num_topics):
        sorted_words = sorted(zip(word_topics[:,topic], vocabulary), reverse=True)
        print(" ".join([w for x, w in sorted_words[:20]]))

def write_state(writer):
    writer.write("Doc\tWordID\tWord\tTopic\tCounts\tChanges\n")
    
    for doc, document in enumerate(documents):
        doc_tokens = document["doc_tokens"]
        doc_topics = document["doc_topics"]
        topic_changes = document["topic_changes"]
        
        doc_length = len(doc_tokens)
        
        for i in range(doc_length):
            word_id = doc_tokens[i]
            word = vocabulary[word_id]
            topic = doc_topics[i]
            changes = topic_changes[i]
            
            writer.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(doc, word_id, word, topic, word_counts[word], changes))
        

#profile()

model = topicmodel.TopicModel(num_topics, vocabulary, doc_smoothing, word_smoothing)
document = documents[0]

for document in documents:
    c_doc = topicmodel.Document(document["doc_tokens"], document["doc_topics"], document["topic_changes"], document["topic_counts"])
    model.add_document(c_doc)

for i in range(20):
    start = timer()
    model.sample(50)
    print(timer() - start)
    model.print_all_topics()
    

#sample(1000)
#topicmodel.sample(10, documents, word_topics, topic_totals, doc_smoothing, word_smoothing, smoothing_times_vocab_size, num_topics)
#print_all_topics()
#with open("state.txt", "w") as writer:
#    write_state(writer)
