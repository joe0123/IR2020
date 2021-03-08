import argparse
import os
import numpy as np
import pandas as pd
from utils import *

K = 1.6
B = 0.75
R_n = 5
R_thres = 0.75
R_iter = 1
ALPHA = 1
BETA = 0.6
GAMMA = 0.1

def ranking(q, d, row_doc):
    sim = np.dot(d, q.T).flatten()
    #sim = np.dot(d, q.T) / np.sqrt(np.sum(d ** 2, axis=1) * np.sum(q ** 2))
    tmp = sorted(zip(sim, range(sim.shape[0])), reverse=True)
    return [i[1] for i in tmp]

def rrf(q, dr, dnr):
    return ALPHA * q + BETA * np.mean(dr, axis=0) - GAMMA * np.mean(dnr, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", action="store_true", default=False)
    parser.add_argument("-i", type=str, help="input query file", required=True)
    parser.add_argument("-o", type=str, help="output ranked list file", required=True)
    parser.add_argument("-m", type=str, help="model dir", required=True)
    parser.add_argument("-d", type=str, help="NTCIR dir", required=True)
    parser.add_argument("-c", type=str, help="train or test?", required=True, choices=["train", "test"])
    args = parser.parse_args()
    
    print("Loading Vocab...", flush=True)
    vocab_index = load_vocab(os.path.join(args.m, "vocab.all")) # vocab_index: {voc: index}
    print("Loading Files...", flush=True)
    doc_list, avg_d = load_doc(os.path.join(args.m, "file-list"), args.d)   # doc_list: [(doc_name, doc_len)]
    print("Loading Terms...", flush=True)
    inverted_file, term_dict = load_term(os.path.join(args.m, "inverted-file")) # term_dict: {term: (offset, raw_df}}
    print("Loading Queries...", flush=True)
    queries, queries_len = load_queries(args.i, vocab_index, term_dict)  # queries: [{term: raw_tf}] # queries_len: [q_len]
    
    results = []
    for i in range(len(queries)):
        print("Retrieving files for {}...".format(i), flush=True)
        q, d, row_doc = to_vec(term_dict, doc_list, inverted_file, avg_d, queries_len[i], queries[i], K, B) # row_doc: [doc_name]
        result = ranking(q, d, row_doc)
        if args.r:
            for j in range(R_iter):
                q = rrf(q, d[result[:R_n]], d[result[-10:]])
                result = ranking(q, d, row_doc)
        results.append(' '.join([row_doc[i] for i in result[:100]]))
    
    print("Outputing results...", flush=True)
    df = pd.DataFrame(columns=["query_id", "retrieved_docs"])
    if args.c == "train":
        df["query_id"] = ['0{}'.format(str(i)) for i in range(1, 11)]
    else:
        df["query_id"] = ['0{}'.format(str(i)) for i in range(11, 31)]
    df["retrieved_docs"] = results
    df.to_csv(args.o, index=False)
