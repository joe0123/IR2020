import os
import numpy as np
import xml.etree.ElementTree as ET


def load_vocab(fn):
    vocab_index = {}
    with open(fn, 'r', encoding='utf8') as rf:
        for voc in rf.readlines():
            voc = voc.strip()
            vocab_index[voc] = len(vocab_index)
    return vocab_index  # vocab_index: {voc: index}


def load_doc(fn, dn):
    doc_list = []
    total_len = 0
    with open(fn, 'r') as rf:
        for f in rf.readlines():
            f = f.strip()
            text_len = int(os.stat(os.path.join(dn, f)).st_size / 3.5)
            doc_list.append((f.split('/')[-1].lower(), text_len))
            total_len += text_len
    return doc_list, total_len / len(doc_list)    # doc_list: [(doc_name, doc_len)]


def load_term(fn):
    term_dict = {}
    with open(fn, 'r') as rf:
        inverted_file = rf.readlines()
        i = 0
        while i < len(inverted_file):
            tmp = [int(k) for k in inverted_file[i].split()]
            term_dict[(tmp[0], tmp[1])] = (i, tmp[2])
            i += tmp[2] + 1
    return inverted_file, term_dict # term_dict: {term: (offset, raw_df}}


def load_queries(fn, vocab_index, term_dict):
    queries = []
    queries_len = []
    with open(fn, 'r') as rf:
        root = ET.parse(rf).getroot()
        for x in root.findall("topic"):
            tmp_dict = {}
            t = x.find("concepts").text.strip().split('。')[0]
            queries_len.append(len(t))
            tmp = t.split('、')
            #t = x.find("title").text.strip()
            #queries_len[-1] += len(t)
            #tmp.append(t)
            #t = x.find("question").text.strip().split('。')[0]
            #queries_len[-1] += len(t)
            #tmp.append(t)
            #t = x.find("narrative").text.strip().split('。')[0]
            #queries_len[-1] += len(t)
            #tmp.append(t)
            for w in tmp:
                for j in range(len(w)):
                    try:
                        unigram = (vocab_index[w[j]], -1)
                    except:
                        continue
                    if unigram in term_dict:
                        if unigram not in tmp_dict:
                            tmp_dict[unigram] = 1
                        else:
                            tmp_dict[unigram] += 1
                    if j < len(w) - 1:
                        try:
                            bigram = (unigram[0], vocab_index[w[j + 1]])
                        except:
                            continue
                        if bigram in term_dict:
                            if bigram not in tmp_dict:
                                tmp_dict[bigram] = 1
                            else:
                                tmp_dict[bigram] += 1
            queries.append(tmp_dict)
    return queries, queries_len  # queries: [{term: raw_tf}]    # queries_len: [q_len]


def idf(df, n):
    return np.log((n - df + 0.5) / (df + 0.5))

def tfidf(tf, idf, d, avg_d, k, b):
    return idf * ((k + 1) * tf) / (k * (1.0 - b + b * (d / avg_d)) + tf)


def to_vec(term_dict, doc_list, inverted_file, avg_d, q_len, q_dict, k, b):
    doc_row = dict()
    row_doc = []
    term_df = []
    q_tf = []
    d_tf = []
    d_len = []
    
    t_count = 0
    for t in q_dict:
        term_df.append(term_dict[t][1])
        q_tf.append(q_dict[t])
        for tmp in inverted_file[term_dict[t][0] + 1: term_dict[t][0] + term_dict[t][1] + 1]:
            tmp = [int(k) for k in tmp.strip().split()] # tmp: [doc_index, doc's raw_tf]
            if tmp[0] not in doc_row:
                doc_row[tmp[0]] = len(doc_row)
                row_doc.append(doc_list[tmp[0]][0])
                d_len.append(doc_list[tmp[0]][1])
                d_tf.append(np.zeros((len(q_dict), )))
            d_tf[doc_row[tmp[0]]][t_count] = tmp[1]
        t_count += 1
    term_df = np.array(term_df).reshape(1, len(term_df))
    q_tf = np.array(q_tf).reshape(1, len(q_tf))
    d_tf = np.array(d_tf)
    d_len = np.array(d_len).reshape(len(d_len), 1)
    term_idf = idf(term_df, len(doc_list))
    q = tfidf(q_tf, term_idf, q_len, avg_d, k, b)
    d = tfidf(d_tf, term_idf, d_len, avg_d, k, b)

    return q, d, row_doc  # row_doc: [doc_name]


