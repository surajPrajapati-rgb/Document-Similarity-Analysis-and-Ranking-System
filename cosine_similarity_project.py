import os
import re
import math
import heapq

DIRECTORY = '25-20240329T124513Z-001/25/'

def tokenize(document):
    tokens_list = []
    current_word = ''
    for char in document:
        if char.isalnum():
            current_word += char
        elif current_word:
            tokens_list.append(current_word.lower())
            current_word = ''
    if current_word:
        tokens_list.append(current_word.lower())
    return tokens_list

def extract_text(directory, filename):
    with open(os.path.join(directory, filename), 'r') as file:
        text_data = file.read()
        s_title = text_data.find("<TITLE>")
        e_title = text_data.find("</TITLE>", 7)

        s_text = text_data.find("<TEXT>")
        e_text = text_data.find("</TEXT>", 6)

        extracted_text = text_data[s_title+7:e_title].strip().lower() + "\n" + text_data[s_text+6:e_text].strip().lower()
        return extracted_text

def calculate_tf(tokens, TOKEN_IDs):
    tfs = {}
    for token in tokens:
        if TOKEN_IDs[token] in tfs:
            tfs[TOKEN_IDs[token]] += 1
        else:
            tfs[TOKEN_IDs[token]] = 1
    return tfs

def calculate_idfs(TermDocumentMap, TOKEN_IDs, DOCS_IDs):
    N = len(DOCS_IDs)
    idfs = {}
    for term, documents in TermDocumentMap.items():
        idf = math.log(N / len(documents))
        # idfs[term] = round(idf, 3)
        idfs[term] = idf
    return idfs

def calculate_tf_idfs(TFs, IDFs):
    tf_idf = {}
    for doc in TFs:
        vector = {}
        for term in TFs[doc]:
            tf = TFs[doc][term]
            idf = IDFs[term]
            vector[term] = tf*idf
        sorted_vector = dict(sorted(vector.items()))
        tf_idf[doc] = sorted_vector
    return tf_idf

def calculate_cosine_similarity(v1, v2):
    v2_keys = list(v2.keys())       # List of all terms 
    v2_j = 0                        # pointer for document v2_keys list
    v1v2 = 0                        # dot products sum pointer
    for v1_i in v1:
        while v2_j < len(v2_keys) and v1_i >= v2_keys[v2_j] :
            if v1_i == v2_keys[v2_j]: # comparing term for both documents
                v1v2 += v1[v1_i]*v2[v2_keys[v2_j]]
                break
            v2_j+=1
    norm_v1 = math.sqrt(sum( v1[i]**2 for i in v1))
    norm_v2 = math.sqrt(sum( v2[i]**2 for i in v2))

    similarity_value =  v1v2/(norm_v1*norm_v2)
    return similarity_value

def calculate_similarity(TF_IDF_VECTORS):
    max_heap = []
    N = len(TF_IDF_VECTORS)
    for doc1 in range(1, N+1):
        for doc2 in range(doc1+ 1, N+1):
            value = calculate_cosine_similarity(TF_IDF_VECTORS[doc1], TF_IDF_VECTORS[doc2])
            heapq.heappush(max_heap, (-value, doc1, doc2))
    return max_heap

def main(DIRECTORY):
    FILES = os.listdir(DIRECTORY)
    DOC_ID  = 1
    TOKEN_ID = 1
    DOCS_IDs = {} # dict of <DOCNO> entry to document-id
    TOKEN_IDs = {} # dict of token to token-id
    TermDocumentMap = {}
    TFs = {}
    for filename in FILES:
        DOCS_IDs[DOC_ID] = filename
        tokens = tokenize(extract_text(DIRECTORY, filename))
        for token in tokens:
            if token not in TOKEN_IDs:
                TOKEN_IDs[token] = TOKEN_ID
                TermDocumentMap[TOKEN_ID] = set() # unique documents
                TermDocumentMap[TOKEN_ID].add(DOC_ID)
                TOKEN_ID += 1
            else:
                TermDocumentMap[TOKEN_IDs[token]].add(DOC_ID)
        TFs[DOC_ID] =  calculate_tf(tokens, TOKEN_IDs)
        DOC_ID += 1
    
    return DOCS_IDs, TOKEN_IDs, TFs, TermDocumentMap

DOCS_IDs, TOKEN_IDs, TFs, TermDocumentMap = main(DIRECTORY)
IDFs = calculate_idfs(TermDocumentMap, TOKEN_IDs, DOCS_IDs)
TF_IDF_VECTORS = calculate_tf_idfs( TFs, IDFs)

pairewiseSimilarity = calculate_similarity(TF_IDF_VECTORS)
limit = 0
for _ in pairewiseSimilarity:
    value, doc1, doc2 = heapq.heappop(pairewiseSimilarity)
    # if limit < 50:
    print(f"Similarity between {[DOCS_IDs[doc1]]} and {[DOCS_IDs[doc2]]}: {-value}")
    #     limit+=1
    # else:
    #     break
