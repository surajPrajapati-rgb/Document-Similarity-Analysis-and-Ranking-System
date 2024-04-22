import os
import re
import math
import heapq

DIRECTORY = '25-20240329T124513Z-001/25/'

def tokenize_document(document):
    tokens = re.findall(r'\w+', document)
    return tokens

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

def calculate_idfs(TERM_TO_DOCS, TOKEN_IDs, DOCS_IDs):
    N = len(DOCS_IDs)
    idfs = {}
    for term, documents in TERM_TO_DOCS.items():
        idf = math.log(N / len(documents))
        idfs[term] = round(idf, 3)
    return idfs

def calculate_tf_idfs(TFs, IDFs):
    tf_idf = {}
    for doc in TFs:
        vector = []
        for term in TFs[doc]:
            tf = TFs[doc][term]
            idf = IDFs[term]
            vector.append(tf*idf)
        tf_idf[doc] = vector
    return tf_idf

def calculate_cosine_similarity(v1, v2):
    min_vect = len(v1) if len(v1) < len(v2) else len(v2)
    v1v2 = 0
    for i in range(min_vect):
        v1v2 += v1[i]*v2[i]
    norm_v1 = math.sqrt(sum( i**2 for i in v1))
    norm_v2 = math.sqrt(sum( i**2 for i in v2))

    similarity_value =  v1v2/(norm_v1*norm_v2)
    return similarity_value

def calculate_similarity(TF_IDF_VECTORS):
    max_heap = []
    N = len(TF_IDF_VECTORS)
    for doc1 in TF_IDF_VECTORS:
        for doc2 in range(doc1+1, N+1):
            value = calculate_cosine_similarity(TF_IDF_VECTORS[doc1], TF_IDF_VECTORS[doc2])
            value = round(value, 3)
            heapq.heappush(max_heap, (-value, doc1, doc2))
    return max_heap

def main(DIRECTORY):
    FILES = os.listdir(DIRECTORY)
    DOC_ID  = 1
    TOKEN_ID = 1
    DOCS_IDs = {} # dict of <DOCNO> entry to document-id
    TOKEN_IDs = {} # dict of token to token-id
    TERM_TO_DOCS = {}
    TFs = {}
    for filename in FILES:
        DOCS_IDs[DOC_ID] = filename
        document = extract_text(DIRECTORY, filename)
        tokens = tokenize_document(document)
        for token in tokens:
            if token not in TOKEN_IDs:
                TOKEN_IDs[token] = TOKEN_ID
                TERM_TO_DOCS[TOKEN_IDs[token]] = set() # unique documents
                TERM_TO_DOCS[TOKEN_IDs[token]].add(DOC_ID)
                TOKEN_ID += 1
            else:
                TERM_TO_DOCS[TOKEN_IDs[token]].add(DOC_ID)
        TFs[DOC_ID] =  calculate_tf(tokens, TOKEN_IDs)
        DOC_ID += 1

    return DOCS_IDs, TOKEN_IDs, TFs, TERM_TO_DOCS

DOCS_IDs, TOKEN_IDs, TFs, TERM_TO_DOCS = main(DIRECTORY)
IDFs = calculate_idfs(TERM_TO_DOCS, TOKEN_IDs, DOCS_IDs)
TF_IDF_VECTORS = calculate_tf_idfs(TFs, IDFs)
pairewiseSimilarity = calculate_similarity(TF_IDF_VECTORS)

for _ in pairewiseSimilarity:
    # print(DOCS_IDs[doc1] + " -> "+ DOCS_IDs[doc2] + ":  " + str(-value))
    value, doc1, doc2 = heapq.heappop(pairewiseSimilarity)
    print(str(doc1) + " -> "+ str(doc2) + ":  " + str(-value))
