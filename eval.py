from gensim.models import Word2Vec, KeyedVectors
import os
import util_timeCD as util
import pickle
import logging
import numpy as np


logging.basicConfig(filename="./logs/log_eval_small", format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

file_path = "./dia_correctness_runs/"

def gensim_eval(model):
    model.wv.accuracy('eval_data/questions-words.txt')
    wordsim_clas = model.wv.evaluate_word_pairs('eval_data/wordsim353.tsv')
    wordsim_rel = model.wv.evaluate_word_pairs('eval_data/wordsim_relatedness_goldstandard.txt')
    wordsim_sim = model.wv.evaluate_word_pairs('eval_data/wordsim_similarity_goldstandard.txt')
    #simlex = model.wv.evaluate_word_pairs('eval_data/SimLex-999.txt', dummy4unknown=True)

    #print("Analogy scores: {0}".format(analogy_scores[:-1]))
    print("Wordsim Classic: {0}".format(wordsim_clas))
    print("Wordsim Relatedness: {0}".format(wordsim_rel))
    print("Wordsim Similarity: {0}".format(wordsim_sim))
    #print("SimLex: {0}".format(simlex))


dynamic = ['art', 'damn', 'gay', 'hell', 'maid', 'muslim', 'apple', 'amazon', 'obama', 'trump', 'iphone', 'twitter', 'mp3', 'clinton', 'bush', 'pippen', 'isis', 'enron', 'qaeda']
additional = ['sick', 'war']
kim = ['checked', 'check', 'actually', 'supposed', 'guess', 'cell', 'headed', 'ass', 'mail']
kulkarni = ['transmitted', 'bitch', 'sex', 'her', 'desk', 'diet', 'recording', 'tape', 'checking', 'plastic', 'peck', 'honey', 'hug', 'windows', 'sink', 'click', 'handle', 'instant', 'twilight', 'rays', 'streaming', 'ray', 'delivery', 'combo', 'candy', 'rally', 'snap', 'mystery', 'stats', 'sandy', 'shades']
hamilton = ['broadcast', 'awful', 'fatal', 'nice', 'monitor', 'record', 'guy', 'call', 'headed', 'calls', 'started', 'wanting', 'starting', 'major']
rosenfeld = ['mail', 'canadian']
sdgs = ['sustainable', 'sustainability', 'poverty', 'hunger', 'health', 'education', 'gender', 'water', 'sanitation', 'affordable', 'energy', 'clean', 'growth', 'economic', 'innovation', 'inequality', 'city', 'consumption', 'production', 'action', 'climate', 'peace', 'justice', 'life', 'land', 'partnerships']
mdgs = ['primary', 'equality', 'empowerment', 'mortality', 'maternal', 'hiv', 'aids', 'malaria', 'diseases', 'environmental', 'global']
energy = ['coal', 'petrol', 'petroleum', 'solar', 'photovoltaic', 'renewable', 'wind', 'hydro', 'tide', 'wave', 'geothermal', 'atomic', 'uranium', 'kerosene', 'nuclear'] 
transport = ['car', 'train', 'bicycle', 'vehicle', 'hybrid', 'plane', 'bus', 'aircraft', 'rail', 'engine'] 
new = ['malaria', 'vaccination', 'clean', 'war', 'green', 'carbon', 'ozone', 'smog', 'fume', 'pollution', 'harm', 'aerosols', 'animalbased', 'vegetarian', 'vegan', 'meat', 'protein', 'alcohol', 'healthy', 'unhealthy', 'drugs', 'addiction', 'cancer', 'safe', 'energyefficient', 'fukushima', 'summit', 'australia', 'america', 'europe', 'reef', 'extinct', 'kyoto', 'protest', 'paris', 'unfccc']
pack = ['plastic', 'paper', 'cardboard', 'carton', 'bottle', 'glass', 'trash', 'garbage', 'biodegradable', 'degradable', 'packaging', 'textiles', 'clothing', 'brands', 'bag', 'bags', 'recycle', 'recycling', 'refuse', 'incinerators', 'mswi']
mine = ['clean', 'kyoto', 'animalbased', 'plastic', 'plastics', 'degradable', 'prevention', 'catastrophes']
all_words = mine


def qualitative_eval_gensim(models, nn=5):
    nns = {}
    for eval_word in all_words:
        nns[eval_word] =[]
        for m in models:
            try:
                ms = m.wv.most_similar(eval_word, topn=nn)
                print('Evaluating word {0}'.format(eval_word))
                print([w[0] for w in ms])
                nns[eval_word].append([w[0] for w in ms])
            except KeyError:
                print('Word {0} not in vocabulary'.format(eval_word))
                nns[eval_word].append([])
    return nns


def qualitative_eval_dynamic(emb, nn=10):
    tok2idx = pickle.load(open('data/tok2idx.pkl', 'rb'))
    idx2tok = pickle.load(open('data/idx2tok.pkl', 'rb'))
    print(len(tok2idx))
    nns = {}
    for eval_word in all_words:
        nns[eval_word] = []
        #for m in models:
        #emb = pickle.load(open('runs/{}'.format(m), 'rb'))
        #for e in emb:
        #print(len(emb))
        #print(e.shape)
        try:
            print('Evaluating word {0}'.format(eval_word))
            print(tok2idx[eval_word])
            print(emb.shape)
            neighbors = util.getclosest(tok2idx[eval_word], 5*[emb], nn)
            #print(neighbors)
            #print(len(neighbors))
            #for t in neighbors:
            #print(len(t))
            #print(len(t[0]))
            #break
            neighbor_list = []
            for i, neighbor in enumerate(neighbors[0]):
                #if i%10==0:
                #print(i)
                #print("Time bin: {}".format(str(i//10)))
                print(neighbor)
                print("{}".format(idx2tok[neighbor]), end=" ")
                neighbor_list.append(idx2tok[neighbor])
            nns[eval_word].append(neighbor_list)
        except KeyError:
            nns[eval_word].append(4*[])
            print('Word {0} not in vocabulary'.format(eval_word))
    return nns


def nn_to_latex(nns, emb_string):
    start_str = "\\begin{table*}[t] \n \\centering \n \\resizebox{\\textwidth}{!}{ \\begin{tabular}{| r | r | r | r | r | r | r | r | r |} \\hline \n"
    
    #print(nns)
    eval_words = [w for w, models in nns.items()]
    ms = [models for w, models in nns.items()]
    #print(eval_res)
    start_str += "& "
    start_str += " & ".join(eval_words)
    start_str += "\\\\ \\hline \n"

    emb_strings = ['EMB-A', 'EMB-B', 'EMB-C', 'EMB-D']
    time_bins = ['1985-2009', '2010-2013', '2014-2016', '2017-2018']
    
    start_str += emb_string
    if emb_string == 'EMB-A':
        time_bins = [' ']
        print(time_bins)
    #for i, t in enumerate(time_bins):
    for x in range(1):
        t = ''
        i = 1
        print(t)
        start_str += (" " + t)
        eval_res = [m[i] for m in ms]
        print(eval_res)
        for j in range(len(eval_res[0])):
            topnn = [e[j] for e in eval_res]
            print(topnn)
            for nn in topnn:
                start_str += " & "
                start_str += nn
            start_str += "\\\\ \n"
        start_str += "\\hline \n"
    end_str = "\\hline \n \\end{tabular}} \n \\end{table*}"
    print(start_str+end_str)

def nn_to_latex_2(nns):
    start_str = "\\begin{table*}[t] \n \\centering \n \\resizebox{\\textwidth}{!}{ \\begin{tabular}{| r | r | r | r | r | r | r | r | r | r | r |} \\hline \n"
    for w, models in nns.items():
        start_str += w
        for m in models:
            for n in m:
                start_str += " & "
                start_str += n
            start_str += "\\\\ \n"
        start_str += "\\hline \n"
    end_str = "\\hline \n \\end{tabular}} \n \\end{table*}"
    print(start_str+end_str)

def matrix_to_word2vecformat(mat, time):
    idx2tok = pickle.load(open('data/idx2tok.pkl', 'rb'))
    #print(len(idx2tok))
    for i in range(1):
    #for i, x in enumerate(mat):
        #print(i)
        print(mat.shape)
        f = open("{0}/dynamic_word2vec_{1}.bin".format(file_path, time), "w")
        f.write("{0} {1}\n".format(mat.shape[0], mat.shape[1]))
        for idx, row in enumerate(mat):
            f.write("{0}".format(idx2tok[idx]))
            for val in row:
                f.write(" {0}".format(val))
            f.write("\n")
        f.close()
        print("Matrix transformed")
        


prefixed = [] 
prefixed = [filename for filename in os.listdir('./dia_correctness_runs/') if filename.startswith("emb_a_mcc_") and filename.endswith(".model")]
dynamic_models_u = [fn for fn in os.listdir(file_path) if fn.startswith("L10T50G100A1ngU_iter") and fn.endswith("4.p")]
#dynamic_models_u = []
dynamic_models_v = [fn for fn in os.listdir(file_path) if fn.startswith("L10T50G100A1ngV_iter") and fn.endswith("4.p")]
#dynamic_models_v =[]
print(dynamic_models_u)
print(prefixed.sort())
models = []
for time_bin in prefixed:
    print(time_bin)
    model = Word2Vec.load('{0}{1}'.format(file_path, time_bin))
    models.append(model)
    #gensim_eval(model)
res_gensim = qualitative_eval_gensim(models)
#res_yao = qualitative_eval_dynamic(dynamic_models)
nn_to_latex(res_gensim, 'EMB-A')
dyn_models = []
for i in range(len(dynamic_models_u)):
    for j in range(4):
        uu = pickle.load(open('{0}{1}'.format(file_path, dynamic_models_u[i]), 'rb'))[j]
        #vv = pickle.load(open('{0}{1}'.format(file_path, dynamic_models_v[i]), 'rb'))[j]
        norms = np.sqrt(np.sum(np.square(uu), axis=1, keepdims=True))
        uu /= np.maximum(norms, 1e-7)
        #unorm = uu / np.sqrt(np.sum(uu*uu, axis=1, keepdims=True))
        #vnorm = vv / np.sqrt(np.sum(vv*vv, axis=1, keepdims=True))
        #word_vecs = np.dot(uu, vv.T)
        #word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs*word_vecs, axis=1, keepdims=True))
        #emb = uu
        #embnrm = np.reshape(np.sqrt(np.sum(emb**2,1)),(emb.shape[0],1))
        #emb_normalized = np.divide(emb, np.tile(embnrm, (1,emb.shape[1])))
        #print(emb_normalized.shape)
        
        matrix_to_word2vecformat(uu, j)
        word_vectors = KeyedVectors.load_word2vec_format('{0}/dynamic_word2vec_{1}.bin'.format(file_path, j))
        dyn_models.append(word_vectors)
        #gensim_eval(word_vectors)
res_dynamic = qualitative_eval_gensim(dyn_models)
nn_to_latex(res_dynamic, 'EMB-D')

