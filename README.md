# gensim-MCC
This is the corresponding repository to my Masterthesis on Multi-Prototype Diachronic Word Embeddings.
I created four different models for Diachronic Word Embeddings: EMB-A, EMB-B, EMB-C and EMB-D. EMB-A is a reference Skipgram Word Embedding without any time context. EMB-B is a Skipgram model trained on defined time bins. EMB-C takes the same time bins, but learns successively on each time bin. EMB-D is a PPMI-based approach of Yao et al.



### Run it
The dataset used for this thesis was provided by the MCC (Mercator Institute On Global Commons and Climate Change) and is suited mainly for this data. I'll still explain how to run the code for that dataset.

1. <python3 preprocess.py input_file output_file vocabulary_size use_PPMI use_all>


[Dynamic Word Embeddings for Evolving Semantic Discovery, Yao et al.](https://arxiv.org/pdf/1703.00607.pdf)
[Diachronic word embeddings and semantic shifts: a survey, Kutuzov et al.](https://www.aclweb.org/anthology/C18-1117.pdf)
