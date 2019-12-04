import logging
from gensim.models import KeyedVectors
from scipy import stats
from gensim import utils, matutils
from gensim.utils import deprecated
import numpy as np


logger = logging.getLogger(__name__)

def process_context(c, vocab):
    """
        Context has format n words split by space, <b> target word </b> another m words
    """
    del_start = c.find('<b>')
    del_end = c.find('</b>')
    c_no_target = c[:del_start] + c[(del_end+4):]
    splitted = c_no_target.split(" ")
    return [word for word in splitted if word in vocab]

def similarity(s1, s2):
    """Compute cosine similarity between two words.
        Parameters
        ----------
        w1 : str
            Input word.
        w2 : str
            Input word.
        Returns
        -------
        float
            Cosine similarity between `w1` and `w2`.
    """
    return np.dot(matutils.unitvec(s1), matutils.unitvec(s2))

def evaluate_word_pairs(keyed_vectors, sense_vectors, pairs, delimiter='\t', restrict_vocab=3000000,
                            case_insensitive=True, dummy4unknown=False, context_eval=False):
    """Compute correlation of the model with human similarity judgments.
        Notes
        -----
        More datasets can be found at
        * http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html
        * https://www.cl.cam.ac.uk/~fh295/simlex.html.
        Parameters
        ----------
        pairs : str
            Path to file, where lines are 3-tuples, each consisting of a word pair and a similarity value.
            See `test/test_data/wordsim353.tsv` as example.
        delimiter : str, optional
            Separator in `pairs` file.
        restrict_vocab : int, optional
            Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.
            This may be meaningful if you've sorted the model vocabulary by descending frequency (which is standard
            in modern word embedding models).
        case_insensitive : bool, optional
            If True - convert all words to their uppercase form before evaluating the performance.
            Useful to handle case-mismatch between training tokens and words in the test set.
            In case of multiple case variants of a single word, the vector for the first occurrence
            (also the most frequent if vocabulary is sorted) is taken.
        dummy4unknown : bool, optional
            If True - produce zero accuracies for 4-tuples with out-of-vocabulary words.
            Otherwise, these tuples are skipped entirely and not used in the evaluation.
        Returns
        -------
        pearson : tuple of (float, float)
            Pearson correlation coefficient with 2-tailed p-value.
        spearman : tuple of (float, float)
            Spearman rank-order correlation coefficient between the similarities from the dataset and the
            similarities produced by the model itself, with 2-tailed p-value.
        oov_ratio : float
            The ratio of pairs with unknown words.
    """
    ok_vocab = [(w, keyed_vectors.vocab[w]) for w in keyed_vectors.index2word[:restrict_vocab]]
    ok_vocab = {w.upper(): v for w, v in reversed(ok_vocab)} if case_insensitive else dict(ok_vocab)

    similarity_gold = []
    similarity_model = []
    oov = 0

    original_vocab = keyed_vectors.vocab
    keyed_vectors.vocab = ok_vocab
    vocab_keys = keyed_vectors.vocab.keys()

    with utils.open_file(pairs) as fin:
        for line_no, line in enumerate(fin):
            line = utils.to_unicode(line)
            if line.startswith('#'):
                # May be a comment
                continue
            else:
                try:
                    if case_insensitive:
                        if context_eval == False:
                            a, b, sim = [word.upper() for word in line.split(delimiter)]
                        else:
                            _, a, _, b, _, a_c, b_c, sim, _, _, _, _, _, _, _, _, _, _ = [word.upper() for word in line.split(delimiter)]
                            a_c = process_context(a_c, vocab_keys)
                            b_c = process_context(b_c, vocab_keys)
                    else:
                        if context_eval == False:
                            a, b, sim = [word for word in line.split(delimiter)]
                        else:
                            _, a, _, b, _, a_c, b_c, sim, _, _, _, _, _, _, _, _, _, _ = [word for word in line.split(delimiter)]
                            a_c = process_context(a_c, vocab_keys)
                            b_c = process_context(b_c, vocab_keys)
                    sim = float(sim)
                except (ValueError, TypeError):
                    logger.info('Skipping invalid line #%d in %s', line_no, pairs)
                    continue
                if a not in ok_vocab or b not in ok_vocab or not sense_vectors[keyed_vectors.vocab.get(a).index] or not sense_vectors[keyed_vectors.vocab.get(b).index]:
                    #print(a)
                    #print(b)
                    oov += 1
                    if dummy4unknown:
                        logger.debug('Zero similarity for line #%d with OOV words: %s', line_no, line.strip())
                        similarity_model.append(0.0)
                        similarity_gold.append(sim)
                        continue
                    else:
                        logger.debug('Skipping line #%d with OOV words: %s', line_no, line.strip())
                        continue
                similarity_gold.append(sim)  # Similarity from the dataset
                ##### this has to change to be able to deal with multiple senses
                if context_eval == False:
                    max_sim = -200
                    for s1 in sense_vectors[keyed_vectors.vocab.get(a).index]:
                        for s2 in sense_vectors[keyed_vectors.vocab.get(b).index]:
                            #print(s1)
                            #print(s2)
                            sim_senses = similarity(s1, s2)
                            if sim_senses > max_sim:
                                max_sim = sim_senses
                    #print(a)
                else:
                    #print(a)
                    #print(a_c)
                    a_context = 1/(len(a_c)) * np.sum([keyed_vectors.get_vector(c) for c in a_c], axis=0)
                    #print(b)
                    #print(a_context.shape)
                    b_context = 1/(len(b_c)) * np.sum([keyed_vectors.get_vector(c) for c in b_c], axis=0)
                    max_sim_a = -200
                    s_a = sense_vectors[keyed_vectors.vocab.get(a).index][0]
                    max_sim_b = -200
                    s_b = sense_vectors[keyed_vectors.vocab.get(b).index][0]
                    for s1 in sense_vectors[keyed_vectors.vocab.get(a).index]:
                        sim_a = similarity(s1, a_context)
                        if sim_a > max_sim_a:
                            max_sim_a = sim_a
                            s_a = s1
                    #print(s_a.shape)
                    #print(max_sim_a)
                    for s2 in sense_vectors[keyed_vectors.vocab.get(b).index]:
                        sim_b = similarity(s2, b_context)
                        if sim_b > max_sim_b:
                            max_sim_b = sim_b
                            s_b = s2
                    max_sim = similarity(s_a, s_b)
                similarity_model.append(max_sim)  # Similarity from the model
                #####
    keyed_vectors.vocab = original_vocab
    spearman = stats.spearmanr(similarity_gold, similarity_model)
    pearson = stats.pearsonr(similarity_gold, similarity_model)
    print(oov)
    print(len(similarity_gold))
    if dummy4unknown:
        oov_ratio = float(oov) / len(similarity_gold) * 100
    else:
        oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

    logger.debug('Pearson correlation coefficient against %s: %f with p-value %f', pairs, pearson[0], pearson[1])
    print('Pearson correlation coefficient against {0}: {1} with p-value {2}'.format(pairs, pearson[0], pearson[1]))
    logger.debug(
        'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
        pairs, spearman[0], spearman[1]
    )
    print('Spearman rank-order correlation coefficient against {0}: {1} with p-value {2}'.format(pairs, spearman[0], spearman[1]))
    #logger.debug('Pairs with unknown words: %d', oov)
    #print('Pairs with unknown words: {0}%'.format(oov))
    keyed_vectors.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
    return pearson, spearman, oov_ratio

def evaluate_word_analogies(self, analogies, restrict_vocab=300000, case_insensitive=True, dummy4unknown=False):
    """Compute performance of the model on an analogy test set.
        This is modern variant of :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.accuracy`, see
        `discussion on GitHub #1935 <https://github.com/RaRe-Technologies/gensim/pull/1935>`_.
        The accuracy is reported (printed to log and returned as a score) for each section separately,
        plus there's one aggregate summary at the end.
        This method corresponds to the `compute-accuracy` script of the original C word2vec.
        See also `Analogy (State of the art) <https://aclweb.org/aclwiki/Analogy_(State_of_the_art)>`_.
        Parameters
        ----------
        analogies : str
            Path to file, where lines are 4-tuples of words, split into sections by ": SECTION NAME" lines.
            See `gensim/test/test_data/questions-words.txt` as example.
        restrict_vocab : int, optional
            Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.
            This may be meaningful if you've sorted the model vocabulary by descending frequency (which is standard
            in modern word embedding models).
        case_insensitive : bool, optional
            If True - convert all words to their uppercase form before evaluating the performance.
            Useful to handle case-mismatch between training tokens and words in the test set.
            In case of multiple case variants of a single word, the vector for the first occurrence
            (also the most frequent if vocabulary is sorted) is taken.
        dummy4unknown : bool, optional
            If True - produce zero accuracies for 4-tuples with out-of-vocabulary words.
            Otherwise, these tuples are skipped entirely and not used in the evaluation.
        Returns
        -------
        score : float
            The overall evaluation score on the entire evaluation set
        sections : list of dict of {str : str or list of tuple of (str, str, str, str)}
            Results broken down by each section of the evaluation set. Each dict contains the name of the section
            under the key 'section', and lists of correctly and incorrectly predicted 4-tuples of words under the
            keys 'correct' and 'incorrect'.
    """
    ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
    ok_vocab = {w.upper(): v for w, v in reversed(ok_vocab)} if case_insensitive else dict(ok_vocab)
    oov = 0
    logger.info("Evaluating word analogies for top %i words in the model on %s", restrict_vocab, analogies)
    sections, section = [], None
    quadruplets_no = 0
    with utils.open(analogies, 'rb') as fin:
        for line_no, line in enumerate(fin):
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    self._log_evaluate_word_analogies(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError("Missing section header before line #%i in %s" % (line_no, analogies))
                try:
                    if case_insensitive:
                        a, b, c, expected = [word.upper() for word in line.split()]
                    else:
                        a, b, c, expected = [word for word in line.split()]
                except ValueError:
                    logger.info("Skipping invalid line #%i in %s", line_no, analogies)
                    continue
                quadruplets_no += 1
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    oov += 1
                    if dummy4unknown:
                        logger.debug('Zero accuracy for line #%d with OOV words: %s', line_no, line.strip())
                        section['incorrect'].append((a, b, c, expected))
                    else:
                        logger.debug("Skipping line #%i with OOV words: %s", line_no, line.strip())
                    continue
                original_vocab = self.vocab
                self.vocab = ok_vocab
                ignore = {a, b, c}  # input words to be ignored
                predicted = None
                # find the most likely prediction using 3CosAdd (vector offset) method
                # TODO: implement 3CosMul and set-based methods for solving analogies
                sims = self.most_similar(positive=[b, c], negative=[a], topn=5, restrict_vocab=restrict_vocab)
                self.vocab = original_vocab
                for element in sims:
                    predicted = element[0].upper() if case_insensitive else element[0]
                    if predicted in ok_vocab and predicted not in ignore:
                        if predicted != expected:
                            logger.debug("%s: expected %s, predicted %s", line.strip(), expected, predicted)
                        break
                if predicted == expected:
                    section['correct'].append((a, b, c, expected))
                else:
                    section['incorrect'].append((a, b, c, expected))
    if section:
        # store the last section, too
        sections.append(section)
        self._log_evaluate_word_analogies(section)

    total = {
        'section': 'Total accuracy',
        'correct': list(chain.from_iterable(s['correct'] for s in sections)),
        'incorrect': list(chain.from_iterable(s['incorrect'] for s in sections)),
    }

    oov_ratio = float(oov) / quadruplets_no * 100
    logger.info('Quadruplets with out-of-vocabulary words: %.1f%%', oov_ratio)
    if not dummy4unknown:
        logger.info(
            'NB: analogies containing OOV words were skipped from evaluation! '
            'To change this behavior, use "dummy4unknown=True"'
        )
    analogies_score = self._log_evaluate_word_analogies(total)
    sections.append(total)
    # Return the overall score and the full lists of correct and incorrect analogies
    return analogies_score, sections
