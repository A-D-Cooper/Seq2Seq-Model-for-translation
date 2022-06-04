
'''Calculate BLEU score for one reference and one hypothesis
'''

from math import exp  # exp(x) gives e^x


def grouper(seq, n):
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    #assert False, "Fill me"
    idx = len(seq) - n
    g = []
    for x in range(0, idx+1):
        g.append(seq[x: x+n])
    return g

def n_gram_precision(reference, candidate, n):
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''
    #assert False, "Fill me"
    g_c = list(grouper(candidate, n))
    r = set(x for i in reference for x in grouper(candidate, n))
    num_g_c = len(g_c)
    counter = sum([1 for x in g_c if x in r])
    return counter/num_g_c


def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    #assert False, "Fill me"
    # return len(ref) / len(cand)
    len_cand = len(candidate)
    len_re = len(reference)
    b = len_re / len_cand
    if b < 1:
        br=1
    else:
        br=exp(1-b)
    return br



def BLEU_score(reference, hypothesis, n):
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    #assert False, "Fill me"
    all_words = hypothesis.split()
    lookup = [x.split() for x in reference]
    proc = n_gram_precision(all_words, lookup, n)
    penalty = brevity_penalty(all_words, lookup)
    final = penalty * proc
    return final
