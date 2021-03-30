"""
    Author: Giorgia Cantisani
    Implementation of the Contrastive-NMF algorithm used in the paper:

    Cantisani et al. "NEURO-STEERED MUSIC SOURCE SEPARATION WITH EEG-BASED AUDITORY 
    ATTENTION DECODING AND CONTRASTIVE-NMF." 2021 IEEE International Conference on 
    Acoustics, Speech and Signal Processing (ICASSP).
"""

from sklearn.decomposition._nmf import _beta_divergence
import numpy as np


def contrastive_NMF(v, w_init, h_init, h_tilde, delta=0, mu=0, beta=0, n_iter=100, nr_src=2):
    """

    Parameters
    ----------
    v: [array of shape (F, N)] magnitude spectrogram of the mixture
    w_init: [array of shape (F, K)] initialization of the dictionary w
    h_init: [array of shape (K, N)] initialization of the activations h
    h_tilde: [array of shape (K1, N)] activations of the source to enhance 
    delta: [float > 0] weight of the contrastive term
    mu: [float > 0] weight of the l1 regularizer on h
    beta: [float > 0] weight of the l1 regularizer on w
    n_iter: [int > 0] number of NMF iterations
    n_src: [int > 0] number of sources in the mixture


    Returns
    -------
    the dictionary w and the corresponding activations h resulting from the 
    factorization of v and a list containing the total cost at each iteration.
    """

    flr = 1e-9
    cost = []

    # initial values
    x = v.copy()
    w = w_init.copy()
    h = h_init.copy()

    # avoid too small values
    x[x <= flr] = flr
    w[w <= flr] = flr
    h[h <= flr] = flr

    # normalize h_tilde
    hn_tilde = np.sqrt(np.sum(h_tilde ** 2, axis=1))
    h_tilde = h_tilde / hn_tilde[:, None]

    # normalize h and rescale w
    hn = np.sqrt(np.sum(h ** 2, axis=1))
    h = h / hn[:, None]
    w = w * hn[None, :]

    # NMF iterations
    for i in range(n_iter):

        # update H
        WH = np.maximum(w @ h, flr)
        h, contrast = update_h(w, h, x, h_tilde, WH, delta, mu, flr, nr_src)
        h[h <= flr] = flr
        
        # normalize h and rescale w
        hn = np.sqrt(np.sum(h ** 2, axis=1))
        h = h / hn[:, None]
        w = w * hn[None, :]

        #  update W        
        WH = np.maximum(w@h, flr)
        w = update_w(w, h, x, WH, beta, flr)
        w[w <= 0] = flr

        # keep track of the cost
        cost.append(_beta_divergence(x, w, h, 'kullback-leibler', square_root=True) - delta * contrast + mu * np.linalg.norm(h) + beta * np.linalg.norm(w))

    return w, h, cost


def update_h(w, h, x, h_tilde, WH, delta, mu, flr, nr_src):
    F = x.shape[0]
    N = x.shape[1]
    K = h.shape[0]
    K1 = int(K / nr_src)

    # normal updates without the contrastive term
    dn_h = w.T @ (x / WH)
    dp_h = w.T @ np.ones((F, N)) + mu

    # margin term
    if delta > 0:

        # select submatrices
        h1 = h[:K1, :]
        h2 = h[K1:, :]
        h1_tilde = h_tilde[:K1, :]        

        # compute covariances
        cov1 = h1 @ h1_tilde.T
        cov2 = h2 @ h1_tilde.T

        # create auxillary matrices
        pp = np.zeros((K, N))
        pn = np.zeros((K, N))

        #  fill denominator
        pp[K1:, :] = cov2 @ h1_tilde

        # fill numerator
        pn[:K1, :] = cov1 @ h1_tilde

        # add contrastive term to the updates
        dn_h = dn_h + delta * pn
        dp_h = dp_h + delta * pp

        # keep track of the cost
        contrast = np.linalg.norm(cov1) - np.linalg.norm(cov2)

    dp_h = np.maximum(dp_h, flr)

    return h * (dn_h / dp_h), contrast


def update_w(w, h, x, WH, beta, flr):
    F = x.shape[0]
    N = x.shape[1]

    h_sum = np.ones((F, N)) @ h.T
    X_WH_HT = (x / WH) @ h.T

    dn_w = X_WH_HT  
    dp_w = h_sum + beta 

    dpw = np.maximum(dp_w, flr)

    return w * (dn_w / dpw)




if __name__ == "__main__":
    F = 512               # number of frequency bins
    N = 3000              # number of time samples
    K = 8                 # number of spectral components
    nr_src = 2            # number of sources in the mix
    K1 = int(K / nr_src)  # number of spectral components used to represent each source

    v_mix = np.abs(np.random.randn(F, N))
    wini = np.abs(np.random.randn(F, K))
    hini = np.abs(np.random.randn(K, N))
    hside = np.abs(np.random.randn(K1, N))
    w, h, cost = contrastive_NMF(v_mix, w_init=wini, h_init=hini, h_tilde=hside, delta=1000, mu=1, beta=1, n_iter=200, nr_src=nr_src)