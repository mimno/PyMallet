# PyMallet

This package provides tools for extracting latent semantic representations of text, particularly probabilistic topic models.

The implementation of LDA uses Gibbs sampling, which is simple but reliable. People often find the resulting models more useful than the stochastic variational algorithm used in Gensim.

To compile:

    python setup.py build_ext --inplace

As an example, the `sample_data` directory contains 10000 posts from the stats Stack Exchange forum.

To run on this sample collection with 50 topics:

    python lda.py sample_data/stats_10k.txt 50

The script `lda_reference.py` contains a reference implementation in pure Python (no Cython) to compare speed. The Cython version is currently about 100x faster.