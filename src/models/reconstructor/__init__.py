"""Reconstructor (purification transformer).

A small causal transformer, trained on matched (adversarial, clean) flow
pairs, that undoes TANTGen's timing (IPAT) and size (IPv4 length) changes on
the ``[L, 1090]`` per-packet tensor. :mod:`realize` writes the purified
length and timing back onto the original packets, which are checked with
``tshark``.
"""
