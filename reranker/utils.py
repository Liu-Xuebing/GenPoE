import torch.nn as nn


def first_word_cap(text):
    words = text.split()
    words[0] = words[0].capitalize()
    text = " ".join(words)
    return text




