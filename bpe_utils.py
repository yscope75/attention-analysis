"""Going from BERT's bpe tokenization to word-level tokenization."""

import utils
from bert import tokenization

import numpy as np
import unicodedata

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def _run_split_on_punc(text, never_split=None):
  """Splits punctuation on a piece of text."""
  if never_split is not None and text in never_split:
      return [text]
  chars = list(text)
  i = 0
  start_new_word = True
  output = []
  while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
          output.append([char])
          start_new_word = True
      else:
          if start_new_word:
              output.append([])
          start_new_word = False
          output[-1].append(char)
      i += 1

  return ["".join(x) for x in output]
      
def tokenize_and_align(tokenizer, words, cased):
  """Given already-tokenized text (as a list of strings), returns a list of
  lists where each sub-list contains BERT-tokenized tokens for the
  correponding word."""

  words = ["<s>"] + words + ["</s>"]
  # basic_tokenizer = tokenizer.basic_tokenizer
  tokenized_words = []
  for word in words:
    word = tokenization.convert_to_unicode(word)
    # word = basic_tokenizer._clean_text(word)
    if word == "<s>" or word == "</s>":
      word_toks = [word]
    else:
      if not cased:
        word = word.lower()
        # word = basic_tokenizer._run_strip_accents(word)
      word_toks = [word]

    tokenized_word = []
    for word_tok in word_toks:
      tokenized_word += tokenizer.tokenize(word_tok)
    tokenized_words.append(tokenized_word)

  i = 0
  word_to_tokens = []
  for word in tokenized_words:
    tokens = []
    for _ in word:
      tokens.append(i)
      i += 1
    word_to_tokens.append(tokens)
  assert len(word_to_tokens) == len(words)

  return word_to_tokens


def get_word_word_attention(token_token_attention, words_to_tokens,
                            mode="first"):
  """Convert token-token attention to word-word attention (when tokens are
  derived from words using something like byte-pair encodings)."""

  word_word_attention = np.array(token_token_attention)
  not_word_starts = []
  for word in words_to_tokens:
    not_word_starts += word[1:]

  # sum up the attentions for all tokens in a word that has been split
  for word in words_to_tokens:
    word_word_attention[:, word[0]] = word_word_attention[:, word].sum(axis=-1)
  word_word_attention = np.delete(word_word_attention, not_word_starts, -1)

  # several options for combining attention maps for words that have been split
  # we use "mean" in the paper
  for word in words_to_tokens:
    if mode == "first":
      pass
    elif mode == "mean":
      word_word_attention[word[0]] = np.mean(word_word_attention[word], axis=0)
    elif mode == "max":
      word_word_attention[word[0]] = np.max(word_word_attention[word], axis=0)
      word_word_attention[word[0]] /= word_word_attention[word[0]].sum()
    else:
      raise ValueError("Unknown aggregation mode", mode)
  word_word_attention = np.delete(word_word_attention, not_word_starts, 0)

  return word_word_attention


def make_attn_word_level(data, tokenizer, cased):
  for features in utils.logged_loop(data):
    words_to_tokens = tokenize_and_align(tokenizer, features["words"], cased)
    assert sum(len(word) for word in words_to_tokens) == len(features["tokens"])
    features["attns"] = np.stack([[
        get_word_word_attention(attn_head, words_to_tokens)
        for attn_head in layer_attns] for layer_attns in features["attns"]])
