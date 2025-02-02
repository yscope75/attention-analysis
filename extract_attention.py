"""Runs BERT over input data and writes out its attention maps to disk."""

import argparse
import os
import numpy as np
# import tensorflow as tf
import torch 

from bert import modeling
from bert import tokenization
from transformers import AutoTokenizer, AutoModel
import bpe_utils
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Example(object):
  """Represents a single input sequence to be passed into BERT."""

  def __init__(self, features, tokenizer, max_sequence_length,):
    self.features = features

    if "tokens" in features:
      self.tokens = features["tokens"]
    else:
      if "text" in features:
        text = features["text"]
      else:
        text = " ".join(features["words"])
      self.tokens = ["<s>"] + tokenizer.tokenize(text) + ["</s>"]

    self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
    self.segment_ids = [0] * len(self.tokens)
    self.input_mask = [1] * len(self.tokens)
    while len(self.input_ids) < max_sequence_length:
      self.input_ids.append(0)
      self.input_mask.append(0)
      self.segment_ids.append(0)


def examples_in_batches(examples, batch_size):
  for i in utils.logged_loop(range(1 + ((len(examples) - 1) // batch_size))):
    yield examples[i * batch_size:(i + 1) * batch_size]


class AttnMapExtractor(object):
  """Runs BERT over examples to get its attention maps."""

  def __init__(self, bert_version):

    # bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    # if debug:
    #   bert_config.num_hidden_layers = 3
    #   bert_config.hidden_size = 144
    # load BERT model by version
    self.bert_model = AutoModel.from_pretrained(bert_version).to(device)
    # self._attn_maps = modeling.BertModel(
    #     config=bert_config,
    #     is_training=False,
    #     input_ids=self._input_ids,
    #     input_mask=self._input_mask,
    #     token_type_ids=self._segment_ids,
    #     use_one_hot_embeddings=True).attn_maps

  def get_attn_maps(self, examples):
    
    _input_ids = torch.from_numpy(np.vstack([e.input_ids for e in examples])).to(device)
    _input_mask = torch.from_numpy(np.vstack([e.input_mask for e in examples])).to(device)
    with torch.no_grad():
      attn_maps = self.bert_model(
        input_ids = _input_ids,
        attention_mask = _input_mask,
        output_attentions = True
      )[-1]
    
    # feed = {
    #     self._input_ids: np.vstack([e.input_ids for e in examples]),
    #     self._segment_ids: np.vstack([e.segment_ids for e in examples]),
    #     self._input_mask: np.vstack([e.input_mask for e in examples])
    # }
    return attn_maps


def main():
  # Set device 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--preprocessed_data_file", required=True,
      help="Location of preprocessed data (JSON file); see the README for "
           "expected data format.")
  parser.add_argument("--bert-dir", required=False,
                      help="Location of the pre-trained BERT model.")
  parser.add_argument("--bert_version", required=True,
                      help="specify version of the pre-trained BERT model")
  parser.add_argument("--cased", default=False, action='store_true',
                      help="Don't lowercase the input.")
  parser.add_argument("--max_sequence_length", default=256, type=int,
                      help="Maximum input sequence length after tokenization "
                           "(default=128).")
  parser.add_argument("--batch_size", default=64, type=int,
                      help="Batch size when running BERT (default=16).")
  parser.add_argument("--debug", default=False, action='store_true',
                      help="Use tiny model for fast debugging.")
  parser.add_argument("--word_level", default=False, action='store_true',
                      help="Get word-level rather than token-level attention.")
  args = parser.parse_args()

  print("Creating examples...")
  # tokenizer = tokenization.FullTokenizer(
  #     vocab_file=os.path.join(args.bert_dir, "vocab.txt"),
  #     do_lower_case=not args.cased)
  # Change to use auto tokenizer
  tokenizer = AutoTokenizer.from_pretrained(args.bert_version)
  examples = []
  for features in utils.load_json(args.preprocessed_data_file):
    example = Example(features, tokenizer, args.max_sequence_length)
    if len(example.input_ids) <= args.max_sequence_length:
      examples.append(example)

  print("Building BERT model...")
  extractor = AttnMapExtractor(args.bert_version)

  print("Extracting attention maps...")
  feature_dicts_with_attn = []
  for batch_of_examples in examples_in_batches(examples, args.batch_size):
    attns = extractor.get_attn_maps(batch_of_examples)
    # convert to numpy tensor 
    attns = torch.stack(list(attns), dim=1).cpu().numpy()
    for e, e_attn in zip(batch_of_examples, attns):
      seq_len = len(e.tokens)
      e.features["attns"] = e_attn[:, :, :seq_len, :seq_len].astype("float16")
      e.features["tokens"] = e.tokens
      feature_dicts_with_attn.append(e.features)

  if args.word_level:
    print("Converting to word-level attention...")
    bpe_utils.make_attn_word_level(
        feature_dicts_with_attn, tokenizer, args.cased)

  outpath = args.preprocessed_data_file.replace(".json", "")
  outpath += "_attn.pkl"
  print("Writing attention maps to {:}...".format(outpath))
  utils.write_pickle(feature_dicts_with_attn, outpath)
  print("Done!")


if __name__ == "__main__":
  main()