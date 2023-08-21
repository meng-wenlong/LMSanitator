# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the pattern-verbalizer pairs (PVPs) for all tasks.
"""
import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Union, Dict

import torch
from transformers import PreTrainedTokenizer, GPT2Tokenizer
from pet.utils import InputExample, get_verbalization_ids

import log
from pet import wrapper as wrp

logger = log.get_logger('root')

FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]


class PVP(ABC):
    """
    This class contains functions to apply patterns and verbalizers as required by PET. Each task requires its own
    custom implementation of a PVP.
    """

    def __init__(self, wrapper, pattern_id: int = 0, seed: int = 42):
        """
        Create a new PVP.

        :param wrapper: the wrapper for the underlying language model
        :param pattern_id: the pattern id to use
        :param verbalizer_file: an optional file that contains the verbalizer to be used
        :param seed: a seed to be used for generating random numbers if necessary
        """
        self.wrapper = wrapper
        self.pattern_id = pattern_id
        self.rng = random.Random(seed)

        """
        if verbalizer_file:
            self.verbalize = PVP._load_verbalizer_from_file(verbalizer_file, self.pattern_id)
        """

        ## if self.wrapper.config.wrapper_type in [wrp.MLM_WRAPPER, wrp.PLM_WRAPPER]:

        self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.wrapper.config.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                assert verbalizer_id != self.wrapper.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.wrapper.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.wrapper.tokenizer.mask_token_id

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        return max(len(self.verbalize(label)) for label in self.wrapper.config.label_list)

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark"""
        if isinstance(s, tuple):
            return PVP.remove_final_punc(s[0]), s[1]
        return s.rstrip(string.punctuation)

    @staticmethod
    def lowercase_first(s: Union[str, Tuple[str, bool]]):
        """Lowercase the first character"""
        if isinstance(s, tuple):
            return PVP.lowercase_first(s[0]), s[1]
        return s[0].lower() + s[1:]

    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False) \
            -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        tokenizer = self.wrapper.tokenizer  # type: PreTrainedTokenizer

        parts_a, parts_b, block_flag_a, block_flag_b = self.get_parts(example)

        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_b if x]

        # self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length)
        num_special = self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length - num_special)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        # tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else []

        ### add
        assert len(parts_a) == len(block_flag_a)
        assert len(parts_b) == len(block_flag_b)

        block_flag_a = [flag for (part, _), flag in zip(parts_a, block_flag_a) for _ in part]
        block_flag_b = [flag for (part, _), flag in zip(parts_b, block_flag_b) for _ in part]

        assert len(tokens_a) == len(block_flag_a)
        assert len(tokens_b) == len(block_flag_b)

        if tokens_b:
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
            block_flag = tokenizer.build_inputs_with_special_tokens(block_flag_a, block_flag_b)
        else:
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a)
            block_flag = tokenizer.build_inputs_with_special_tokens(block_flag_a)


        block_flag = [item if item in [0, 1] else 0 for item in block_flag]
        assert len(input_ids) == len(block_flag)

        ### return input_ids, token_type_ids
        return input_ids, token_type_ids, block_flag


    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)


    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences (text_a and text_b) containing exactly one
        mask token (or one consecutive sequence of mask tokens for PET with multiple masks). If a task requires only a
        single sequence of text, the second sequence should be an empty list.

        :param example: the input example to process
        :return: Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        Return all verbalizations for a given label.

        :param label: the label
        :return: the list of verbalizations
        """
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.wrapper.config.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits

    def convert_plm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[1] == 1
        logits = torch.squeeze(logits, 1)  # remove second dimension as we always have exactly one <mask> per example
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(lgt) for lgt in logits])
        return cls_logits

    @staticmethod
    def _load_verbalizer_from_file(path: str, pattern_id: int):

        verbalizers = defaultdict(dict)  # type: Dict[int, Dict[str, List[str]]]
        current_pattern_id = None

        with open(path, 'r') as fh:
            for line in fh.read().splitlines():
                if line.isdigit():
                    current_pattern_id = int(line)
                elif line:
                    label, *realizations = line.split()
                    verbalizers[current_pattern_id][label] = realizations

        logger.info("Automatically loaded the following verbalizer: \n {}".format(verbalizers[pattern_id]))

        def verbalize(label) -> List[str]:
            return verbalizers[pattern_id][label]

        return verbalize


class RtePVP(PVP):

    VERBALIZER = {
        "not_entailment": ["No"],
        "entailment": ["Yes"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        # switch text_a and text_b to get the correct order
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b.rstrip(string.punctuation))

        if self.pattern_id == 1:
            
            # searched patterns in fully-supervised.
            # string_list_a = [text_a, '[SEP]', text_b, "?", "the" , self.mask]
            # string_list_a = [text_a, '[SEP]', text_b, "?", "the" , "answer:", self.mask]
            # string_list_a = [text_a, 'Question:', text_b, "?", "the" , self.mask]
            
            # few-shot
            string_list_a = [text_a, 'Question:', text_b, "?", "the", "Answer:", self.mask, "."]
            string_list_b = []
            block_flag_a = [0, 0, 0, 0, 1, 0, 0, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b
        elif self.pattern_id == 8: # X
            string_list_a = ['Premise', text_a, 'Hypothesis:', text_b,
            "Is", "the", "hypothesis", "valid", "?", "The", "answer", "is",
            self.mask, "."]
            string_list_b = []
            block_flag_a = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b
        else:
            raise ValueError("unknown pattern_id.")


    def verbalize(self, label) -> List[str]:
        return RtePVP.VERBALIZER[label]


class CbPVP(PVP):

    VERBALIZER = {
        "contradiction": ["No"],
        "entailment": ["Yes"],
        "neutral": ["Maybe"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        # switch text_a and text_b to get the correct order
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        
        # searched patterns in fully-supervised learning
        # string_list_a = [text_a, ' question: ', text_b, ' true, false or neither? answer:', "the", self.mask]
        # string_list_a = [text_a,  "[SEP]", example.text_b, "?", 'the',  " answer: ", self.mask]
        # string_list_a = [text_a,  "the",  text_b, "?",  "Answer:", self.mask]
        # string_list_a = [text_a, 'the the', 'question:', text_b, '?', 'the the', 'answer:', self.mask]
        # string_list_a = [text_a, "[SEP]", text_b, "?", "the", self.mask]
        
        # few-shot
        if self.pattern_id == 1:

            string_list_a =  [text_a,  "[SEP]", example.text_b, "?", 'the',  " answer: ", self.mask]
            string_list_b = []
            block_flag_a = [0, 0, 0, 0, 1, 0, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b



    def verbalize(self, label) -> List[str]:
        return CbPVP.VERBALIZER[label]


class CopaPVP(PVP):

    def get_parts(self, example: InputExample) -> FilledPattern:

        premise = self.remove_final_punc(self.shortenable(example.text_a))
        choice1 = self.remove_final_punc(self.lowercase_first(example.meta['choice1']))
        choice2 = self.remove_final_punc(self.lowercase_first(example.meta['choice2']))

        question = example.meta['question']
        assert question in ['cause', 'effect']

        example.meta['choice1'], example.meta['choice2'] = choice1, choice2
        num_masks = max(len(get_verbalization_ids(c, self.wrapper.tokenizer, False)) for c in [choice1, choice2])

        if question == "cause":
            joiner = "because"
        else:
            joiner = "so"
            
        # searched patterns in fully-supervised learning
        # string_list_a = [choice1, 'or', choice2, '?', 'the', premise, joiner, 'the', self.mask]
        # string_list_a = [choice1, 'or', choice2, '?', premise, joiner, 'the', self.mask * num_masks]
        # string_list_a = ['"', choice1, '" or "', choice2, '"?', 'the', premise,  'the', joiner, self.mask*num_masks]
        # string_list_a = ['"', choice1, '" or "', choice2, '"?', premise,  , joiner, 'the', self.mask*num_masks]
        
        # few-shot
        if self.pattern_id == 1:
            if question == "cause":

                string_list_a = [choice1, 'or', choice2, '?', premise, 'because', 'the', self.mask * num_masks, '.']
                string_list_b = []
                block_flag_a = [0, 0, 0, 0, 0, 0, 1, 0, 0]
                block_flag_b = []
                assert len(string_list_a) == len(block_flag_a)
                assert len(string_list_b) == len(block_flag_b)
                return string_list_a, string_list_b, block_flag_a, block_flag_b

            elif question == "effect":

                string_list_a = [choice1, 'or', choice2, '?', premise, 'so', 'the', self.mask * num_masks, '.']
                string_list_b = []
                block_flag_a = [0, 0, 0, 0, 0, 0, 1, 0, 0]
                block_flag_b = []
                assert len(string_list_a) == len(block_flag_a)
                assert len(string_list_b) == len(block_flag_b)
                return string_list_a, string_list_b, block_flag_a, block_flag_b

            else:
                raise ValueError("currently not support the kind of questions.")
        else:
            raise ValueError("unknown pattern_ids.")

    def verbalize(self, label) -> List[str]:
        return []


class WscPVP(PVP):

    def get_parts(self, example: InputExample) -> FilledPattern:
        pronoun = example.meta['span2_text']
        target = example.meta['span1_text']
        pronoun_idx = example.meta['span2_index']

        words_a = example.text_a.split()
        words_a[pronoun_idx] = '*' + words_a[pronoun_idx] + '*'
        text_a = ' '.join(words_a)
        text_a = self.shortenable(text_a)

        num_pad = self.rng.randint(0, 3) if 'train' in example.guid else 1
        num_masks = len(get_verbalization_ids(target, self.wrapper.tokenizer, force_single_token=False)) + num_pad
        masks = self.mask * num_masks

        # searched patterns in fully-supervised learning
        # string_list_a = [text_a, "the", "'*", pronoun, "*'", "the", masks]
        # string_list_a = [text_a, "the", "pronoun '*", pronoun, "*' refers to",  masks]
        # string_list_a = [text_a, "the", "pronoun '*", pronoun, "*'", "the", masks]
        # string_list_a = [text_a, "the", "pronoun '*", pronoun, "*' refers to", "the", masks]
        
        # few-shot
        if self.pattern_id == 1:

            string_list_a = [text_a, "the", "pronoun '*", pronoun, "*' refers to",  masks + '.']
            string_list_b = []
            block_flag_a = [0, 1, 0, 0, 0, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b

        elif self.pattern_id == 2:
            string_list_a = ["the", text_a, "the", "pronoun '*", pronoun, "*' refers to",  masks + '.']
            string_list_b = []
            block_flag_a = [1, 0, 1, 0, 0, 0, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b



    def verbalize(self, label) -> List[str]:
        return []


class BoolQPVP(PVP):

    VERBALIZER_A = {
        "False": ["No"],
        "True": ["Yes"]
    }
    """
    VERBALIZER_B = {
        "False": ["false"],
        "True": ["true"]
    }
    """

    def get_parts(self, example: InputExample) -> FilledPattern:
        passage = self.shortenable(example.text_a)
        question = self.shortenable(example.text_b)

        # searched patterns in fully-supervised learning
        # string_list_a = [passage, '.', 'the', 'Question:', question, '?', 'the', 'Answer:', self.mask]
        # string_list_a = [passage, '.', 'the', question, '?', 'the', self.mask]
        # string_list_a = [passage, 'the', question, '?', 'the', self.mask]
        
        # few-shot
        if self.pattern_id == 1:

            string_list_a = [passage, '.', 'the', ' Question: ', question, '? Answer: ', self.mask, '.']
            string_list_b = []
            block_flag_a = [0, 0, 1, 0, 0, 0, 0, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b

        else:
            raise ValueError("unknown pattern_id.")


    def verbalize(self, label) -> List[str]:
        return BoolQPVP.VERBALIZER_A[label]


class MultiRcPVP(PVP):

    VERBALIZER = {
        "0": ["No"],
        "1": ["Yes"]
    }
    
    # search patterns in fully-supervised learning
    # string_list_a = [passage, 'Question: ', question, '?', "Is it", answer, '?', 'the', self.mask]
    # string_list_a = [passage, 'Question: ', question, '?', "the", answer, '?', 'the', self.mask]
    
    
    # few-shot
    def get_parts(self, example: InputExample) -> FilledPattern:
        passage = self.shortenable(example.text_a)
        question = example.text_b
        answer = example.meta['answer']

        if self.pattern_id == 1:
            string_list_a = [passage, '. Question: ', question, '? Is it ', answer, '?', "the", self.mask, '.']
            string_list_b = []
            block_flag_a = [0, 0, 0, 0, 0, 0, 1, 0, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b

        else:
            raise ValueError("unknown pattern_id.")


    def verbalize(self, label) -> List[str]:
        return MultiRcPVP.VERBALIZER[label]


class WicPVP(PVP):
    VERBALIZER = {
        "F": ["No"],
        "T": ["Yes"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        word = "*" + example.meta['word'] + " *"

        # searched patterns in fully-supervised learning
        # string_list_a = [text_a, '[SEP]', text_b, "the" , word, '?', self.mask]
        # string_list_a = [text_a, '[SEP]', text_b, "the" , word, '?', "the", self.mask]
        # string_list_a = [text_a, 'the', text_b, "the" , word, '?', "the", self.mask]
        
        # few-shot
        if self.pattern_id == 1:
            
            string_list_a = [text_a, '[SEP]', text_b , "the", word + '?', self.mask]
            string_list_b = []
            block_flag_a = [0, 0, 0, 1, 0, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b


        elif self.pattern_id == 2:
            string_list_a = [text_a, '[SEP]', text_b, "the" , word + '?', "the", self.mask]
            string_list_b = []
            block_flag_a = [0, 0, 0, 1, 0, 1, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b

        elif self.pattern_id == 3:
            string_list_a = ["the", text_a, '[SEP]', text_b, "the" , word + '?', "the", self.mask]
            string_list_b = []
            block_flag_a = [1, 0, 0, 0, 1, 0, 1, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b

        elif self.pattern_id == 4:
            string_list_a = ["the", text_a, '[SEP]', text_b, "the" , word + '?', "the", self.mask, "the"]
            string_list_b = []
            block_flag_a = [1, 0, 0, 0, 1, 0, 1, 0, 1]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b
        else:
            raise ValueError("unknown pattern_id.")

    def verbalize(self, label) -> List[str]:
        return WicPVP.VERBALIZER[label]


class AGNewsPVP(PVP):

    VERBALIZER = {
        "World": ["World"],
        "Sports": ["Sports"],
        "Business": ["Business"],
        "Sci/Tech": ["Tech"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        # text_b == None

        if self.pattern_id == 1:
            string_list_a = [self.mask, 'News:', "the", text_a]
            string_list_b = []
            block_flag_a = [0, 0, 1, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b
        
        else:
            raise ValueError("unknown pattern_id.")

    def verbalize(self, label) -> List[str]:
        return AGNewsPVP.VERBALIZER[label]


class YelpPVP(PVP):

    VERBALIZER = {
        "1 star": ["terrible"],
        "2 star": ["bad"],
        "3 stars": ["okay"],
        "4 stars": ["good"],
        "5 stars": ["great"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        # text_b == None

        if self.pattern_id == 2:
            string_list_a = ['It', "was", self.mask, '.', text_a]
            string_list_b = []
            block_flag_a = [1, 1, 0, 0, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b

    def verbalize(self, label) -> List[str]:
        return YelpPVP.VERBALIZER[label] #


class SmsSpamPVP(PVP):

    VERBALIZER = {
        "ham": ["No"],
        "spam": ["Yes"],
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        # text_b == None

        if self.pattern_id == 2:
            string_list_a = ['Email:', text_a, "Is it a spam?", "Answer", ":", self.mask]
            string_list_b = []
            block_flag_a = [0, 0, 0, 1, 1, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b
        
        else:
            raise ValueError("unknown pattern_id.")
    
    def verbalize(self, label) -> List[str]:
        return SmsSpamPVP.VERBALIZER[label]


class EnronSpamPVP(PVP):

    VERBALIZER = {
        "ham": ["No"],
        "spam": ["Yes"],
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        # text_b == None

        if self.pattern_id == 2:
            string_list_a = ['Email:', text_a, "Is it a spam?", "Answer", ":", self.mask]
            string_list_b = []
            block_flag_a = [0, 0, 0, 1, 1, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b
        
        else:
            raise ValueError("unknown pattern_id.")

    def verbalize(self, label) -> List[str]:
        return EnronSpamPVP.VERBALIZER[label]


PVPS = {
    'rte': RtePVP,
    'wic': WicPVP,
    'cb': CbPVP,
    'wsc': WscPVP,
    'boolq': BoolQPVP,
    'copa': CopaPVP,
    'multirc': MultiRcPVP,
    'ag_news': AGNewsPVP,
    "yelp": YelpPVP,
    'sms_spam': SmsSpamPVP,
    'enron_spam': EnronSpamPVP,
}
