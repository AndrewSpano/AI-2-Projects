import os
import re
import json
import pandas as pd
import tqdm.notebook as tq
from collections import namedtuple



def parse_squad(raw_squad):
    """ function used to parse the squad dataset """

    def fix_index(context, idx, step=1):
        """ function used to fix an index of an answer such that it covers the while token, not
            just part of it """

        # if character is in the extremes of a string, it covers a whole token, so return
        if not 0 <= idx + step < len(context):
            return idx

        def is_letter(s):
            """ return true if the given character s is a letter """
            return bool(re.match(r'[a-zA-Z]', s))

        def is_allowed_character(s):
            """ returns true if the given character is in a list of allowed characters """
            allowed_characters = ['ṣ', '£']
            return s in allowed_characters

        def is_digit(s):
            """ return true if the given character s is a digit """
            return bool(re.match(r'\d', s))

        def is_allowed_punctuation(s):
            """ return true if the given character is either a forslash '⁄' or a dash '−' """
            allowed_punctuation = ['⁄', '−']
            return s in allowed_punctuation

        def character_matches(c, border_c):
            """ return true if the character "matches" the border character; Else false """
            return ((is_letter(c) or is_digit(c) or is_allowed_character(c)) and \
                    (is_digit(border_c) or is_letter(border_c))) or \
                   (is_allowed_punctuation(c) and is_digit(border_c))

        # get the border character
        border_c = context[idx]

        # fix the index -> make sure it covers the whole token it belongs to
        while 0 <= idx + step < len(context) and character_matches(context[idx + step], border_c):
            idx += step

        return idx

    # namedtuple used to store all the squad examples
    squad_example = namedtuple('SQuAD_Example', 'id context question answer ans_start ans_end ' \
                                                'orig_answer orig_ans_start orig_ans_end ' \
                                                'is_impossible')
    # list that will contain all the examples
    examples = []

    # iterate through all the available question and construct the corresponding squad_example
    for document in raw_squad['data']:
        for paragraph in document['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:

                orig_ans_text = qa['answers'][0]['text'] if qa['answers'] else ''
                orig_ans_start = qa['answers'][0]['answer_start'] if qa['answers'] else -1
                orig_ans_end = orig_ans_start + len(orig_ans_text) if orig_ans_start != -1 else -1

                ans_start = fix_index(context, orig_ans_start, step=-1) if orig_ans_start != -1 \
                                                                        else -1
                ans_end = fix_index(context, orig_ans_end-1, step=1) + 1 if orig_ans_end != -1 \
                                                                         else -1
                ans_text = context[ans_start:ans_end]

                ex = squad_example(qa['id'], context, qa['question'], ans_text, ans_start, ans_end,
                                   orig_ans_text, orig_ans_start, orig_ans_end, qa['is_impossible'])
                examples.append(ex)

    return examples



def preprocess_dataframe(df, tokenizer, maxlen=512):
    """ preprocess a dataframe that contains squad examples so that it matches the BERT input """

    def find_beggining_of_subsequence(seq, subseq):
        """ function used to find the index of a subsequence in a larger sequence.
            e.g. seq = [1, 2, 3, 4, 5, 6, 7, 8], subseq = [4, 5, 6], then the function returns 3 """
        for i in range(len(seq) - len(subseq)):
            if seq[i:i+len(subseq)] == subseq:
                return i

        return None

    def fix_bert_input(bert_input, ans_start, ans_end, question_len, maxlen):
        """ fixes the input of a bert example such that it does not exceed the maximum permitted
            length of a sequence """
        context_beginning = question_len
        while len(bert_input) > maxlen or ans_end >= maxlen:
            if ans_start - context_beginning > len(bert_input) - ans_end:
                ans_start -= 1
                ans_end -= 1
                del bert_input[context_beginning]
            else:
                del bert_input[-1]

    # namedtuple used to store all the input examples
    bert_io = namedtuple('BERT_IO', 'id bert_input question_len ans_start ans_end is_impossible')
    # list that will contain all the examples
    examples = []

    # iterate through each example (row) in the dataframe
    for idx, row in tq.tqdm(df.iterrows(), total=df.shape[0], position=0, leave=True):

        # construct the BERT input: [CLS] [tokenized_question] [SEP] [tokenized_context] [SEP]
        cls_question_sep = "[CLS] " + row['question'] + " [SEP] "
        context_sep = row['context'] + " [SEP]"
        tokenized_bert_input = tokenizer.tokenize(cls_question_sep + context_sep)

        # find the length of the question + [CLS] and [SEP] tokens
        question_len = len(tokenizer.tokenize(cls_question_sep))

        # find the indices of the answer in the bert input
        if row['ans_start'] == -1:
            ans_start = 0
            ans_end = 0
        else:
            # encode the answer in order to search it in the bert input
            tokenized_answer = tokenizer.tokenize(row['answer'])
            ans_start = find_beggining_of_subsequence(tokenized_bert_input, tokenized_answer)
            ans_end = ans_start + len(tokenized_answer)

            # fix the BERT input if it is too large
            if ans_end >= maxlen:
                fix_bert_input(tokenized_bert_input, ans_start, ans_end, question_len, maxlen)
                ans_start = find_beggining_of_subsequence(tokenized_bert_input, tokenized_answer)
                ans_end = ans_start + len(tokenized_answer)

        # fix the BERT input if it is too large
        if len(tokenized_bert_input) > maxlen:
            fix_bert_input(tokenized_bert_input, ans_start, ans_end, question_len, maxlen)

        # convert the tokens to word indices in the BERT vocabulary
        bert_input = tokenizer.convert_tokens_to_ids(tokenized_bert_input)

        # construct the corresponding example
        ex = bert_io(row['id'], bert_input, question_len, ans_start, ans_end, row['is_impossible'])
        examples.append(ex)

    return examples



def prepare_dataframe(df, maxlen):
    """ function that pads the bert input of a DataFrame, adds the segment ID and mask ID tokens """
    # build the segment IDs
    segment_ids = df['question_len'].apply(lambda qlen: qlen * [0] + (maxlen - qlen) * [1])
    # build the final the bert inputs
    padded_bert_input = df['bert_input'].apply(lambda b_inp: b_inp + (maxlen - len(b_inp)) * [0])
    # build the mask IDs
    mask_ids = df['bert_input'].apply(lambda b_inp: len(b_inp) * [1] + (maxlen - len(b_inp)) * [0])

    return padded_bert_input, mask_ids, segment_ids
