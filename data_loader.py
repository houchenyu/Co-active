# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import random

import numpy as np
import pandas as pd
import os
import csv


def get_task_processor(task, data_dir):
    """
    A TSV processor for stsa, trec and snips dataset.
    """
    if task == 'sst2':
        return sst2DataProcessor(data_dir=data_dir, skip_header=False, label_col=0, text_col=1)
    elif task == 'sst5':
        return sst5DataProcessor(data_dir=data_dir, skip_header=False, label_col=0, text_col=1)
    elif task == 'amazon':
        return amazonDataProcessor(data_dir=data_dir, skip_header=False, label_col=0, text_col=1)
    elif task == 'yelp':
        return yelpVataProcessor(data_dir=data_dir, skip_header=False, label_col=1, text_col=0)
    else:
        raise ValueError('Unknown task')


def get_data(task, data_dir, data_seed=159):
    random.seed(data_seed)
    processor = get_task_processor(task, data_dir)

    examples = dict()

    examples['train'] = processor.get_train_examples()
    examples['dev'] = processor.get_dev_examples()
    examples['test'] = processor.get_test_examples()

    for key, value in examples.items():
        print('#{}: {}'.format(key, len(value)))
    return examples, processor.get_labels(task)


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class errorExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,confidence = None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.confidence = confidence

class errorFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,confidence):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids#[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
       
        self.label_id = label_id
        self.confidence = confidence

    def __getitem__(self, item):
        return [self.input_ids, self.input_mask,
                self.segment_ids, self.label_id,self.confidence][item]

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,confidence):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids#[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
      
        self.label_id = label_id

    def __getitem__(self, item):
        return [self.input_ids, self.input_mask,
                self.segment_ids, self.label_id][item]


class DatasetProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, task_name):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class sst2DataProcessor(DatasetProcessor):
    """Processor for dataset to be augmented."""

    def __init__(self, data_dir, skip_header, label_col, text_col):
        self.data_dir = data_dir
        self.skip_header = skip_header
        self.label_col = label_col
        self.text_col = text_col

    def get_train_examples(self):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(self.data_dir, "train.csv")), "train")
        return self._create_examples(os.path.join(self.data_dir, "train.csv"), "train")

    def get_error_examples(self):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(self.data_dir, "train.csv")), "train")
        return self._create_errorexamples( "data/errorData_c.csv", "error")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "dev.csv")), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            # self._read_tsv(os.path.join(self.data_dir, "test.csv")), "test")
            self._read_tsv( "data/test.csv"), "test")

    def get_labels(self, task_name):
        """add your dataset here"""
        labels = set()
        # with open(os.path.join(self.data_dir, "train.csv"), "r") as f:
        #     reader = csv.reader(f, delimiter="\t")
        #     lines = []
        #     for line in reader:
        #         lines.append(line)
        # # with open(os.path.join(self.data_dir, "train.csv"), "r") as in_file:
        # for line in lines:
        #     labels.add(line[self.label_col])
        content = pd.read_csv(os.path.join("data/allData.csv"), sep=None, engine='python', error_bad_lines=False)
        l=content['label']
        for i in l:
            labels.add(i)
        return sorted(labels)

    def _create_examples(self, datdir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        content= pd.read_csv(datdir,sep=None,engine='python',error_bad_lines=False)
        data=content['data']
        l=content['label']
        # for (i, line) in enumerate(lines):
        #     if self.skip_header and i == 0:
        #         continue
        for i in range(len(data)):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i]
            label = l[i]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def _create_errorexamples(self, datdir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        content= pd.read_csv(datdir,sep=None,engine='python',error_bad_lines=False)
        data=content['data']
        l=content['label']
        c= content['confidence']
        # for (i, line) in enumerate(lines):
        #     if self.skip_header and i == 0:
        #         continue
        for i in range(len(data)):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i]
            label = l[i]
            confidence = c[i]
            examples.append(
                errorExample(guid=guid, text_a=text_a, label=label,confidence=confidence))
        return examples

class sst5DataProcessor(DatasetProcessor):
    """Processor for dataset to be augmented."""

    def __init__(self, data_dir, skip_header, label_col, text_col):
        self.data_dir = data_dir
        self.skip_header = skip_header
        self.label_col = label_col
        self.text_col = text_col

    def get_train_examples(self):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(self.data_dir, "train.csv")), "train")
        return self._create_examples(os.path.join(self.data_dir, "train.csv"), "train")

    def get_error_examples(self):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(self.data_dir, "train.csv")), "train")
        return self._create_errorexamples( "data/errorData_c.csv", "error")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "dev.csv")), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            # self._read_tsv(os.path.join(self.data_dir, "test.csv")), "test")
            self._read_tsv( "data/test.csv"), "test")

    def get_labels(self, task_name):
        """add your dataset here"""
        labels = set()
        # with open(os.path.join(self.data_dir, "train.csv"), "r") as f:
        #     reader = csv.reader(f, delimiter="\t")
        #     lines = []
        #     for line in reader:
        #         lines.append(line)
        # # with open(os.path.join(self.data_dir, "train.csv"), "r") as in_file:
        # for line in lines:
        #     labels.add(line[self.label_col])
        content = pd.read_csv(os.path.join("data/allData.csv"), sep=None, engine='python', error_bad_lines=False)
        l=content['label']
        for i in l:
            labels.add(i)
        return sorted(labels)

    def _create_examples(self, datdir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        content= pd.read_csv(datdir,sep=None,engine='python',error_bad_lines=False)
        data=content['data']
        l=content['label']
        # for (i, line) in enumerate(lines):
        #     if self.skip_header and i == 0:
        #         continue
        for i in range(len(data)):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i]
            label = l[i]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def _create_errorexamples(self, datdir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        content= pd.read_csv(datdir,sep=None,engine='python',error_bad_lines=False)
        data=content['data']
        l=content['label']
        c= content['confidence']
        # for (i, line) in enumerate(lines):
        #     if self.skip_header and i == 0:
        #         continue
        for i in range(len(data)):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i]
            label = l[i]
            confidence = c[i]
            examples.append(
                errorExample(guid=guid, text_a=text_a, label=label,confidence=confidence))
        return examples

class amazonDataProcessor(DatasetProcessor):
    """Processor for dataset to be augmented."""

    def __init__(self, data_dir, skip_header, label_col, text_col):
        self.data_dir = data_dir
        self.skip_header = skip_header
        self.label_col = label_col
        self.text_col = text_col

    def get_train_examples(self):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(self.data_dir, "train.csv")), "train")
        return self._create_examples(os.path.join(self.data_dir, "train_0.4_asym.csv"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "dev.csv")), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "test.csv")), "test")
            # self._read_tsv( "data/test.csv"), "test")

    def get_labels(self, task_name):
        """add your dataset here"""
        labels = set()
        l=np.loadtxt(os.path.join(self.data_dir, "labellist.txt"))
        for i in l:
            labels.add(i)
        return sorted(labels)

    def _create_examples(self, datdir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        content= pd.read_csv(datdir,sep=None,engine='python',error_bad_lines=False)
        data=content['data']
        l=content['label']

        for i in range(len(data)):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i]
            label = l[i]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class yelpDataProcessor(DatasetProcessor):
    """Processor for dataset to be augmented."""

    def __init__(self, data_dir, skip_header, label_col, text_col):
        self.data_dir = data_dir
        self.skip_header = skip_header
        self.label_col = label_col
        self.text_col = text_col

    def get_train_examples(self):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(self.data_dir, "train.csv")), "train")
        return self._create_examples(os.path.join(self.data_dir, "train_0.4_asym.csv"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "dev.csv")), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "test.csv")), "test")
            # self._read_tsv( "data/test.csv"), "test")

    def get_labels(self, task_name):
        """add your dataset here"""
        labels = set()
        l=np.loadtxt(os.path.join(self.data_dir, "labellist.txt"))
        for i in l:
            labels.add(i)
        return sorted(labels)

    def _create_examples(self, datdir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        content= pd.read_csv(datdir,sep=None,engine='python',error_bad_lines=False)
        data=content['data']
        l=content['label']

        for i in range(len(data)):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i]
            label = l[i]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

