import numpy as np
import torch,pandas as pd
import random,csv,os

def _fix_cls_to_idx(ds):
    for cls in ds.class_to_idx:
        ds.class_to_idx[cls] = int(cls)

def get_task_processor(args):
    """
    A TSV processor for stsa, trec and snips dataset.
    """
    if args.task == 'SST5':
        return SST5DataProcessor(args=args, skip_header=False, label_col=0, text_col=1)
    elif args.task == 'SST2':
            return SST2DataProcessor(args=args, skip_header=False, label_col=0, text_col=1)
    elif args.task == 'AG':
            return AGDataProcessor(args=args, skip_header=False, label_col=0, text_col=1)
    elif args.task == 'AG4':
            return AG4DataProcessor(args=args, skip_header=False, label_col=0, text_col=1)
    else:
        raise ValueError('Unknown task')

def get_data(args, data_seed=159):
    random.seed(data_seed)
    processor = get_task_processor(args.task)

    examples = dict()

    examples['train'] = processor.get_train_examples()
    examples['dev'] = processor.get_dev_examples()
    examples['test'] = processor.get_test_examples()
    examples['clean_valid'] = processor.get_clean_valid_examples()

    for key, value in examples.items():
        print('#{}: {}'.format(key, len(value)))
    return examples, processor.get_labels()

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

class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class DatasetProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_clean_valid_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
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

class SST5DataProcessor(DatasetProcessor):
    """Processor for dataset to be augmented."""

    def __init__(self, args, skip_header, label_col, text_col):
        self.args = args
        self.trainData = args.trainData
        self.skip_header = skip_header
        self.label_col = label_col
        self.text_col = text_col

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(self.trainData), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.args.dataPath, "dev.csv")), "dev")

    def get_clean_valid_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.args.dataPath, "clean_valid.csv")), "clean_valid")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.args.dataPath, "test.csv")), "test")

    def get_labels(self):
        """add your dataset here"""
        labellist = np.loadtxt(os.path.join(self.args.dataPath, "labellist.txt"))
        return labellist

   
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

class SST2DataProcessor(DatasetProcessor):
    """Processor for dataset to be augmented."""

    def __init__(self, args, skip_header, label_col, text_col):
        self.args = args
        self.trainData = args.trainData
        self.skip_header = skip_header
        self.label_col = label_col
        self.text_col = text_col

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(self.trainData), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.args.dataPath, "dev.csv")), "dev")

    def get_clean_valid_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.args.dataPath, "clean_valid.csv")), "clean_valid")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.args.dataPath, "test.csv")), "test")

    def get_labels(self):
        """add your dataset here"""
        labellist = np.loadtxt(os.path.join(self.args.dataPath, "labellist.txt"))
        return labellist

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

class AGDataProcessor(DatasetProcessor):
    """Processor for dataset to be augmented."""

    def __init__(self, args, skip_header, label_col, text_col):
        self.args = args
        self.trainData = args.trainData
        self.skip_header = skip_header
        self.label_col = label_col
        self.text_col = text_col

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(self.trainData), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.args.dataPath, "dev.csv")), "dev")

    def get_clean_valid_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.args.dataPath, "clean_valid.csv")), "clean_valid")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.args.dataPath, "test.csv")), "test")

    def get_labels(self):
        """add your dataset here"""
        labellist = np.loadtxt(os.path.join(self.args.dataPath, "labellist.txt"))
        return labellist

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

class AG4DataProcessor(DatasetProcessor):
    """Processor for dataset to be augmented."""

    def __init__(self, args, skip_header, label_col, text_col):
        self.args = args
        self.trainData = args.trainData
        self.skip_header = skip_header
        self.label_col = label_col
        self.text_col = text_col

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(self.trainData), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.args.dataPath, "dev.csv")), "dev")

    def get_clean_valid_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.args.dataPath, "clean_valid.csv")), "clean_valid")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.args.dataPath, "test.csv")), "test")

    def get_labels(self):
        """add your dataset here"""
        labellist = np.loadtxt(os.path.join(self.args.dataPath, "labellist.txt"))
        return labellist

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

if __name__ == '__main__':
    labellist = np.loadtxt('labellist.txt')
    print('ok')