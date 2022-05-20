import os

import torch
from d2l import torch as d2l
from torch.utils.data import Dataset

from Utils.dl_utils import reduce_sum, astype
from Utils.download_zip import download, zip
from torch.utils import data


def download_zip_fra_eng(path):
    """Load the English-French dataset.

    Defined in :numref:`sec_machine_translation`"""
    if not os.path.exists(f'{path}/fra-eng/fra.txt'):
        file_path = download('http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip', path)
        zip(path, file_path, f'{path}/fra-eng')
    with open(os.path.join(f'{path}/fra-eng/fra.txt'), 'rb') as f:
        return f.read()

def preprocess_fra_eng(text):
    """Preprocess the English-French dataset.

    Defined in :numref:`sec_machine_translation`"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.decode().replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def split_fra_eng(text, num_examples=None):
    """Tokenize the English-French dataset.

    Defined in :numref:`sec_machine_translation`"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs.

    Defined in :numref:`sec_machine_translation`"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.

    Defined in :numref:`sec_machine_translation`"""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

def build_array_fra_eng(lines, vocab, num_steps):
    """ Transform text sequences of machine translation into minibatches.

    Defined in :numref:`subsec_mt_data_loading`"""

    # Transform sequence to words idxs and add 'eos' tag
    lines = [vocab[words] + [vocab['<eos>']] for words in lines]

    # truncate or padding sentences
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])

    valid_len = reduce_sum(
        astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len

def load_fra_eng(root_dir, batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset.

    Defined in :numref:`subsec_mt_data_loading`"""
    text = preprocess_fra_eng(download_zip_fra_eng(root_dir))
    source, target = split_fra_eng(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_fra_eng(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_fra_eng(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

def load_array(data_arrays, batch_size, is_train=False):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

class FRA_ENG(Dataset):

    def __init__(self, dir, num_steps=10, train=True):

        self.train = train

        if not os.path.exists(f'{dir}/fra-eng/fra.txt'):
            file_path = download('http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip', dir)
            zip(dir, file_path, f'{dir}/fra-eng')

        with open(os.path.join(f'{dir}/fra-eng/fra.txt'), 'rb') as f:
            text = f.read()

        processed_text = self._preprocess_fra_eng(text)

        eng_text, fra_text = self._split_fra_eng(processed_text)

        if train:

            self.eng_vocab = d2l.Vocab(eng_text, min_freq=2,
                                  reserved_tokens=['<pad>', '<bos>', '<eos>'])
            self.fra_vocab = d2l.Vocab(fra_text, min_freq=2,
                                  reserved_tokens=['<pad>', '<bos>', '<eos>'])

            self.ENG, self.ENG_VALID_LENS = self._build_array_fra_eng(eng_text, self.eng_vocab, num_steps)
            self.FRA, self.FRA_VALID_LENS = self._build_array_fra_eng(fra_text, self.fra_vocab, num_steps, tgt=True)
        else:
            self.eng = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .', 'hi .', 'run !', 'fire !']
            self.fra = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .', 'salut !', 'courez !', 'au feu !']
            # self.eng = [' '.join(words) for words in eng_text[: 20]]
            # self.fra = [' '.join(words) for words in fra_text[: 20]]

    def __getitem__(self, index):
        if self.train:
            return (self.ENG[index], self.ENG_VALID_LENS[index]), (self.FRA[index], self.FRA_VALID_LENS[index])
        else:
            return self.eng[index], self.fra[index]

    def __len__(self):

        if self.train:
            return self.ENG.shape[0]
        else:
            return len(self.eng)


    def _preprocess_fra_eng(self, text):
        """Preprocess the English-French dataset.

        Defined in :numref:`sec_machine_translation`"""

        def no_space(char, prev_char):
            return char in set(',.!?') and prev_char != ' '

        # Replace non-breaking space with space, and convert uppercase letters to
        # lowercase ones
        text = text.decode().replace('\u202f', ' ').replace('\xa0', ' ').lower()
        # Insert space between words and punctuation marks
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
               for i, char in enumerate(text)]
        return ''.join(out)

    def _split_fra_eng(self, text, num_examples=600):
        """Tokenize the English-French dataset.

        Defined in :numref:`sec_machine_translation`"""
        source, target = [], []
        for i, line in enumerate(text.split('\n')):
            if num_examples and i > num_examples:
                break
            parts = line.split('\t')
            if len(parts) == 2:
                source.append(parts[0].split(' '))
                target.append(parts[1].split(' '))
        return source, target

    def _build_array_fra_eng(self, lines, vocab, num_steps, tgt=False):
        """ Transform text sequences of machine translation into minibatches.

        Defined in :numref:`subsec_mt_data_loading`"""

        # Transform sequence to words idxs and add 'eos' tag
        lines = [vocab[words] + [vocab['<eos>']] for words in lines]

        # truncate or padding sentences
        array = d2l.tensor([truncate_pad(
            l, num_steps, vocab['<pad>']) for l in lines])

        valid_len = reduce_sum(
            astype(array != vocab['<pad>'], d2l.int32), 1)

        return array, valid_len

