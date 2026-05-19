import spacy
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from collections import Counter


class Multi30kDataset(Dataset):
    def __init__(self, split='train', min_freq=2, max_len=100):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.min_freq = min_freq
        self.max_len = max_len

        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        self.dataset = load_dataset("bentrevett/multi30k", split=split)

        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            self.spacy_de = spacy.blank("de")

        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            self.spacy_en = spacy.blank("en")
        # here
        self.special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]
        self.unk_idx = 0
        self.pad_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3

        self.src_vocab = None
        self.tgt_vocab = None
        self.src_itos = None
        self.tgt_itos = None

        self.build_vocab()
        self.data = self.process_data()

    def tokenize_de(self, text):
        return [token.text.lower() for token in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [token.text.lower() for token in self.spacy_en.tokenizer(text)]

    def _get_text_pair(self, example):
        """
        Handles common Multi30k formats.
        """
        if "de" in example and "en" in example:
            return example["de"], example["en"]

        if "translation" in example:
            translation = example["translation"]
            return translation["de"], translation["en"]

        raise KeyError(
            "Expected dataset example to contain either keys 'de' and 'en' "
            "or a 'translation' dictionary."
        )

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        train_dataset = load_dataset("bentrevett/multi30k", split="train")

        src_counter = Counter()
        tgt_counter = Counter()

        for example in train_dataset:
            de_text, en_text = self._get_text_pair(example)
            src_counter.update(self.tokenize_de(de_text))
            tgt_counter.update(self.tokenize_en(en_text))

        self.src_itos = list(self.special_tokens)
        self.tgt_itos = list(self.special_tokens)

        for token, freq in sorted(src_counter.items(), key=lambda x: (-x[1], x[0])):
            if freq >= self.min_freq and token not in self.src_itos:
                self.src_itos.append(token)

        for token, freq in sorted(tgt_counter.items(), key=lambda x: (-x[1], x[0])):
            if freq >= self.min_freq and token not in self.tgt_itos:
                self.tgt_itos.append(token)

        self.src_vocab = {token: idx for idx, token in enumerate(self.src_itos)}
        self.tgt_vocab = {token: idx for idx, token in enumerate(self.tgt_itos)}

        return self.src_vocab, self.tgt_vocab
    
    def numericalize_src(self, tokens):
        return [self.src_vocab.get(token, self.unk_idx) for token in tokens]

    def numericalize_tgt(self, tokens):
        return [self.tgt_vocab.get(token, self.unk_idx) for token in tokens]

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        processed = []

        for example in self.dataset:
            de_text, en_text = self._get_text_pair(example)

            src_tokens = self.tokenize_de(de_text)[: self.max_len - 2]
            tgt_tokens = self.tokenize_en(en_text)[: self.max_len - 2]

            src_indices = (
                [self.sos_idx]
                + self.numericalize_src(src_tokens)
                + [self.eos_idx]
            )

            tgt_indices = (
                [self.sos_idx]
                + self.numericalize_tgt(tgt_tokens)
                + [self.eos_idx]
            )

            processed.append(
                {
                    "src": torch.tensor(src_indices, dtype=torch.long),
                    "tgt": torch.tensor(tgt_indices, dtype=torch.long),
                    "src_text": de_text,
                    "tgt_text": en_text,
                }
            )

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, pad_idx=1):
    """
    Pads source and target sequences in a batch.
    """
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=pad_idx,
    )

    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=pad_idx,
    )

    return src_padded, tgt_padded