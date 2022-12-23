import string
import pandas as pd
import numpy as np
import hashlib


class FileReaderSingleton:
    __instance = None
    def __init__(self, filepath: str):
        if FileReaderSingleton.__instance is not None \
        and FileReaderSingleton.__instance.filepath == filepath:
            raise Exception(
                'This is a singleton class. Use readlines method instead.'
                )
        self.filepath = filepath
        with open(filepath, 'r') as f:
            self.lines = f.readlines()
        
    @staticmethod
    def readlines(filepath: str) -> list:
        if FileReaderSingleton.__instance is None \
        or FileReaderSingleton.__instance.filepath != filepath:
            FileReaderSingleton.__instance = FileReaderSingleton(filepath)
        return FileReaderSingleton.__instance.lines


class AntiPlagiarismSimulator:
    @staticmethod
    def remove_punctuation(s: str) -> str:
        """
        IN: a string s
        OUT: the string s without punctuation
        """
        return s.translate(
            str.maketrans('', '', string.punctuation)
            )

    @staticmethod
    def compute_hash(s: str, hash_dim: int) -> int:
        s_hash = hashlib.md5(s.encode('utf-8'))
        int_hash = int(s_hash.hexdigest(), 16)
        return int_hash % hash_dim

    def __init__(
            self,
            filepath: str,
            window_size: int):
        self.window_size = window_size
        self.text_lines = FileReaderSingleton.readlines(filepath)
        self.process_text()

    def process_text(self):
        self.text_lines = pd.Series(self.text_lines) \
                    .apply(self.remove_punctuation) \
                    .apply(lambda s: str.lower(s[:-1]))
        self.words = self.text_lines.apply(str.split) \
                                    .explode(ignore_index=True) \
                                    .dropna()
        self.distinct_words: np.ndarray = pd.unique(self.words)
        self.sentences = np.lib.stride_tricks.sliding_window_view(
            x=self.words.values,
            window_shape=(self.window_size,)
            )
        self.sentences = pd.DataFrame(self.sentences) \
                        .apply(' '.join, axis=1) 
        self.distinct_sentences = pd.Series(pd.unique(self.sentences))
        self.set_distinct_sentences = set(self.distinct_sentences)

    def store_hash_sentences(self, hash_dim: int):
        self.hash_sentences = self.distinct_sentences.apply(
            lambda s: AntiPlagiarismSimulator.compute_hash(s, hash_dim)
            )
        self.distinct_hash_sentences = pd.unique(self.hash_sentences)
        self.set_distinct_hash_sentences = set(self.distinct_hash_sentences)
    
    def false_positive_probability(self, hash_dim: int) -> float:
        self.store_hash_sentences(hash_dim)
        return len(self.distinct_hash_sentences) / hash_dim
