'''Modelo TF-ID'''
import argparse
from collections import Counter
import math
from operator import or_
from functools import reduce
from json import dump

def split_words(line):
    return line.split()


def raw_frequency(text):
    '''Retorna la frequencia de cada palabra dentro
    del texto'''
    counter = Counter()
    for line in text:
        counter.update(line.split())

    return  counter


def tf(f):
    return 0 if f <= 0 else (1 + math.log2(f))


def count_docs_found(frequencys_per_doc, word):
    '''Retorna el número de documentos en los que se
    encuentra la palabra'''
    return sum([1 for freq in frequencys_per_doc.values() 
                if word in freq])


def inverse_frequency(N, n):
    '''Frequencia inversa de una palabra en un corpus
    de N documentos y con aparicion de la palabra en n 
    documentos'''
    return math.log2(N/n) if n > 0 else 0


class TFID:
    def __init__(self, ):
        self.N = 0
        self.raw_frequencys_per_doc = {}
        self.inverse_frequencys = {}
        self.documents = []

    def train(self, documents):
        # Total documents
        self.documents = documents
        self.N = len(documents)

        # Raw frequencys
        self.raw_frequencys_per_doc = {name: raw_frequency(fp) 
                                for (name, fp) in documents}

        # Inverse frequencys per word
        all_words = reduce(lambda a, c: a | c, [freq.keys() for freq in self.raw_frequencys_per_doc.values()])
        self.inverse_frequencys = {word: inverse_frequency(self.N, count_docs_found(self.raw_frequencys_per_doc, word))
                            for word in all_words}

        # Final weigth of each word on each document
        self.weigths_per_doc = {}
        for (name, _) in self.documents:
            self.weigths_per_doc[name] = {word: tf(self.raw_frequencys_per_doc[name][word]) * self.inverse_frequencys[word]
                                          for word in self.raw_frequencys_per_doc[name]}

        with open('resultados_parciales/pesos_por_documento.txt', encoding='utf8', mode='w') as file:
            dump(self.weigths_per_doc, file, indent=4)

        return self.weigths_per_doc

    def execute(self, query):
        range_per_doc = {}
        query_words = set(query.split())

        for (name, _) in self.documents:
            range_per_doc[name] = 0
            for word in query_words:
                range_per_doc[name] += self.weigths_per_doc[name].get(word, 0)

        return range_per_doc
