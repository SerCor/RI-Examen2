import argparse
import sys
import os
from text_processing.source import from_file
from text_processing.pipeline import (
    drop_punctuations, to_lower,
    drop_stop_words, drop_numbers, parse,
    tee_and_write_to_file
)
from functools import partial, wraps
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from tf_idf import TFID
from json import dump


def worker(path, fd_partial1, fd_partial2):
    print(f'[+] Inicio de procesamiento de {path}')
    tee_and_write_lexic = partial(tee_and_write_to_file, fd=fd_partial1)
    tee_and_write_stopwords = partial(tee_and_write_to_file, fd=fd_partial2)

    base_pipeline = (
        from_file(path)
        >> parse
        >> drop_numbers
        >> drop_punctuations
        >> to_lower
        >> tee_and_write_lexic
        >> drop_stop_words
        >> tee_and_write_stopwords
    )

    return base_pipeline


def main():
    open('resultados_parciales/lexic.txt', encoding='utf8', mode='w').close()
    open('resultados_parciales/drop_stopwords.txt', encoding='utf8', mode='w').close()
    filename = 'Noticias 2P.txt'
    with open('resultados_parciales/lexic.txt', encoding='utf8', mode='a+') as fd1, \
         open('resultados_parciales/drop_stopwords.txt', encoding='utf8', mode='a+') as fd2:

        clean_news = worker(filename, fd1, fd2).value
        prepare_news = [(new_id, [new]) for new_id, new in enumerate(clean_news, start=1)]
        querys = (
            'internet libre para todos', 'reconocimiento de estudios',
            'situacion de cuba en la actualidad', 'nuevas propuestas de leyes',
            'impacto de las redes sociales'
        )

        with open('resultados_parciales/representacion_query.txt', encoding='utf8', mode='w') as file:
            for query in querys:
                tfid = TFID()
                tfid.train(prepare_news)

                table = tfid.execute(query)
                file.write(f'Consulta: {query}\n')
                dump(table, file, indent=4)
                file.write(f'\n')


if __name__ == '__main__':
    main()
