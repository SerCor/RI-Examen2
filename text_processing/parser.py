

def parser_file(doc):
    for line in doc:
        doc_id, _, *content = line  
        yield doc_id, ' '.join(content)
