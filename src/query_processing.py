import re

def preprocess_query(query):

    query=query.lower()

    query=re.sub(r'[^a-zA-Z0-9 ]','',query)

    return query.strip()