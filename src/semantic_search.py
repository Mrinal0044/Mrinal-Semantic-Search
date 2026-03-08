class SemanticSearch:

    def __init__(self,documents,vector_index):

        self.documents=documents
        self.vector_index=vector_index

    def search(self,query_embedding):

        indices=self.vector_index.search(query_embedding)

        return self.documents[indices[0]]