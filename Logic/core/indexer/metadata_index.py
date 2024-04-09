from index_reader import Index_reader
from indexes_enum import Indexes, Index_types
import json

class Metadata_index:
    def __init__(self, path='index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """

        #TODO
        self.path = path
        self.documents = self.read_documents()
        self.metadata_index = self.create_metadata_index()

    def read_documents(self):
        """
        Reads the documents.
        
        """

        #TODO
        document_index = Index_reader('index/', index_name=Indexes.DOCUMENTS).index
        documents = []
        for key, value in document_index.items():
            documents.append(value)
        return documents


    def create_metadata_index(self):    
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['average_document_length'] = {
            'stars': self.get_average_document_field_length('stars'),
            'genres': self.get_average_document_field_length('genres'),
            'summaries': self.get_average_document_field_length('summaries')
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index
    
    def get_average_document_field_length(self,where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """

        #TODO
        path = self.path + where + '_' + Index_types.DOCUMENT_LENGTH.value + '_index.json'
        with open(path, 'r') as f:
            json_data = f.read()
        document_lengths = json.loads(json_data)
        avg_length = 0
        for key, value in document_lengths.items():
            avg_length += value
        avg_length /= len(document_lengths)
        return avg_length


    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path =  path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)
        print('metadata index stored successfully!')


    
if __name__ == "__main__":
    meta_index = Metadata_index()
    meta_index.store_metadata_index(meta_index.path)