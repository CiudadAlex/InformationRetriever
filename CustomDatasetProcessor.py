# Get all content from *.rst files under docs airflow/docs/apache-airflow
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_experimental.text_splitter import SemanticChunker  #TODO try use instead of above
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os


def extract_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    # Can clean up data here
    return file_content


def generate_vector_database_with_files(dir_path, chunk_size=1000, chunk_overlap=150):

    text_data = []
    metadata = []

    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        doc_text = extract_text_from_file(file_path)
        text_data.append(doc_text)
        metadata.append({"file-name": file})

    # Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
    # It splits text into chunks of "chunk_size" characters each with a "chunk_overlap"-character overlap.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 'data' holds the text you want to split, split the text into documents using the text splitter.
    text_data_to_docs = text_splitter.create_documents(text_data, metadata)
    docs = text_splitter.split_documents(text_data_to_docs)

    print("--------------------")
    print(docs[0])
    print("--------------------")

    # Define the path to the pre-trained model you want to use
    model_path = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

    db = FAISS.from_documents(docs, embeddings) #may take a 4-5 minutes
    return db

