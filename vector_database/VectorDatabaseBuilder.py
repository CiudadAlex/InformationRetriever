from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pickle
import time


def extract_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    # Can clean up data here
    return file_content


def generate_vector_database_with_files(dataset_name, dir_path, chunk_size=1000, chunk_overlap=150):

    db = load_database_in_file(dataset_name, dir_path)
    if db:
        print("Loaded vector database from file")
        return db

    print("Generating vector database from dataset")

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

    print_some_examples(docs, 5)

    # Define the path to the pre-trained model you want to use
    model_path = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device': 'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,       # Provide the pre-trained model's path
        model_kwargs=model_kwargs,   # Pass the model configuration options
        encode_kwargs=encode_kwargs  # Pass the encoding options
    )

    db = build_vector_database(docs, embeddings)
    print("Created vector database from documents")

    store_database_in_file(dataset_name, dir_path, db)
    print("Stored vector database in file")

    return db


def print_some_examples(docs, num):

    for i in range(num):
        print("--------------------")
        print(docs[i])
        print("--------------------")


def store_database_in_file(dataset_name, dir_path, db):

    file_path = get_path_file_vector_database(dataset_name, dir_path)
    with open(file_path, 'wb') as f:  # open a text file
        pickle.dump(db, f)  # serialize the list
        f.close()


def load_database_in_file(dataset_name, dir_path):

    file_path = get_path_file_vector_database(dataset_name, dir_path)
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            db = pickle.load(f)  # deserialize using load()
            return db
    else:
        return None


def get_path_file_vector_database(dataset_name, dir_path):
    return dir_path + '/../vector_database/' + dataset_name + '.pkl'


def build_vector_database(docs, embeddings):

    db = None
    number_of_documents = len(docs)
    count = 0
    start_time = time.time()

    for d in docs:
        if db:
            db.add_documents([d])
        else:
            db = FAISS.from_documents([d], embeddings)

        count = count + 1
        print_progress(count, number_of_documents, start_time)

    return db


def print_progress(count, number_of_documents, start_time):

    if count % 1000 == 0:
        now = time.time()
        elapsed_seconds = now - start_time
        elapsed_peronage = count / number_of_documents
        remaining_peronage = 1 - elapsed_peronage
        remaining_seconds = remaining_peronage * elapsed_seconds / elapsed_peronage

        remaining_print_minutes = remaining_seconds / 60
        remaining_print_seconds = remaining_seconds % 60

        percentage_progress = round(1000 * elapsed_peronage) / 10
        print(str(count) + " of " + str(number_of_documents) + " " + str(percentage_progress)
              + "% | Estimated time: " + str(remaining_print_minutes) + " min " + remaining_print_seconds + " s")
