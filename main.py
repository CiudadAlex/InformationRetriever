"""
# Big download of the model ~4Gb

pip install -U "huggingface_hub[cli]"
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir models/ --local-dir-use-symlinks False

or download in:
https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_K_M.gguf

# Install Visual Studio for libraries:

https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
https://visualstudio.microsoft.com/es/vs/preview/
"""

from custom_dataset_processors.pubmed import PubmedDatasetProcessor
from vector_database import VectorDatabaseBuilder
from llm.InformationRetriever import InformationRetriever

##############################################################################################################
# Configuration ##############################################################################################
##############################################################################################################

preprocess_dataset = False
base_dir = "C:/Alex/Dev/data_corpus/InformationRetrieval"
dataset_name = "pubmed"
dataset_dir_path = base_dir + '/' + dataset_name
processed_dataset_dir_path = dataset_dir_path + "/processed"
model_llama_ccp_path = "C:/Alex/Dev/models/llm/llama-2-7b.Q4_K_M.gguf"
number_db_results = 2

question = "what is the molecular weight phosphodiesterase?"

##############################################################################################################
##############################################################################################################
##############################################################################################################

if preprocess_dataset:
    PubmedDatasetProcessor.generate_separated_files_of_xml_in_dir(dataset_dir_path, processed_dataset_dir_path)

db = VectorDatabaseBuilder.generate_vector_database_with_files(dataset_name, processed_dataset_dir_path, chunk_size=1000, chunk_overlap=250)

searchDocs = db.similarity_search(question)
print("_______________________________________________")
print("Documents retrieved: " + str(len(searchDocs)))
print(searchDocs[0].page_content)
print("_______________________________________________")

informationRetriever = InformationRetriever(model_llama_ccp_path)

answer_raw = informationRetriever.get_answer(question)
answer_context = informationRetriever.get_answer_with_context(db, question, number_db_results)

print("### QUESTION ##################################################################################################")
print(question)
print("###############################################################################################################")
print("### DOCUMENTS #################################################################################################")
for i in range(number_db_results):
    print(searchDocs[i].page_content)
    print("_______________________________________________")
print("###############################################################################################################")
print("### ANSWER RAW ################################################################################################")
print(answer_raw)
print("###############################################################################################################")
print("###ANSWER WITH CONTEXT ########################################################################################")
print(answer_context)
print("###############################################################################################################")

