"""

# llama-spp-python library may require different ARGS depending on where you are installing it
# Check https://python.langchain.com/docs/integrations/llms/llamacpp for installation options
# Below is for Apple silicon
#!CMAKE_ARGS="-DLLAMA_METAL=on" pip install -q llama-cpp-python
!pip install -q llama-cpp-python

# Big download of the model ~4Gb
# Can choose any text generation model found here: https://huggingface.co/models?pipeline_tag=text-generation&sort=trending
# Recommended to choose models from TheBloke who converts to GGUF format to run in limited memory
!huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir models/ --local-dir-use-symlinks False

# Install:
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
https://visualstudio.microsoft.com/es/vs/preview/
"""

from custom_dataset_processors.pubmed import PubmedDatasetProcessor
from vector_database import VectorDatabaseBuilder

preprocess_dataset = False
base_dir = "C:/Alex/Dev/data_corpus/InformationRetrieval"
dataset_name = "pubmed"
dataset_dir_path = base_dir + '/' + dataset_name
processed_dataset_dir_path = dataset_dir_path + "/processed"

if preprocess_dataset:
    PubmedDatasetProcessor.generate_separated_files_of_xml_in_dir(dataset_dir_path, processed_dataset_dir_path)

db = VectorDatabaseBuilder.generate_vector_database_with_files(dataset_name, processed_dataset_dir_path, chunk_size=1000, chunk_overlap=250)
question = "do you know something about a phosphodiesterase that was purified from cultured tobacco?"
searchDocs = db.similarity_search(question)
print(searchDocs[0].page_content)


