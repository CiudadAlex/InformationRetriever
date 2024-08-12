"""
# Install libraries
!pip install -q datasets sentence-transformers faiss-cpu transformers langchain langchain_experimental huggingface-hub
# Git clone apache-airflow
# Later, can clone list of repos
!git clone --depth 1 -b main https://github.com/apache/airflow.git

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

import PubmedSetup
import CustomDatasetProcessor

base_dir = "C:/Alex/Dev/data_corpus/InformationRetrieval"
file_path = base_dir
output_dir_path = base_dir + "/processed"
# PubmedSetup.generate_separated_files_of_xml_in_dir(file_path, output_dir_path)

db = CustomDatasetProcessor.generate_vector_database_with_files(output_dir_path, chunk_size=1000, chunk_overlap=250)
question = "Does Airflow have audit logs?"
searchDocs = db.similarity_search(question)
print(searchDocs[0].page_content)
