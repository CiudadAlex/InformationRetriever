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
