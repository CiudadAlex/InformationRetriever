
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="models/llama-2-7b.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

# Hugging Face token
from google.colab import userdata
hf_token = userdata.get('huggingface')

# Load the tokenizer associated with the specified model
# You may notice that we are using "meta-llama/Llama-2-7b-hf" instead of "TheBloke/Llama-2-7B-GGUF"
# This is because the tokenizer model is not included in the GGUF converted LLM, but it is sufficient to use the tokenizer
# from the original model. As far as I know, as long as your LLM and tokenizer speak the same language, it is fine.
# In this case, the LLM and tokenizer are of model_type "llama".
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding=True, truncation=True, max_length=512, token=hf_token)


# Baseline answer from LLM without retrieval
llm(question)

# RAG prompt
from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
Context: {context}
Question: {question}
Helpful Answer:"""
qa_chain_prompt = PromptTemplate.from_template(template)

# Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
retriever = db.as_retriever(search_kwargs={"k": 2})

# Create a question-answering instance (qa) using the RetrievalQA class.
# It's configured with a language model (llm), a chain type "refine", the retriever we created, and an option to not return source documents.
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    verbose=True,
    chain_type_kwargs={"prompt": qa_chain_prompt, "verbose": True},
)

# Can use the same question or use your own
question = "Does Airflow have audit logs?"
result = qa({"query": question})
print(result["result"])

