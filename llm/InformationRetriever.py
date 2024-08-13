
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def build_llm(model_llama_ccp_path):

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=model_llama_ccp_path,
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
        n_ctx=2048
    )

    return llm


def get_answer(db, question, model_llama_ccp_path):

    llm = build_llm(model_llama_ccp_path)

    # Baseline answer from LLM without retrieval
    llm(question)

    # RAG prompt
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

    result = qa({"query": question})
    print(result["result"])

