import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
import gradio as gr
import ollama

file_path = "/Users/rmadiraju/workspace/python-projects/vehicle-manual/2019-Toyota-Warranty-Handbook-1.pdf"
file_name = "2019-Toyota-Warranty-Handbook-1.pdf"

class VectorStore:
    def __init__(self):
        self.pdf_name = file_name
        self.pdf_path = file_path
        ollama_model = "llama3"
        self.embeddings = OllamaEmbeddings(model=ollama_model)
        self.retrieval_chain = None
        self.vector_store = None
        self.load_data()

    def load_data(self):
        loader = PyPDFLoader(file_path=file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=900000, chunk_overlap=300, separator="\n"
        )
        docs = text_splitter.split_documents(documents=documents)

        self.vector_store = FAISS.from_documents(docs, self.embeddings)



    def retrieve_doc(self, query, top_k):
        """Loading existing index for the RAG model."""
        results = self.vector_store.similarity_search(query=query, k=top_k)
        chunks = []
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
            chunks.append(doc.page_content)
        return chunks


vectorStore = VectorStore()
# system_prompt = "Answer the question from user with the below context"
system_prompt = "You are expert in deciding if the claims coming from user is part of manufacturer warranty or not. Answer yes or no and give explanation with less than 100 words"

def extract(question, sys_prompt, top_k):
    try:
        chunks = vectorStore.retrieve_doc(question, top_k)
    except Exception as e:
        return f"Error occurred retriving similar documents from vector store {e}"
    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[
                {
                    'role': 'system',
                    'content': f"{sys_prompt}"
                               "\n\n"
                               "####### Context ######"
                               f"{'-------------'.join(chunks)}"
                },
                {
                'role': 'user',
                'content': question
                }
            ]
        )
        print(f"\n\n Response \n\n\n{response['message']['content']}")
        return response['message']['content']
    except Exception as ex:
        return f"Error occurred generating answer: {e}"



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            text_system = gr.Textbox(label="System Prompt")
            text_input = gr.Textbox(label="Question")
            top_k = gr.Number(label="Top K", value=1)
            image_button = gr.Button("Submit")
        with gr.Column(scale=2, min_width=300):
            text_output = gr.Textbox(label="Output")
    image_button.click(extract, inputs=[text_input, text_system, top_k], outputs=text_output)

demo.launch(debug=True)
