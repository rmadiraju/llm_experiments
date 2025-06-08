import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
import gradio as gr
import ollama
import os, fitz  # PyMuPDF

file_path = "/Users/rmadiraju/workspace/python-projects/vehicle-manual/2019-Toyota-Warranty-Handbook-1.pdf"
file_name = "2019-Toyota-Warranty-Handbook-1.pdf"

system_prompt = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    <task> Task: Your task is to determine if the claim that is submitted by customer is covered under warranty </task>
    
    <context> Warranty Manual: {warranty_doc} </context>
    
    <rules>
    Rules: Here are some rules that you need follow.
    1. Check if the part mentioned is covered under warranty.
    2. If the part is covered, check if the details such as miles and years mentioned in claim satisfies conditions
        2a. If the details does not satisfy conditions, respond as not covered
    3. If the part is not covered, respond as not covered
    4. Check if there are any state specific policy rules, if yes, override Federal warranties
    5. Respond only with Covered or Not-Covered and give explanation on why it is covered or not.
    </rules>
    
    <output>
        Respond with short answer with less than 100 words, if possible point out page numbers in the policy manual
    </output>
    
    
    <|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>
        {claim}
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
"""



def extract_text_from_pdf(pdf_path):
    """Extracts and returns text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += "\n - Page - "
        text += page.get_text()
    doc.close()
    #print(text)
    return text

class VectorStore:
    def __init__(self):
        self.pdf_name = file_name
        self.pdf_path = file_path
        ollama_model = "llama3"
        self.embeddings = OllamaEmbeddings(model=ollama_model)
        self.retrieval_chain = None
        self.vector_store = None
        # self.load_data()

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

def extract(question, top_k):
    try:
        # chunks = vectorStore.retrieve_doc(question, top_k)
        pdf_text = extract_text_from_pdf(file_path)
    except Exception as e:
        return f"Error occurred retriving similar documents from vector store {e}"
    try:
        response = ollama.generate(
            model='llama3.1',
            prompt=system_prompt.format(warranty_doc=pdf_text, claim=question),
        )
        print(f"\n\n Response \n\n\n{response['response']}")
        return response['response']
    except Exception as ex:
        return f"Error occurred generating answer: {ex}"



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            # text_system = gr.Textbox(label="System Prompt")
            text_input = gr.Textbox(label="Claim")
            top_k = gr.Number(label="Top K", value=1)
            image_button = gr.Button("Submit")
        with gr.Column(scale=2, min_width=300):
            text_output = gr.Textbox(label="Output")
    image_button.click(extract, inputs=[text_input, top_k], outputs=text_output)

demo.launch(debug=True)
