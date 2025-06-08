import os
import fitz  # PyMuPDF
import gradio as gr
import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings

# Constants
file_path = "/Users/p0n00ct/PycharmProjects/llm_experiments/SampleFiles/2019-Toyota-Warranty-Handbook-1.pdf"
file_name = "2019-Toyota-Warranty-Handbook-1.pdf"
ollama_model = "llama3.1"


class VectorStore:
    def __init__(self):
        self.pdf_name = file_name
        self.pdf_path = file_path
        self.embeddings = OllamaEmbeddings(model=ollama_model)
        self.vector_store = None
        self.load_data()

    def load_data(self):
        loader = PyPDFLoader(file_path=self.pdf_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(
            chunk_size=900000,
            chunk_overlap=300,
            separator="\n"
        )
        docs = text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def retrieve_doc(self, query, top_k):
        """Retrieves top_k relevant document chunks using similarity search."""
        results = self.vector_store.similarity_search(query=query, k=top_k)
        chunks = [doc.page_content for doc in results]
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        return chunks

vectorstore = VectorStore()

def extract_text_from_pdf(pdf_path):
    """Extracts and returns all text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_text_chat(sys_prompt, question, top_k, pdf_option):
    """
    Sends either full PDF content or relevant chunks to Ollama's LLaMA model
    and returns the response.
    """
    try:
        if pdf_option:
            print("Using Entire PDF")
            pdf_text = extract_text_from_pdf(file_path)
            messages = [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': pdf_text},
                {'role': 'user', 'content': question}
            ]
        else:
            try:
                print ("Chunking the PDF")
                chunks = vectorstore.retrieve_doc(question, top_k)
            except Exception as e:
                return f"Error retrieving document chunks: {e}"

            context = "-------------".join(chunks)
            messages = [
                {
                    'role': 'system',
                    'content': f"{sys_prompt}\n\n####### Context ######\n{context}"
                },
                {'role': 'user', 'content': question}
            ]

        response = ollama.chat(model=ollama_model, messages=messages)
        print(f"\n\nResponse:\n\n{response['message']['content']}")
        return response['message']['content']

    except Exception as ex:
        return f"Error generating response: {ex}"


def handle_pdf_option(option):
    """Returns True if 'Entire PDF' is selected, else False."""
    return option == "Entire PDF"


# UI setup
with gr.Blocks() as demo:
    pdf_option_state = gr.State(value=True)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            text_system = gr.Textbox(label="System Prompt")
            text_input = gr.Textbox(label="Question")
            top_k = gr.Number(label="Top K", value=1)

            radio = gr.Radio(
                ["Entire PDF", "Chunk PDF"],
                label="PDF Processing Mode",
                interactive=True
            )
            radio.change(fn=handle_pdf_option, inputs=radio, outputs=pdf_option_state)

            submit_button = gr.Button("Submit")

        with gr.Column(scale=2, min_width=300):
            text_output = gr.Textbox(label="Output")

    submit_button.click(
        fn=extract_text_chat,
        inputs=[text_system, text_input, top_k, pdf_option_state],
        outputs=text_output
    )

demo.launch(debug=True)