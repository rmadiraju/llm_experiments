import os
import fitz  # PyMuPDF
import gradio as gr
import ollama
import boto3
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings

# --- Constants ---
file_path = "/Users/p0n00ct/PycharmProjects/llm_experiments/SampleFiles/2019-Toyota-Warranty-Handbook-1.pdf"
file_name = "2019-Toyota-Warranty-Handbook-1.pdf"
ollama_model = "llama3.1"
claude_model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
bedrock_region = "us-east-1"

# --- Vector Store (initialized once) ---
class VectorStore:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=ollama_model)
        self.vector_store = None
        self.load_data()

    def load_data(self):
        loader = PyPDFLoader(file_path=file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=900000, chunk_overlap=300)
        docs = text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def retrieve_doc(self, query, top_k):
        results = self.vector_store.similarity_search(query=query, k=top_k)
        return [doc.page_content for doc in results]

#vectorstore_instance = VectorStore()

# --- Helpers ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text

def query_claude_bedrock(system_prompt, user_question):
    """Call Claude 3.5 via Amazon Bedrock using the Messages API."""
    bedrock = boto3.client("bedrock-runtime", region_name=bedrock_region)

    # Combine system prompt and user question into one user message
    full_prompt = f"{system_prompt}\n\n{user_question}"

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9
    }

    response = bedrock.invoke_model(
        modelId=claude_model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    return result["content"]

def extract_text_chat(system_prompt, question, top_k, pdf_option, backend_choice):
    try:
        if pdf_option:
            content = extract_text_from_pdf(file_path)
        else:
            chunks = vectorstore_instance.retrieve_doc(question, top_k)
            content = "\n-------------\n".join(chunks)

        if backend_choice == "Claude 3.5":
            prompt = f"{system_prompt}\n\nContext:\n{content}\n\nQuestion:\n{question}"
            return query_claude_bedrock(system_prompt, prompt)
        else:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': content},
                {'role': 'user', 'content': question}
            ]
            response = ollama.chat(model=ollama_model, messages=messages)
            return response['message']['content']

    except Exception as e:
        return f"❌ Error: {e}"

def handle_pdf_option(option):
    return option == "Entire PDF"

# --- UI ---
with gr.Blocks() as demo:
    pdf_option_state = gr.State(value=True)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            text_system = gr.Textbox(label="System Prompt", placeholder="You are a helpful assistant.")
            text_input = gr.Textbox(label="Question", placeholder="Ask a question about the PDF...")
            top_k = gr.Number(label="Top K Chunks", value=2, precision=0)

            radio_pdf = gr.Radio(
                ["Entire PDF", "Chunk PDF"],
                label="PDF Processing Mode",
                interactive=True
            )
            radio_pdf.change(fn=handle_pdf_option, inputs=radio_pdf, outputs=pdf_option_state)

            backend_choice = gr.Radio(
                ["llama3.1", "Claude 3.5"],
                label="Choose LLM Backend",
                interactive=True
            )

            submit_button = gr.Button("Submit")

        with gr.Column(scale=2, min_width=300):
            text_output = gr.Textbox(label="Response", lines=15)

    submit_button.click(
        fn=extract_text_chat,
        inputs=[text_system, text_input, top_k, pdf_option_state, backend_choice],
        outputs=text_output
    )

demo.launch(debug=True)