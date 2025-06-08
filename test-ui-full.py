import os, fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
import gradio as gr
import ollama

file_path = "/Users/p0n00ct/PycharmProjects/llm_experiments/SampleFiles/2019-Toyota-Warranty-Handbook-1.pdf"
file_name = "2019-Toyota-Warranty-Handbook-1.pdf"


def extract_text_from_pdf(pdf_path):
    """Extracts and returns text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    #print(text)
    return text

def extract_text_chat(sys_prompt, question):
    """Loads PDF and chats with llama3 using ollama.chat()."""

    # If the PDF is too long, truncate or chunk here
    # truncated_text = pdf_text[:3000]  # Limit input size for single message
    try:
        #chunks = vectorStore.retrieve_doc(question, top_k)
        pdf_text = extract_text_from_pdf(file_path)
    except Exception as e:
        return f"Error occurred extracting PDF file {e}"
    try:
        print (question)
        response = ollama.chat(
            model='llama3.1',
            messages=[
                {
                    'role': 'system',
                    'content': f"{sys_prompt}"
                },
               {"role": "user", "content": f"{pdf_text}"},
                {
                    'role': 'user',
                    'content': question
                }
            ]
        )
        print(f"\n\n Response \n\n\n{response['message']['content']}")
        return response['message']['content']
    except Exception as ex:
        return f"Error occurred generating answer: {ex}"


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            text_system = gr.Textbox(label="System Prompt")
            text_input = gr.Textbox(label="Question")
            top_k = gr.Number(label="Top K", value=1)
            image_button = gr.Button("Submit")
        with gr.Column(scale=2, min_width=300):
            text_output = gr.Textbox(label="Output")
    image_button.click(extract_text_chat, inputs=[text_system, text_input], outputs=text_output)

demo.launch(debug=True)
