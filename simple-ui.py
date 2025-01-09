import gradio as gr
import ollama


def extract(question, file):

    question = question
    document = file
    if document is None:
        response = ollama.chat(
            model='llama3',
            messages=[{
                'role': 'user',
                'content': question
            }]
        )
    else:
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': question,
                'images': [document]
            }]
        )

    return response['message']['content']

def upload_file(file, question):
    response = extract(question, file)
    print(f"==============\nquestion: {question}\nresponse: {response}")
    return response

def upload_video(file):
    response = extract(file)
    print(f"==============\nquestion: {question}\nresponse: {response}")
    return response

# with gr.Blocks() as demo:
#     interface = gr.Interface(fn=extract,
#                  inputs=gr.Textbox(lines=3, label='INPUT', placeholder="Type here..."),
#                  outputs="text")
#     file_output = gr.File()
#     upload_button = gr.UploadButton("Click to Upload a File", file_types=["image", "video"], file_count="multiple")
#     interface.upload(upload_file, upload_button, file_output)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            text_input = gr.Textbox(label="Question")
            image_input = gr.Image(label="Image", type="filepath")
            # video_input = gr.Video(label="Video")
            image_button = gr.Button("Submit")
            # video_button = gr.Button("Process Video")
        with gr.Column(scale=2, min_width=300):
            # image_output = gr.Image(width=300, height=300)
            text_output = gr.Textbox(label="Output")
    image_button.click(upload_file, inputs=[image_input, text_input], outputs=text_output)
    # video_button.click(upload_file, inputs=image_input, outputs=text_output)


# demo = gr.Interface(
#     fn=upload_file,
#     inputs=[
#         "file",
#     ],
#     outputs="text"
# )
demo.launch(debug=True)