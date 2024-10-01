# -*- coding: utf-8 -*-
"""Doc-Processing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qVGiApyRUoH7pceK-hln0fB87dSYMs1o
"""

!pip install vllm

!pip install --upgrade mistral_common

import huggingface_hub

# This will automatically retrieve your token from the secrets
huggingface_hub.login()

from vllm import LLM
from vllm.sampling_params import SamplingParams

#model_name = "mistralai/Pixtral-12B-2409"

#sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(
    model = "mistralai/Pixtral-12B-2409",
    tokenizer_mode= "mistral",
    max_model_len = 4000
)

def vqa(prompt: str, image_url: str):
  messages = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": prompt},
              {"type": "image_url", "image_url": {"url": image_url}}
          ]
      }
  ]
  outputs = llm.chat(
      messages,
      sampling_params=SamplingParams(max_tokens=8192)
  )
  return outputs

image_url = "https://github.com/rmadiraju/llm_experiments/blob/doc-processing/data/Bookout-1.png?raw=true"

question = "What is the vehicle vin?"
outputs = vqa(prompt=question, image_url=image_url)

print(f"{question} == {outputs[0].outputs[0].text}")


question = "What is the vehicle?"
outputs = vqa(prompt=question, image_url=image_url)

print(f"{question} == {outputs[0].outputs[0].text}")


question = "What is the odometer reading?"
outputs = vqa(prompt=question, image_url=image_url)

print(f"{question} == {outputs[0].outputs[0].text}")

question = "What is the value of the vehicle?"
outputs = vqa(prompt=question, image_url=image_url)

print(f"{question} == {outputs[0].outputs[0].text}")

image_url = "https://github.com/rmadiraju/llm_experiments/blob/doc-processing/data/Bookout-6.gif?raw=true"

question = "What is the vehicle vin?"
outputs = vqa(prompt=question, image_url=image_url)

print(f"{question} == {outputs[0].outputs[0].text}")


question = "What is the vehicle?"
outputs = vqa(prompt=question, image_url=image_url)

print(f"{question} == {outputs[0].outputs[0].text}")


question = "What is the odometer reading?"
outputs = vqa(prompt=question, image_url=image_url)

print(f"{question} == {outputs[0].outputs[0].text}")

question = "What is the value of the vehicle?"
outputs = vqa(prompt=question, image_url=image_url)

print(f"{question} == {outputs[0].outputs[0].text}")

image_url = "https://github.com/rmadiraju/llm_experiments/blob/doc-processing/data/Bookout-5.jpeg?raw=true"

question = "What is the vehicle vin?"
outputs = vqa(prompt=question, image_url=image_url)

print(f"{question} == {outputs[0].outputs[0].text}")


question = "What is the vehicle?"
outputs = vqa(prompt=question, image_url=image_url)

print(f"{question} == {outputs[0].outputs[0].text}")


question = "What is the odometer reading?"
outputs = vqa(prompt=question, image_url=image_url)

print(f"{question} == {outputs[0].outputs[0].text}")

question = "What is the value of the vehicle?"
outputs = vqa(prompt=question, image_url=image_url)

print(f"{question} == {outputs[0].outputs[0].text}")

image_url =  "https://raw.githubusercontent.com/rmadiraju/llm_experiments/doc-processing/Paystub-1.jpeg"

prompt = "What is the customers name?"
outputs = vqa(prompt=prompt, image_url=image_url)

outputs[0].outputs[0].text

prompt = "What is the net income from the document?"
outputs = vqa(prompt=prompt, image_url=image_url)

outputs[0].outputs[0].text

prompt = "What is the pay period from the document?"
outputs = vqa(prompt=prompt, image_url=image_url)

outputs[0].outputs[0].text

"""***The following is example for buyers order***"""

image_url =  "https://raw.githubusercontent.com/rmadiraju/llm_experiments/doc-processing/Sample-buyers-order.gif"

prompt = "What is the customers name?"
outputs = vqa(prompt=prompt, image_url=image_url)

outputs[0].outputs[0].text

prompt = "What is year, make, model of the trade-in vehicle?"
outputs = vqa(prompt=prompt, image_url=image_url)

outputs[0].outputs[0].text

prompt = "What is odometer reading of the trade-in vehicle?"
outputs = vqa(prompt=prompt, image_url=image_url)

outputs[0].outputs[0].text

prompt = "What is odometer reading of the order vehicle?"
outputs = vqa(prompt=prompt, image_url=image_url)

outputs[0].outputs[0].text

!nvidia-smi

!/usr/local/cuda/bin/nvcc --version