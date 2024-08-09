import json
from typing import List

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator,
)
# from langchain_experimental.tabular_synthetic_data.prompts import (
#     SYNTHETIC_FEW_SHOT_PREFIX,
#     SYNTHETIC_FEW_SHOT_SUFFIX,
# )
from langchain_openai import ChatOpenAI
from pydantic.json import pydantic_encoder

import os
os.environ["OPENAI_API_KEY"] = "OPEN_KEY"

class ConversationItem(BaseModel):
    customer: str
    agent: str


examples = [
    {
        "example": """
                        conversation_1 : [
                             { 
                                customer: 2003 Buick Lesabre Custom. Is there availability to view this Monday?,
                                agent: I will be happy to help you with requesting a test drive for Monday. Do you mind providing your name and phone number so we can help schedule a test drive for you?
                             },
                             { 
                                customer: Yes, Mark Cuban 123-456-7890,
                                agent: Got it. I just sent over your test drive request and our sales specialist should be in touch soon to schedule your appointment. In the meanwhile, do you want to trade-in a vehicle, estimate your payment or get pre-qualified? 
                             },
                             { 
                                customer: Yes, I do have a car for trade-in. I am looking fo a quote to sell my 2022 telluride currently 30K miles, prestige pack and no accidents or damages,
                                agent: You can get your valuation by submitted some information about your vehicle. Also, note that dealer may still need to see your vehicle for final value. Click on the button below to get started. [Button] 
                             }
                         ]
        
                   """
    }
]

OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

SYNTHETIC_FEW_SHOT_PREFIX = (
    "You are an expert in generating synthetic data about a chatbot application on car dealership website. "
    "The chatbot should answer questions about generic car details, availability and scheduling test drives."
    "In this test data customer can ask about trade in or test drive in different ways and can jump from one to other"
    "The customer might not always give the details but can jump to a different topic"
    "Be as random as possible in generating conversations"
    "Use the simple example below to get an idea of conversation. The conversation should be multi turn chat conversation"
    "Generate at least 10 conversations"
    "Examples below:"
    "{examples}"
)
SYNTHETIC_FEW_SHOT_SUFFIX = (
    """Now you generate synthetic data about dealership chatbot. Make sure to {extra}:"""
)

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE,
)
model = ChatOpenAI(
        temperature=1,
        max_tokens=2048
    )

messages = [
    (
        "system",
        SYNTHETIC_FEW_SHOT_PREFIX,
    ),
    ("human", "Generate Synthetic Data using example above"),
]
# synthetic_data_generator = create_openai_data_generator(
#     output_schema=ConversationItem,
#     llm=model,  # You'll need to replace with your actual Language Model instance
#     prompt=prompt_template,
# )


prompt = ChatPromptTemplate.from_messages(
    messages
)

print("generating synthetic data started....")

chain = prompt | model
ai_message = chain.invoke(
    {
        "examples": examples
    }
)

print("\n\ngenerating synthetic data completed....")

print(f"\n\n\n{ai_message} \n\n\n")
print(f"\n\n{json.dumps(json.loads(ai_message.content))}")
