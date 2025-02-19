from openai import OpenAI

# init the client but point it to TGI
client = OpenAI(
    base_url="https://ai.ls3.rnet.torontomu.ca/llm/v1/",
    api_key="-"
)

input = '''
list top 5 topics discussed in this text using {'topic':[]} json format:
So today we will be asking you to reflect on your experiences
representing claimants at the Rpd. In answering, please reflect on anything that you believe 
may give you insight into the members, thinking, including, for example, their on the record 
comments and demeanor during the hearing their off, the record comments and demeanor before 
the start of the hearing or during the breaks, and the reasons that they give for their decisions.
''' 

chat_completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that summerizes interview transctipt and answer questions regarding it."},
        {"role": "user", "content": input}
    ]
)

# iterate and print stream
print(chat_completion.choices[0].message.content)
