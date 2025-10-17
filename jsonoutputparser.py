from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    # template = 'Give me the name,age and city of a fictional person \n {format_instruction}'
    template='Give me 5 facts about {topic} \n {format_instruction}', 
    input_variables=['topic'], # input_variable =[]
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format()
#print(prompt)
# result = model.invoke(prompt)
# print(result)
# final_result=parser.parse(result.content)

chain = template | model | parser
# result = chain.invoke({})
result = chain.invoke({'topic':'black hole'}) # we want to pass a empty dictanary if no input variable 

print(result)


