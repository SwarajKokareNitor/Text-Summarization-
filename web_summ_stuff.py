from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import os
 
load_dotenv()
genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))
llm = os.getenv('GOOGLE_API_KEY') 
#Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-pro")

#Load the blog
loader = WebBaseLoader("https://en.wikipedia.org/wiki/A._P._J._Abdul_Kalam")
docs = loader.load()

#Define the Summarize Chain
template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""

prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

#Invoke Chain
response=stuff_chain.invoke(docs)
print(type(response))
print(response["output_text"])