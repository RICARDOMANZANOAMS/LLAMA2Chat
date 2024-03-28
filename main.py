from flask import Flask, request, render_template,jsonify
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def predict_picture():
    #Upload llama model from Ollama
    llm = Ollama(model="llama2")
    # Check if the request contains a file
    if 'file' not in request.files:
        return 'No file part in the request', 400

    file = request.files['file']
    text = request.form['text']

    # Save the uploaded file temporarily
    upload_dir = 'uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    # Create PyPDFLoader instance with the file path
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    embeddings = OllamaEmbeddings()  #import embedding from ollama
    text_splitter = RecursiveCharacterTextSplitter()  
    documents = text_splitter.split_documents(pages)
    vector = FAISS.from_documents(documents, embeddings)  #db vector embedding
    
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": text})
    print(response["answer"])
    return jsonify({'response': response["answer"]})

    # # Delete the temporary file after processing
    # os.remove(file_path)

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)