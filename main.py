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
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def predict_picture():
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

    print(pages)
    print(text)

    embeddings = OllamaEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(pages)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

  

    chat_history = [HumanMessage(content="How old Ricardo is"), AIMessage(content="Yes!")]
    retriever_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me why"
    })
   

    prompt = ChatPromptTemplate.from_messages([("system", "Answer the user's questions based on the below context:\n\n{context}"),MessagesPlaceholder(variable_name="chat_history"),("user", "{input}"),])
    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    response=retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": text
    })
    # # Delete the temporary file after processing
    # os.remove(file_path)
    print(response["answer"])
    return jsonify({'response': response["answer"]})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)