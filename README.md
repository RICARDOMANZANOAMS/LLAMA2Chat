The following repo uses Llama2 with 7B of paramaters to answers questions related to a pre-defined document.

It uses flask as a wrapper of the program to make it a web app.


Installation

To run the program is necessary to install Ollama from: https://ollama.com/

Then, it is necessary to run 

ollama run llama2

Then, it is necessary to create an env in conda

conda create --name llama-ric python
pip install requirements.txt

After running the main.py, it is necessary to open a web browser and paste the url

You need to upload a pdf file and put a question.

The answer will pop up 