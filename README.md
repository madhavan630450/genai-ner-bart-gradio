## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Named Entity Recognition (NER) is an important task in Natural Language Processing (NLP) that involves identifying and classifying entities such as person names, locations, organizations, dates, and other important information from unstructured text. Traditional rule-based approaches often fail to generalize across different contexts and languages.

The objective of this experiment is to develop a prototype NER system using a fine-tuned BART (Bidirectional and Auto-Regressive Transformers) model capable of accurately recognizing entities in text. The model will be integrated with the Gradio framework to create a simple web-based interface where users can input text and visualize the extracted entities in real time.

### DESIGN STEPS:


#### STEP 1:

Install the required libraries such as Transformers, PyTorch, and Gradio and import the necessary modules for loading the fine-tuned BART model.

#### STEP 2:

Load the fine-tuned BART model and tokenizer, preprocess the input text, and implement a function to perform Named Entity Recognition on the given input.

#### STEP 3:

Integrate the NER prediction function with the Gradio interface, allowing users to input text through a web interface and display the identified named entities interactively.

### PROGRAM:
```
NAME   : MARIMUTHU MATHAVAN
REF NO : 212224230153
```
```
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
```
```
# Helper function
import requests, json

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))
```
```
def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}
```
```
import gradio as gr
def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']
    
gr.close_all()
demo = gr.Interface(fn=summarize, inputs="text", outputs="text")
demo.launch()
```
```
gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    #Here we introduce a new tag, examples, easy to use examples for your application
                    examples=["My name is Andrew and I live in California", "My name is Poli and work at HuggingFace"])
demo.launch(share=True, server_port=int(os.environ['PORT3']))
```
```
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is SUSANTH, studying in SAVEETHA ENGINEERING COLLEGE and I live in ANDHRA PRADESH", "My name is RASOOL, I live in TAMIL NADU"])

demo.launch(share=True, server_port=int(os.environ['PORT4']))
```


### OUTPUT 1:

<img width="1172" height="622" alt="image" src="https://github.com/user-attachments/assets/09c655b6-c1e3-4a30-b109-a3135b7a79e7" />

### OUTPUT 2:

<img width="1173" height="623" alt="image" src="https://github.com/user-attachments/assets/4b52e20f-7b79-41cf-9196-8044f9130712" />


### RESULT:
The Named Entity Recognition (NER) prototype was successfully developed using a fine-tuned BART model. The system accurately identifies entities from input text and displays the results through an interactive Gradio web interface. Thus, the objective of building and deploying the NER application was achieved.
