import streamlit as st
import streamlit.components.v1 as components
import base64

from flask import Flask
from flask import render_template

st.header("test html import")
###--------------------------1-------------------------###
# 1st way using components

# https://discuss.streamlit.io/t/change-the-display-position-of-an-html-file-on-a-webpage-in-streamlit/7779

#HtmlFile = open("index.html", 'r', encoding='utf-8')
#source_code = HtmlFile.read() 
#print(source_code)
#components.html(source_code, height= 1600, width=1600)

###--------------------------2-------------------------###
# 2nd way using Flask

# https://hackmd.io/@shaoeChen/HJkOuSagf?type=view
# https://stackoverflow.com/questions/60032983/record-voice-with-recorder-js-and-upload-it-to-python-flask-server-but-wav-file

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('templates\\index.html')
  

if __name__ == '__main__':

