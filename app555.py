import streamlit as st
import streamlit.components.v1 as components
import base64

st.header("test html import")
HtmlFile = open("index.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
