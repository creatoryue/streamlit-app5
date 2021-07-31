import streamlit.components.v1 as components

# Declare the component:
my_component = components.declare_component("my_component", path="frontend/build")

# Use it:
my_component(greeting="Hello", name="World")
