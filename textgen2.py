import streamlit as st #for web dev
from transformers import GPT2LMHeadModel, GPT2Tokenizer

st.title("Text Generation Web App (100 words limit)")

# instantiate the model / download
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)

# create a prompt text for the text generation 
#prompt_text = "Python is awesome"
prompt_text = st.text_input(label = "Enter your prompt text...", value="What is the weather today?")

input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

with st.spinner("AI is at Work........"):
    output = model.generate(
    input_ids, 
    max_length=100, 
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
    )
st.success("AI Successfully generated the below text ")
st.balloons()

st.write(tokenizer.decode(output[0], skip_special_tokens=True))


