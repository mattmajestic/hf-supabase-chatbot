import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import supabase

# Set your Supabase credentials as environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# Initialize Supabase
supabase_client = supabase.Client(SUPABASE_URL, SUPABASE_KEY)

def ai_chat():
    st.title("HuggingFace + Supabase Chatbot ⚡")
    st.write("")
    doc_expander = st.expander("Documentation 📚")
    with doc_expander:
        # Read the contents of the README.md file
        with open('README.md', 'r') as file:
            readme_text = file.read()
        st.markdown(readme_text)

    prompt = st.chat_input("Chat your Business with AI")
    with st.spinner("Generating Bot Response..."):
        default_prompt = "This is a chatbot for @majesticodding YouTube channel with partner @supabase"
        prompt_detail = st.text_area("Enter a Prompt for AI 📣", default_prompt)
        if prompt:
            model_name = "gpt2"  
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            def generate_response(prompt):
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                response_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
                bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
                return bot_response
            bot_response = generate_response(prompt)
            st.write(f"User has sent the following prompt: {prompt}")
            st.write("Bot:", bot_response)
            response = supabase_client.table("hf-supabase-chat").insert([{"prompt": prompt, "created_at": datetime.now().isoformat(),"prompt_detail":prompt_detail}]).execute()

if __name__ == '__main__':
    ai_chat()



if __name__ == '__main__':
    ai_chat()
