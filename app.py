import streamlit as st
from main import model_with_tools  # Import the instance already bound with tools

st.title("ContentCraft Bot")
st.markdown("""
This bot can write **Articles**, **Tweets**, and **Blogs** for you based on your input query.  
Just type your request and click **Invoke** to see it in action.
""")
user_input = st.text_input("Enter your query:", "")

if st.button("Invoke"):
    if not user_input.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing..."):
            response = model_with_tools.invoke(user_input)
        if "error" in response:
            st.error(f"Error: {response['error']}")
            if "raw_output" in response:
                st.code(response["raw_output"])
        else:
            st.success("Tool call executed successfully!")
            st.markdown(f"**Tool Used:** `{response['tool_call']['name']}`")
            st.markdown(f"**Arguments:** `{response['tool_call']['args']}`")
            st.markdown(f"**Result:**\n\n{response['result']}")
