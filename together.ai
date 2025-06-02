import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import re
from dotenv import load_dotenv
from together import Together

# Load API Key from .env
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    st.error("API key not found. Make sure TOGETHER_API_KEY is set in your .env file.")
    st.stop()

# Initialize Together client
client = Together(api_key=api_key)

# Function to extract code from markdown-style response
def extract_code_block(text):
    # Extract first Python code block (```python ... ```)
    matches = re.findall(r"```python(.*?)```", text, re.DOTALL)
    return matches[0].strip() if matches else text.strip()

# Get code from Together AI and extract only valid Python
def get_code_from_together(prompt):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a data visualization assistant using matplotlib and seaborn. Generate only valid Python code for plotting. Do not include explanation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )
    raw_content = response.choices[0].message.content
    return extract_code_block(raw_content)

# Streamlit App
def main():
    st.title("📊 Together AI Data Visualizer")

    f = st.file_uploader("Upload your CSV file", type=["csv"])
    if not f:
        st.info("Upload a CSV file to begin.")
        return

    try:
        df = pd.read_csv(f)
        st.dataframe(df.head())

        user_prompt = st.text_input("Describe your plot (e.g., 'bar chart of type counts'): ")

        if st.button("Generate Visualization"):
            if not user_prompt.strip():
                cols = df.columns.tolist()
                if len(cols) >= 2:
                    user_prompt = f"Create a seaborn scatter plot using 'df' with x='{cols[0]}' and y='{cols[1]}'."
                else:
                    user_prompt = f"Create a histogram of the column '{cols[0]}' in DataFrame 'df'."

            try:
                code = get_code_from_together(user_prompt)
                # Uncomment to view generated code
                # st.code(code, language="python")

                try:
                    ast.parse(code)  # Validate syntax
                    exec_globals = {"df": df, "sns": sns, "plt": plt, "pd": pd}
                    plt.figure()
                    exec(code, exec_globals)
                    st.pyplot(plt.gcf())
                    plt.clf()
                except SyntaxError as syn_err:
                    st.error(f"Syntax error in generated code: {syn_err}")
                except Exception as e:
                    st.error(f"Error running generated code: {e}")

            except Exception as e:
                st.error(f"Failed to generate visualization: {e}")

    except Exception as e:
        st.error(f"Error reading CSV: {e}")

if __name__ == "__main__":
    main()






