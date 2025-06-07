import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import google.generativeai as genai
from dotenv import load_dotenv
import io
import contextlib
import re

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def sanitize_code(code):
    # Remove markdown code fences if present
    code = re.sub(r"```(?:python)?", "", code)
    code = re.sub(r"```", "", code)
    return code.strip()

def call_gemini_api(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        f"""You are a helpful assistant that returns only Python matplotlib/seaborn code.
The DataFrame is named df. Generate code that directly produces a plot using matplotlib or seaborn.
Do not include explanations or comments.
Do not include any imports or print statements.
Do not include plt.show().
Only return valid Python code.
Prompt: {prompt}
"""
    )
    return sanitize_code(response.text)

def main():
    st.title("🤖📊 Gemini Powered- Data Visualizer")
    st.write("Upload a CSV file and describe your plot in plain English!")

    uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("🔍 Data Preview")
            st.dataframe(df.head())

            prompt = st.text_input("🧠 Describe your plot (e.g., 'scatter plot of age vs income by gender')")

            if st.button("Generate Plot") and prompt.strip():
                code_output = call_gemini_api(prompt)

                try:
                    # Only run code if it contains key plotting functions
                    if any(keyword in code_output for keyword in ["sns.", "plt.", "df"]):
                        with contextlib.redirect_stdout(io.StringIO()):
                            exec(code_output, {"df": df, "plt": plt, "sns": sns})
                        st.pyplot(plt.gcf())
                        plt.clf()
                    else:
                        st.warning("⚠ Gemini did not return usable plotting code.")
                except SyntaxError as e:
                    st.error(f"⚠ Syntax error in generated code: {e}")
                    st.text("Generated code was:\n" + code_output)
                except Exception as e:
                    st.warning(f"⚠ Error executing Gemini code: {e}")

            else:
                st.info("Please enter a plot description and click 'Generate Plot'.")

        except Exception as e:
            st.error(f"❌ Could not read CSV file: {e}")

if __name__ == "__main__":
    main()
