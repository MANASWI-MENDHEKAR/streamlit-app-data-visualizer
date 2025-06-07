# 📊 LLM-Based Data Visualizer App (Streamlit Wrapper)

An interactive data visualization app powered by multiple large language models (LLMs), allowing users to upload CSV files and generate insightful plots using natural language or manual fallback.

## 🔧 Features
- Upload your own CSV data
- Ask the model to generate charts using natural language
- Manual plotting mode when LLM API fails
- Unified wrapper structure for modular LLM switching

## 🤖 Versions Deployed

| LLM Version | Live Demo Link | Model Used |
|-------------|----------------|------------|
| 💬 ChatGPT | [Launch ChatGPT App](https://app-app-data-visualizer-rur7dwx7hnjr4ljnspnuou.streamlit.app/)
| 🔗 Together.ai | [Launch Together App](https://app-app-data-visualizer-jjsc7fywce9jvhgnoy78sg.streamlit.app/) 
| ✨ Gemini | [Launch Gemini App](https://app-app-data-visualizer-ntchgqxdkb7wtqfex3vrqb.streamlit.app/)

> Replace the links above with your actual deployed links.

## 🛠️ Tech Stack

- Streamlit
- Python
- OpenAI / Google Gemini / Together.ai
- Pandas, Matplotlib, Seaborn
- dotenv for API key management

## 📁 Structure

```bash
├── chatgpt_visualizer.py
├── gemini_visualizer.py
├── together_visualizer.py
├── requirements.txt
└── README.md
