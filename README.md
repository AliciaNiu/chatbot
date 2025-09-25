# Chatbot  

A lightweight chatbot that combines **RAG (Retrieval-Augmented Generation)** with [Qdrant](https://qdrant.tech/) for vector search, and uses [OpenAI](https://platform.openai.com/) to generate answers. The frontend is a simple JavaScript interface that lets users chat with the bot.  

## ✨ Features  
- ⚡ **Qdrant-powered RAG** – store and retrieve relevant documents for better context  
- 🤖 **OpenAI integration** – generate accurate and conversational responses  
- 💬 **Minimal frontend** – basic JavaScript UI for chatting with the bot  

## 🔄 How it Works  
1. User sends a message via the JavaScript chat interface  
2. The backend retrieves context from Qdrant  
3. OpenAI generates an answer based on the retrieved context
4. The chatbot responds in real time  