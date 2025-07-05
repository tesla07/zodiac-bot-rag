import streamlit as st
import requests
import json
from typing import Dict, Any
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Zodiac RAG Chatbot",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mystical theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
        color: #000000;
    }
    
    .bot-message {
        background-color: #e8f4fd;
        border-left-color: #764ba2;
        color: #000000;
    }
    
    .source-info {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: #666;
    }
    
    .suggestion-button {
        margin: 0.2rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid #667eea;
        background: white;
        color: #667eea;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .suggestion-button:hover {
        background: #667eea;
        color: white;
    }
    
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Dict[str, Any]:
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.RequestException:
        return {"status": "error", "message": "API not reachable"}


def upload_pdf(file) -> Dict[str, Any]:
    """Upload a PDF file to the API"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=30)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Upload failed: {str(e)}"}


def ask_question(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Ask a question to the RAG chatbot"""
    try:
        data = {"query": query, "top_k": top_k}
        response = requests.post(f"{API_BASE_URL}/ask", data=data, timeout=30)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"answer": f"Error: {str(e)}", "sources": [], "context_used": False}


def get_suggestions() -> list:
    """Get suggested questions"""
    try:
        response = requests.get(f"{API_BASE_URL}/suggestions", timeout=5)
        data = response.json()
        return data.get("suggestions", [])
    except requests.exceptions.RequestException:
        return []


def get_stats() -> Dict[str, Any]:
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        return response.json()
    except requests.exceptions.RequestException:
        return {}


def display_chat_message(message: str, is_user: bool = False, sources: list = None):
    """Display a chat message with proper styling"""
    message_class = "user-message" if is_user else "bot-message"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <strong>{'You' if is_user else '✨ Astrology Bot'}</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)
    
    # Display sources if available
    if sources and not is_user:
        for i, source in enumerate(sources):
            score = source.get('similarity_score', 0)
            filename = source.get('source_file', 'Unknown')
            preview = source.get('text_preview', '')
            
            st.markdown(f"""
            <div class="source-info">
                <strong>Source {i+1}:</strong> {filename} (Relevance: {score:.3f})<br>
                <em>{preview}</em>
            </div>
            """, unsafe_allow_html=True)


def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✨ Zodiac RAG Chatbot ✨</h1>
        <p>Channeling the mystical wisdom of Linda Goodman's Sun Signs and Love Signs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔮 Settings")
        
        # Health check
        health_status = check_api_health()
        if health_status.get("status") == "healthy":
            st.success("✅ API Connected")
        else:
            st.error("❌ API Not Available")
            st.error(health_status.get("message", "Unknown error"))
            return
        
        # Top-k slider
        top_k = st.slider("Number of sources to retrieve", 1, 10, 5)
        
        # File upload
        st.header("📚 Upload PDFs")
        uploaded_file = st.file_uploader(
            "Upload Linda Goodman's books (PDF)",
            type=['pdf'],
            help="Upload 'Sun Signs' and 'Love Signs' PDFs"
        )
        
        if uploaded_file is not None:
            if st.button("Upload and Index"):
                with st.spinner("Processing PDF..."):
                    result = upload_pdf(uploaded_file)
                    
                    if result.get("status") == "success":
                        st.success(f"✅ {result.get('message')}")
                        st.info(f"Created {result.get('chunks_created')} chunks")
                    else:
                        st.error(f"❌ {result.get('message')}")
        
        # Stats
        st.header("📊 Statistics")
        stats = get_stats()
        if stats:
            index_stats = stats.get("index_stats", {})
            if "total_vectors" in index_stats:
                st.metric("Indexed Chunks", index_stats["total_vectors"])
            
            uploaded_files = stats.get("uploaded_files", [])
            if uploaded_files:
                st.write("**Uploaded Files:**")
                for file in uploaded_files:
                    st.write(f"• {file}")
        
        # Reset button
        st.header("⚠️ Danger Zone")
        if st.button("Reset Index", type="secondary"):
            try:
                response = requests.delete(f"{API_BASE_URL}/reset", timeout=10)
                if response.status_code == 200:
                    st.success("Index reset successfully")
                    st.rerun()
                else:
                    st.error("Failed to reset index")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Main chat area
    st.header("💬 Ask About Astrology")
    

    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            message["content"], 
            message["is_user"], 
            message.get("sources")
        )
    
    # Suggestions
    suggestions = get_suggestions()
    if suggestions:
        st.write("**💡 Try asking:**")
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions[:6]):  # Show first 6 suggestions
            col = cols[i % 2]
            if col.button(suggestion, key=f"sugg_{i}"):
                st.session_state.user_input = suggestion
                st.rerun()
    
    # Chat input
    user_input = st.text_area(
        "Ask about zodiac signs, compatibility, or Linda Goodman's insights...",
        key="user_input",
        height=100,
        placeholder="e.g., Tell me about Aries personality traits"
    )
    
    # Send button
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("🔮 Ask", type="primary")
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Process user input
    if send_button and user_input.strip():
        # Add user message to chat
        st.session_state.messages.append({
            "content": user_input,
            "is_user": True
        })
        
        # Get bot response
        with st.spinner("🔮 Consulting the stars..."):
            response = ask_question(user_input, top_k)
        
        # Add bot response to chat
        st.session_state.messages.append({
            "content": response.get("answer", "Sorry, I couldn't process your question."),
            "is_user": False,
            "sources": response.get("sources", [])
        })
        
        # Clear input and rerun
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Powered by Linda Goodman's mystical wisdom ✨<br>
        Built with FastAPI, OpenAI, and FAISS
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 