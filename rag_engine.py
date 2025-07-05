import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()


class RAGEngine:
    """Main RAG engine that combines retrieval and generation"""
    
    def __init__(self, 
                 embedding_index_manager,
                 model: str = "gpt-4",
                 max_tokens: int = 500,
                 temperature: float = 0.8):
        self.embedding_index_manager = embedding_index_manager
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Linda Goodman's direct and concise tone prompt
        self.system_prompt = """You are an astrological advisor with access to Linda Goodman's "Sun Signs" and "Love Signs." 
        
        When responding to questions about astrology, relationships, and zodiac signs:
        - Give blunt, practical answers without unnecessary flowery language
        - Focus on key facts and actionable insights
        - Keep responses under 4 - 5 sentences when possible
        - Be honest and straightforward about what the sources say
        - Provide practical wisdom wrapped in mystical understanding
        
        
        If the question cannot be answered from the provided context, respond with: 
        "This isn't mentioned in the source material."
        
        Always base your responses on the astrological knowledge provided in the context."""
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query"""
        return self.embedding_index_manager.search_similar(query, top_k)
    
    def create_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Create context string from retrieved chunks"""
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('source_file', 'Unknown')
            text = chunk.get('text', '')
            score = chunk.get('similarity_score', 0)
            
            context_parts.append(f"Source {i} ({source}, relevance: {score:.3f}):\n{text}\n")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI GPT-4"""
        if not context.strip():
            return "This isn't mentioned in the source material."
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nGive a direct, blunt answer based on the context. Keep it short and to the point."}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a complete query through the RAG pipeline"""
        if not query.strip():
            return {
                "answer": "Please provide a question about astrology.",
                "sources": [],
                "context_used": False
            }
        
        # Retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(query, top_k)
        
        # Check if we have relevant information
        if not chunks or all(chunk.get('similarity_score', 0) < 0.5 for chunk in chunks):
            return {
                "answer": "This isn't mentioned in the source material.",
                "sources": [],
                "context_used": False
            }
        
        # Create context
        context = self.create_context(chunks)
        
        # Generate response
        answer = self.generate_response(query, context)
        
        # Prepare sources information
        sources = []
        for chunk in chunks:
            sources.append({
                "source_file": chunk.get('source_file', 'Unknown'),
                "similarity_score": chunk.get('similarity_score', 0),
                "text_preview": chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": True,
            "query": query,
            "chunks_retrieved": len(chunks)
        }
    
    def get_astrology_insight(self, zodiac_sign: str, insight_type: str = "general") -> Dict[str, Any]:
        """Get specific astrology insights for a zodiac sign"""
        query = f"Tell me about {zodiac_sign} {insight_type} characteristics and traits"
        return self.process_query(query)
    
    def get_love_compatibility(self, sign1: str, sign2: str) -> Dict[str, Any]:
        """Get love compatibility between two zodiac signs"""
        query = f"What is the love compatibility between {sign1} and {sign2}?"
        return self.process_query(query)
    
    def get_daily_horoscope_style(self, zodiac_sign: str) -> Dict[str, Any]:
        """Get horoscope-style insights for a zodiac sign"""
        query = f"Give me a mystical horoscope reading for {zodiac_sign} in Linda Goodman's style"
        return self.process_query(query)


class QueryProcessor:
    """Helper class for processing and validating queries"""
    
    @staticmethod
    def clean_query(query: str) -> str:
        """Clean and normalize query"""
        return query.strip()
    
    @staticmethod
    def extract_zodiac_signs(query: str) -> List[str]:
        """Extract zodiac signs from query"""
        zodiac_signs = [
            "aries", "taurus", "gemini", "cancer", "leo", "virgo",
            "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"
        ]
        
        found_signs = []
        query_lower = query.lower()
        
        for sign in zodiac_signs:
            if sign in query_lower:
                found_signs.append(sign.title())
        
        return found_signs
    
    @staticmethod
    def is_astrology_related(query: str) -> bool:
        """Check if query is astrology-related"""
        astrology_keywords = [
            "zodiac", "horoscope", "astrology", "birth chart", "natal chart",
            "aries", "taurus", "gemini", "cancer", "leo", "virgo",
            "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces",
            "sun sign", "moon sign", "rising sign", "ascendant",
            "compatibility", "love", "relationship", "personality",
            "linda goodman", "sun signs", "love signs"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in astrology_keywords)
    
    @staticmethod
    def suggest_queries() -> List[str]:
        """Suggest example queries"""
        return [
            "Tell me about Aries personality traits",
            "What is the compatibility between Leo and Aquarius?",
            "How does Linda Goodman describe Cancer in relationships?",
            "What are the characteristics of a Taurus sun sign?",
            "Tell me about love and romance for Scorpio",
            "What does Linda Goodman say about Virgo's approach to love?",
            "How do Gemini and Sagittarius get along?",
            "What are the mystical qualities of Pisces?",
            "Tell me about Capricorn's career and ambition",
            "What does Linda Goodman say about Libra's sense of balance?"
        ] 