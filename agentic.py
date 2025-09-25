import os
import streamlit as st
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
# Updated Google AI import - using the NEW unified SDK
from google import genai
from google.genai import types
from tavily import TavilyClient
import json
import hashlib
from datetime import datetime
import logging
from langgraph.graph import Graph, StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Astellas Agentic RAG Assistant",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stAlert > div {
        padding: 10px;
        border-radius: 5px;
    }
    .thinking-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .source-box {
        background-color: #e8f4fd;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #1f77b4;
        margin: 5px 0;
    }
    .agent-status {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
    }
    .status-thinking { background-color: #fff3cd; color: #856404; }
    .status-complete { background-color: #d4edda; color: #155724; }
    .status-error { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    COMPLETE = "complete"
    ERROR = "error"

# Use TypedDict for better LangGraph compatibility
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    query_plan: Dict[str, Any]
    retrieved_chunks: List[Dict]
    chunk_grades: List[Dict]
    web_search_results: List[Dict]
    final_answer: str
    data_lineage: List[Dict]
    agent_thoughts: List[Dict]
    current_agent: str
    agent_status: Dict[str, str]

# Helper function to create initial state
def create_initial_state(user_query: str) -> AgentState:
    return AgentState(
        messages=[],
        user_query=user_query,
        query_plan={},
        retrieved_chunks=[],
        chunk_grades=[],
        web_search_results=[],
        final_answer="",
        data_lineage=[],
        agent_thoughts=[],
        current_agent="",
        agent_status={
            "query_analyzer": AgentStatus.IDLE.value,
            "retriever": AgentStatus.IDLE.value,
            "evaluator": AgentStatus.IDLE.value,
            "web_searcher": AgentStatus.IDLE.value,
            "answer_generator": AgentStatus.IDLE.value
        }
    )

class MongoDBRetriever:
    def __init__(self, mongodb_uri: str, db_name: str = "astellas", collection_name: str = "astellas-web"):
        self.client = MongoClient(mongodb_uri, server_api=ServerApi('1'))
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
    def vector_search(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """Perform vector search in MongoDB Atlas"""
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "astellas_vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "chunk_text": 1,
                        "filename": 1,
                        "file_type": 1,
                        "chunk_index": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []

class EmbeddingModel:
    def __init__(self, model_name: str = "multi-qa-mpnet-base-cos-v1"):
        try:
            # Initialize with specific device and trust_remote_code settings for compatibility
            self.model = SentenceTransformer(
                model_name, 
                device='cpu',  # Force CPU to avoid GPU compatibility issues
                trust_remote_code=False
            )
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, falling back to all-MiniLM-L6-v2: {str(e)}")
            # Fallback to a more reliable model
            self.model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device='cpu',
                trust_remote_code=False
            )
        
    def encode(self, text: str) -> List[float]:
        try:
            # Ensure text is properly formatted
            text = str(text).strip()
            if not text:
                text = "empty query"
            
            # Encode with error handling
            embedding = self.model.encode(text, convert_to_tensor=False, show_progress_bar=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            # Return a default embedding vector (384 dimensions for all-MiniLM-L6-v2)
            return [0.0] * 384

class ModernGeminiClient:
    """Updated Gemini client using the new Google GenAI SDK"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-001"):
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize the new Google GenAI client
        self.client = genai.Client(api_key=api_key)
        
        # Store default generation parameters
        self.default_generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048
        }
        
        # Safety settings - using the new format
        self.default_safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", 
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            )
        ]

    def generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using the new Google GenAI SDK"""
        try:
            # Merge default params with any overrides
            generation_params = {**self.default_generation_config, **kwargs}
            
            # Create the generation config using the new types
            config = types.GenerateContentConfig(
                temperature=generation_params.get("temperature", 0.7),
                top_p=generation_params.get("top_p", 0.8),
                top_k=generation_params.get("top_k", 40),
                max_output_tokens=generation_params.get("max_output_tokens", 2048),
                safety_settings=self.default_safety_settings
            )
            
            # Make the API call using the new client structure
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # Extract text from response - the new SDK has a .text property
            if hasattr(response, 'text') and response.text:
                return response.text
            else:
                logger.warning("No text in response or response blocked")
                return ""
            
        except Exception as e:
            logger.error(f"Error generating content with new SDK: {str(e)}")
            return ""

    async def generate_content_async(self, prompt: str, **kwargs) -> str:
        """Async version of content generation using the new SDK"""
        try:
            # Merge default params with any overrides
            generation_params = {**self.default_generation_config, **kwargs}
            
            # Create the generation config
            config = types.GenerateContentConfig(
                temperature=generation_params.get("temperature", 0.7),
                top_p=generation_params.get("top_p", 0.8),
                top_k=generation_params.get("top_k", 40),
                max_output_tokens=generation_params.get("max_output_tokens", 2048),
                safety_settings=self.default_safety_settings
            )
            
            # Make the async API call
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # Extract text from response
            if hasattr(response, 'text') and response.text:
                return response.text
            else:
                logger.warning("No text in async response or response blocked")
                return ""
            
        except Exception as e:
            logger.error(f"Error in async content generation with new SDK: {str(e)}")
            return ""

class AgenticRAGWorkflow:
    def __init__(self, mongodb_uri: str, gemini_api_key: str, tavily_api_key: str):
        # Initialize components
        self.retriever = MongoDBRetriever(mongodb_uri)
        self.embedding_model = EmbeddingModel()

        # Initialize the modern Gemini client with the new SDK
        self.gemini_client = ModernGeminiClient(gemini_api_key)

        # Configure Tavily
        self.tavily_client = TavilyClient(api_key=tavily_api_key)

        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("query_analyzer", self.query_analyzer_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("evaluator", self.evaluator_node)
        workflow.add_node("web_searcher", self.web_searcher_node)
        workflow.add_node("answer_generator", self.answer_generator_node)
        
        # Add edges
        workflow.add_edge(START, "query_analyzer")
        workflow.add_edge("query_analyzer", "retriever")
        workflow.add_edge("retriever", "evaluator")
        workflow.add_conditional_edges(
            "evaluator",
            self.should_web_search,
            {
                "web_search": "web_searcher",
                "generate": "answer_generator"
            }
        )
        workflow.add_edge("web_searcher", "answer_generator")
        workflow.add_edge("answer_generator", END)
        
        return workflow.compile()
    
    def query_analyzer_node(self, state: AgentState) -> AgentState:
        """Analyze user query and create a plan"""
        state['current_agent'] = "Query Analyzer"
        state['agent_status']["query_analyzer"] = AgentStatus.THINKING.value

        state['agent_thoughts'].append({
            "agent": "Query Analyzer",
            "timestamp": datetime.now().isoformat(),
            "thought": "Analyzing user query to understand intent and entities..."
        })

        try:
            analysis_prompt = f"""
            Analyze the following user query and provide a structured plan in valid JSON format:

            Query: {state['user_query']}

            Please provide a JSON response with these fields:
            - "intent": one of ["factual_qa", "multi_step_reasoning", "comparison", "explanation"]
            - "entities": list of key entities mentioned
            - "complexity": one of ["simple", "moderate", "complex"]
            - "retrieval_strategy": one of ["single_search", "multi_search"]
            - "multi_search_needed": boolean

            Only respond with valid JSON, no additional text.
            """

            response_text = self.gemini_client.generate_content(
                analysis_prompt,
                temperature=0.3,
                max_output_tokens=512
            )

            if not response_text or response_text.strip() == "":
                logger.warning("Empty response from Gemini, using default plan")
                raise ValueError("Empty response from LLM")

            # Clean up response text to extract JSON
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Additional cleanup for common formatting issues
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            try:
                query_plan = json.loads(response_text)
                
                # Validate the required fields exist
                required_fields = ["intent", "entities", "complexity", "retrieval_strategy", "multi_search_needed"]
                for field in required_fields:
                    if field not in query_plan:
                        logger.warning(f"Missing field {field} in query plan, using default")
                        query_plan[field] = self._get_default_value(field)
                        
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing query plan JSON: {str(e)}")
                logger.error(f"Response text was: {response_text}")
                # Create a default plan
                query_plan = self._create_default_plan()
                query_plan["parse_error"] = True

            state['query_plan'] = query_plan
            state['agent_status']["query_analyzer"] = AgentStatus.COMPLETE.value

            state['agent_thoughts'].append({
                "agent": "Query Analyzer",
                "timestamp": datetime.now().isoformat(),
                "thought": f"Query analysis complete. Intent: {query_plan.get('intent', 'unknown')}, Complexity: {query_plan.get('complexity', 'unknown')}"
            })

        except Exception as e:
            logger.error(f"Error in query analyzer node: {str(e)}")
            state['agent_status']["query_analyzer"] = AgentStatus.ERROR.value
            state['agent_thoughts'].append({
                "agent": "Query Analyzer",
                "timestamp": datetime.now().isoformat(),
                "thought": f"Error in analysis: {str(e)}"
            })
            state['query_plan'] = self._create_default_plan()
            state['query_plan']["error"] = str(e)

        return state
    
    def _get_default_value(self, field: str):
        """Get default value for missing fields"""
        defaults = {
            "intent": "factual_qa",
            "entities": [],
            "complexity": "simple",
            "retrieval_strategy": "single_search",
            "multi_search_needed": False
        }
        return defaults.get(field, None)
    
    def _create_default_plan(self) -> Dict[str, Any]:
        """Create a default query plan"""
        return {
            "intent": "factual_qa",
            "entities": [],
            "complexity": "simple",
            "retrieval_strategy": "single_search",
            "multi_search_needed": False
        }
    
    def retriever_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant chunks from MongoDB"""
        state['current_agent'] = "Retriever"
        state['agent_status']["retriever"] = AgentStatus.THINKING.value

        state['agent_thoughts'].append({
            "agent": "Retriever",
            "timestamp": datetime.now().isoformat(),
            "thought": "Generating query embeddings and searching vector database..."
        })

        try:
            query_embedding = self.embedding_model.encode(state['user_query'])
            chunks = self.retriever.vector_search(query_embedding, limit=5)
            state['retrieved_chunks'] = chunks

            state['agent_status']["retriever"] = AgentStatus.COMPLETE.value
            state['agent_thoughts'].append({
                "agent": "Retriever",
                "timestamp": datetime.now().isoformat(),
                "thought": f"Retrieved {len(chunks)} relevant chunks from the knowledge base"
            })

            state['data_lineage'].append({
                "step": "retrieval",
                "source": "MongoDB Vector Atlas",
                "num_chunks": len(chunks),
                "files": list(set([chunk.get("filename", "unknown") for chunk in chunks]))
            })

        except Exception as e:
            logger.error(f"Error in retriever node: {str(e)}")
            state['agent_status']["retriever"] = AgentStatus.ERROR.value
            state['agent_thoughts'].append({
                "agent": "Retriever",
                "timestamp": datetime.now().isoformat(),
                "thought": f"Error in retrieval: {str(e)}"
            })
            state['retrieved_chunks'] = []

        return state
    
    def evaluator_node(self, state: AgentState) -> AgentState:
        """Evaluate the quality and relevance of retrieved chunks"""
        state['current_agent'] = "Evaluator"
        state['agent_status']["evaluator"] = AgentStatus.THINKING.value
        
        state['agent_thoughts'].append({
            "agent": "Evaluator",
            "timestamp": datetime.now().isoformat(),
            "thought": "Evaluating retrieved chunks for relevance and completeness..."
        })
        
        try:
            chunk_grades = []
            
            for i, chunk in enumerate(state['retrieved_chunks']):
                evaluation_prompt = f"""
                Evaluate if the following chunk is relevant to answer the user query.
                
                User Query: {state['user_query']}
                
                Chunk Content: {chunk.get('chunk_text', '')[:500]}...
                
                Respond with valid JSON only:
                {{"score": [1-10], "reason": "brief explanation", "can_answer": [true/false]}}
                """
                
                try:
                    response_text = self.gemini_client.generate_content(
                        evaluation_prompt,
                        temperature=0.2,
                        max_output_tokens=256
                    )
                    
                    # Clean up response text
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    grade = json.loads(response_text)
                except:
                    grade = {"score": 5, "reason": "evaluation failed", "can_answer": False}
                
                grade["chunk_index"] = i
                grade["filename"] = chunk.get("filename", "unknown")
                chunk_grades.append(grade)
            
            state['chunk_grades'] = chunk_grades
            
            # Determine if chunks are sufficient
            high_quality_chunks = [g for g in chunk_grades if g.get("score", 0) >= 7]
            answerable_chunks = [g for g in chunk_grades if g.get("can_answer", False)]
            
            state['agent_status']["evaluator"] = AgentStatus.COMPLETE.value
            state['agent_thoughts'].append({
                "agent": "Evaluator",
                "timestamp": datetime.now().isoformat(),
                "thought": f"Evaluation complete. {len(high_quality_chunks)} high-quality chunks, {len(answerable_chunks)} potentially answerable chunks"
            })
            
        except Exception as e:
            logger.error(f"Error in evaluator node: {str(e)}")
            state['agent_status']["evaluator"] = AgentStatus.ERROR.value
            state['agent_thoughts'].append({
                "agent": "Evaluator",
                "timestamp": datetime.now().isoformat(),
                "thought": f"Error in evaluation: {str(e)}"
            })
            state['chunk_grades'] = []
        
        return state
    
    def should_web_search(self, state: AgentState) -> str:
        """Decide whether to perform web search"""
        if not state['chunk_grades']:
            return "web_search"
        
        # Check if we have good enough chunks
        high_quality_chunks = [g for g in state['chunk_grades'] if g.get("score", 0) >= 7]
        answerable_chunks = [g for g in state['chunk_grades'] if g.get("can_answer", False)]
        
        if len(high_quality_chunks) >= 2 or len(answerable_chunks) >= 1:
            return "generate"
        else:
            return "web_search"
    
    def web_searcher_node(self, state: AgentState) -> AgentState:
        """Perform web search using Tavily"""
        state['current_agent'] = "Web Searcher"
        state['agent_status']["web_searcher"] = AgentStatus.THINKING.value
        
        state['agent_thoughts'].append({
            "agent": "Web Searcher",
            "timestamp": datetime.now().isoformat(),
            "thought": "Knowledge base insufficient. Searching the web for additional information..."
        })
        
        try:
            # Perform web search
            search_results = self.tavily_client.search(
                query=state['user_query'],
                search_depth="advanced",
                max_results=3
            )
            
            state['web_search_results'] = search_results.get("results", [])
            
            state['agent_status']["web_searcher"] = AgentStatus.COMPLETE.value
            state['agent_thoughts'].append({
                "agent": "Web Searcher",
                "timestamp": datetime.now().isoformat(),
                "thought": f"Found {len(state['web_search_results'])} web sources to supplement the answer"
            })
            
            # Add to data lineage
            state['data_lineage'].append({
                "step": "web_search",
                "source": "Tavily Web Search",
                "num_results": len(state['web_search_results']),
                "urls": [result.get("url", "") for result in state['web_search_results']]
            })
            
        except Exception as e:
            logger.error(f"Error in web searcher node: {str(e)}")
            state['agent_status']["web_searcher"] = AgentStatus.ERROR.value
            state['agent_thoughts'].append({
                "agent": "Web Searcher",
                "timestamp": datetime.now().isoformat(),
                "thought": f"Error in web search: {str(e)}"
            })
            state['web_search_results'] = []
        
        return state
    
    def answer_generator_node(self, state: AgentState) -> AgentState:
        """Generate final answer using all available information"""
        state['current_agent'] = "Answer Generator"
        state['agent_status']["answer_generator"] = AgentStatus.THINKING.value
        
        state['agent_thoughts'].append({
            "agent": "Answer Generator",
            "timestamp": datetime.now().isoformat(),
            "thought": "Synthesizing information from knowledge base and web search to generate comprehensive answer..."
        })
        
        try:
            # Prepare context from chunks
            kb_context = ""
            for i, chunk in enumerate(state['retrieved_chunks']):
                if i < len(state['chunk_grades']) and state['chunk_grades'][i].get("score", 0) >= 6:
                    kb_context += f"Source: {chunk.get('filename', 'unknown')}\n"
                    kb_context += f"Content: {chunk.get('chunk_text', '')}\n\n"
            
            # Prepare context from web search
            web_context = ""
            for result in state['web_search_results']:
                web_context += f"Source: {result.get('url', 'unknown')}\n"
                web_context += f"Title: {result.get('title', '')}\n"
                web_context += f"Content: {result.get('content', '')[:500]}...\n\n"
            
            # Generate answer
            answer_prompt = f"""
            Based on the following information sources, provide a comprehensive answer to the user's question.
            
            User Question: {state['user_query']}
            
            Knowledge Base Information:
            {kb_context}
            
            Web Search Information:
            {web_context}
            
            Please provide:
            1. A clear, comprehensive answer
            2. Cite your sources appropriately
            3. If information is conflicting or incomplete, mention this
            4. Be factual and precise
            """
            
            final_answer = self.gemini_client.generate_content(
                answer_prompt,
                temperature=0.7,
                max_output_tokens=1024
            )
            
            state['final_answer'] = final_answer
            
            state['agent_status']["answer_generator"] = AgentStatus.COMPLETE.value
            state['agent_thoughts'].append({
                "agent": "Answer Generator",
                "timestamp": datetime.now().isoformat(),
                "thought": "Generated comprehensive answer combining knowledge base and web search results"
            })
            
            # Final data lineage
            state['data_lineage'].append({
                "step": "answer_generation",
                "source": "Google Gemini 2.0 Flash",
                "kb_sources": len([g for g in state['chunk_grades'] if g.get("score", 0) >= 6]),
                "web_sources": len(state['web_search_results'])
            })
            
        except Exception as e:
            logger.error(f"Error in answer generator node: {str(e)}")
            state['agent_status']["answer_generator"] = AgentStatus.ERROR.value
            state['agent_thoughts'].append({
                "agent": "Answer Generator",
                "timestamp": datetime.now().isoformat(),
                "thought": f"Error in answer generation: {str(e)}"
            })
            state['final_answer'] = f"I encountered an error while generating the answer: {str(e)}"
        
        return state
    
    def run_workflow(self, user_query: str) -> AgentState:
        """Run the complete agentic workflow"""
        initial_state = create_initial_state(user_query)

        try:
            # Run the workflow
            final_state_result = self.graph.invoke(initial_state)

            # LangGraph returns the state as a dict-like object
            # Ensure we have all the required fields
            final_state = AgentState(
                messages=final_state_result.get('messages', []),
                user_query=final_state_result.get('user_query', user_query),
                query_plan=final_state_result.get('query_plan', {}),
                retrieved_chunks=final_state_result.get('retrieved_chunks', []),
                chunk_grades=final_state_result.get('chunk_grades', []),
                web_search_results=final_state_result.get('web_search_results', []),
                final_answer=final_state_result.get('final_answer', "No answer generated"),
                data_lineage=final_state_result.get('data_lineage', []),
                agent_thoughts=final_state_result.get('agent_thoughts', []),
                current_agent=final_state_result.get('current_agent', ""),
                agent_status=final_state_result.get('agent_status', {
                    "query_analyzer": AgentStatus.IDLE.value,
                    "retriever": AgentStatus.IDLE.value,
                    "evaluator": AgentStatus.IDLE.value,
                    "web_searcher": AgentStatus.IDLE.value,
                    "answer_generator": AgentStatus.IDLE.value
                })
            )
            
            return final_state

        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}")
            # Return an error state
            error_state = create_initial_state(user_query)
            error_state.update({
                'final_answer': f"Error: {str(e)}",
                'agent_thoughts': [{
                    "agent": "System",
                    "timestamp": datetime.now().isoformat(),
                    "thought": f"Workflow error: {str(e)}"
                }],
                'current_agent': "System",
                'agent_status': {
                    "query_analyzer": AgentStatus.ERROR.value,
                    "retriever": AgentStatus.ERROR.value,
                    "evaluator": AgentStatus.ERROR.value,
                    "web_searcher": AgentStatus.ERROR.value,
                    "answer_generator": AgentStatus.ERROR.value
                }
            })
            return error_state

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "workflow" not in st.session_state:
        st.session_state.workflow = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = True

def main():
    with open("icon.png", "r") as f:
    png_icon = f.read()
    st.markdown(f"""
        <div style="display: flex; align-items: center;">
            {png_icon}
            <h1>Agentic RAG Assistant</h1>
        </div>""", unsafe_allow_html=True)
    
    st.title("🤖 Agentic RAG Assistant")
    st.markdown("*Intelligent document search with multi-agent reasoning and web search capabilities*")
    
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        mongodb_uri = st.text_input("MongoDB URI", type="password", value=os.getenv("MONGODB_URI", ""))
        gemini_api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        tavily_api_key = st.text_input("Tavily API Key", type="password", value=os.getenv("TAVILY_API_KEY", ""))
        
        # Options
        st.session_state.show_thinking = st.checkbox("Show Agent Thinking Process", value=True)
        
        # Initialize workflow
        if st.button("Initialize System"):
            if mongodb_uri and gemini_api_key and tavily_api_key:
                try:
                    st.session_state.workflow = AgenticRAGWorkflow(
                        mongodb_uri, gemini_api_key, tavily_api_key
                    )
                    st.success("✅ System initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Initialization failed: {str(e)}")
            else:
                st.error("Please provide all required API keys")
        
        # Status indicator
        if st.session_state.workflow:
            st.success("🟢 System Ready")
        else:
            st.warning("🟡 System Not Initialized")
    
    # Main interface
    if st.session_state.workflow:
        # Query input
        user_query = st.text_input(
            "Ask your question:",
            placeholder="What would you like to know?",
            key="query_input"
        )
        
        if st.button("🔍 Search", disabled=not user_query):
            if user_query:
                # Create columns for real-time updates
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    answer_placeholder = st.empty()
                    
                with col2:
                    if st.session_state.show_thinking:
                        thinking_placeholder = st.empty()
                        status_placeholder = st.empty()
                
                # Run workflow
                with st.spinner("Processing your query..."):
                    try:
                        result = st.session_state.workflow.run_workflow(user_query)
                        
                        # Display answer
                        with col1:
                            answer_placeholder.markdown("### 🎯 Answer")
                            st.markdown(result['final_answer'])
                            
                            # Display data lineage
                            st.markdown("### 📊 Data Lineage")
                            for step in result['data_lineage']:
                                with st.expander(f"Step: {step['step'].title()}", expanded=False):
                                    st.json(step)
                            
                            # Display sources
                            if result['retrieved_chunks']:
                                st.markdown("### 📚 Knowledge Base Sources")
                                for i, chunk in enumerate(result['retrieved_chunks']):
                                    if i < len(result['chunk_grades']):
                                        grade = result['chunk_grades'][i]
                                        score = grade.get('score', 0)
                                        if score >= 6:
                                            with st.expander(f"📄 {chunk.get('filename', 'Unknown')} (Score: {score}/10)"):
                                                st.text(chunk.get('chunk_text', '')[:500] + "...")
                            
                            if result['web_search_results']:
                                st.markdown("### 🌐 Web Sources")
                                for result_item in result['web_search_results']:
                                    with st.expander(f"🔗 {result_item.get('title', 'Web Result')}"):
                                        st.write(f"**URL:** {result_item.get('url', 'N/A')}")
                                        st.write(f"**Content:** {result_item.get('content', '')[:300]}...")
                        
                        # Display thinking process
                        if st.session_state.show_thinking:
                            with col2:
                                thinking_placeholder.markdown("### 🧠 Agent Thinking")
                                
                                # Agent status
                                status_placeholder.markdown("### 📊 Agent Status")
                                for agent, status in result['agent_status'].items():
                                    status_class = f"status-{status}"
                                    st.markdown(
                                        f'<span class="agent-status {status_class}">{agent.replace("_", " ").title()}: {status.title()}</span>',
                                        unsafe_allow_html=True
                                    )
                                
                                # Thinking process
                                for thought in result['agent_thoughts']:
                                    st.markdown(
                                        f"""
                                        <div class="thinking-box">
                                            <strong>{thought['agent']}</strong><br>
                                            <small>{thought['timestamp']}</small><br>
                                            {thought['thought']}
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "query": user_query,
                            "answer": result['final_answer'],
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
        
        # Chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 💬 Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q: {chat['query'][:50]}..." if len(chat['query']) > 50 else f"Q: {chat['query']}", expanded=False):
                    st.markdown(f"**Query:** {chat['query']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
                    st.markdown(f"**Time:** {chat['timestamp']}")
    
    else:
        st.info("👆 Please configure and initialize the system using the sidebar.")
        
        # Demo section
        st.markdown("---")
        st.markdown("### 🚀 Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔍 Intelligent Query Analysis**
            - Intent recognition
            - Entity extraction
            - Complexity assessment
            - Multi-step planning
            """)
        
        with col2:
            st.markdown("""
            **📊 Smart Retrieval & Evaluation**
            - Vector similarity search
            - Chunk relevance scoring
            - Quality assessment
            - Source validation
            """)
        
        with col3:
            st.markdown("""
            **🌐 Web Search Integration**
            - Fallback web search
            - Multi-source synthesis
            - Data lineage tracking
            - Source attribution
            """)

if __name__ == "__main__":
    main()
