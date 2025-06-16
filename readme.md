# 🤖 Agentic RAG Solution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-FF6B6B.svg)](https://streamlit.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://www.mongodb.com/atlas)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4.8-purple.svg)](https://langchain-ai.github.io/langgraph/)

An intelligent document search and question-answering system that combines **Retrieval-Augmented Generation (RAG)** with **multi-agent reasoning** capabilities. The system intelligently retrieves information from a knowledge base and supplements it with real-time web search when needed.

## 🎯 Overview

This solution implements an advanced RAG architecture using multiple AI agents that work collaboratively to provide comprehensive, accurate answers to user queries. The system automatically determines when local knowledge is sufficient or when external web search is needed.

### Key Features

- 🧠 **Multi-Agent Architecture**: Five specialized agents working in orchestrated workflow
- 🔍 **Intelligent Query Analysis**: Automatic intent recognition and complexity assessment
- 📊 **Vector Search**: MongoDB Atlas vector search with semantic similarity
- 🌐 **Hybrid Information Retrieval**: Knowledge base + real-time web search
- ⚡ **Smart Evaluation**: Automatic relevance scoring and quality assessment
- 📈 **Real-time Monitoring**: Live agent status and thinking process visualization
- 🔄 **Data Lineage Tracking**: Complete transparency of information sources

## 🏗️ Architecture Overview

```mermaid
graph TB
    subgraph "User Interface"
        UI[Streamlit Web App]
        USER[User Query]
    end
    
    subgraph "Agentic Workflow (LangGraph)"
        QA[Query Analyzer Agent]
        RET[Retriever Agent]
        EVAL[Evaluator Agent]
        WS[Web Search Agent]
        AG[Answer Generator Agent]
    end
    
    subgraph "Data Sources"
        MONGO[(MongoDB Atlas<br/>Vector Database)]
        WEB[Tavily Web Search]
    end
    
    subgraph "AI Models"
        EMBED[Sentence Transformer<br/>Embeddings]
        LLM[Google Gemini 2.0<br/>Flash]
    end
    
    USER --> UI
    UI --> QA
    
    QA --> RET
    RET --> MONGO
    RET --> EMBED
    
    RET --> EVAL
    EVAL --> |Insufficient Knowledge| WS
    EVAL --> |Sufficient Knowledge| AG
    
    WS --> WEB
    WS --> AG
    
    AG --> LLM
    AG --> UI
    
    style QA fill:#e1f5fe
    style RET fill:#f3e5f5
    style EVAL fill:#fff3e0
    style WS fill:#e8f5e8
    style AG fill:#fce4ec
```

## 🔄 Agent Workflow

The system operates through a sophisticated multi-agent workflow:

### 1. **Query Analyzer Agent** 🔍
- **Purpose**: Understand user intent and query complexity
- **Actions**: 
  - Analyzes query intent (factual QA, reasoning, comparison, explanation)
  - Extracts key entities and concepts
  - Determines complexity level (simple, moderate, complex)
  - Plans retrieval strategy

```python
# Example Query Analysis Output
{
    "intent": "factual_qa",
    "entities": ["AI services", "HCL"],
    "complexity": "moderate",
    "retrieval_strategy": "multi_search",
    "multi_search_needed": false
}
```

### 2. **Retriever Agent** 📚
- **Purpose**: Fetch relevant information from knowledge base
- **Actions**:
  - Generates query embeddings using Sentence Transformers
  - Performs vector similarity search in MongoDB Atlas
  - Retrieves top-k most relevant document chunks
  - Maintains data lineage for transparency

### 3. **Evaluator Agent** ⚖️
- **Purpose**: Assess quality and relevance of retrieved information
- **Actions**:
  - Scores each chunk for relevance (1-10 scale)
  - Determines if chunks can answer the query
  - Provides reasoning for evaluation decisions
  - Decides if web search is needed

### 4. **Web Search Agent** 🌐
- **Purpose**: Supplement knowledge base with real-time information
- **Actions**:
  - Performs advanced web search using Tavily
  - Retrieves current, relevant web sources
  - Adds web results to context for answer generation
  - Only activated when knowledge base is insufficient

### 5. **Answer Generator Agent** ✨
- **Purpose**: Synthesize final comprehensive answer
- **Actions**:
  - Combines knowledge base and web search results
  - Generates coherent, well-sourced responses
  - Provides proper attribution to sources
  - Maintains factual accuracy and completeness

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit 1.45.1 | Interactive web interface |
| **Orchestration** | LangGraph 0.4.8 | Multi-agent workflow management |
| **Vector DB** | MongoDB Atlas | Document storage and vector search |
| **Embeddings** | Sentence Transformers | Text-to-vector conversion |
| **LLM** | Google Gemini 2.0 Flash | Natural language generation |
| **Web Search** | Tavily API | Real-time web information retrieval |
| **Text Processing** | LangChain Text Splitters | Document chunking |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MongoDB Atlas account
- Google AI API key (Gemini)
- Tavily API key

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd agentic-rag-solution

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DB_NAME=hcl
MONGODB_COLLECTION_NAME=hcl-web
VECTOR_INDEX_NAME=hcl_vector_index

# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Model Configuration
EMBEDDING_MODEL=multi-qa-mpnet-base-cos-v1
LLM_MODEL=gemini-2.0-flash-001

# Search Configuration
VECTOR_SEARCH_LIMIT=5
WEB_SEARCH_MAX_RESULTS=3
RELEVANCE_THRESHOLD=7.0
```

### 3. Data Ingestion

#### Option A: Web Scraping (Automated)
```bash
# Scrape HCL websites and download PDFs
python scrape.py
```

#### Option B: Manual Document Upload
1. Place PDF files in `downloaded_pdfs/` directory
2. Place Markdown files in `scraped_content/` directory

#### Ingest Documents to MongoDB
```bash
# Process and ingest documents
python ingestion_mongo.py
```

### 4. Launch Application

```bash
# Using the launcher script
python app.py

# Or directly with Streamlit
streamlit run agentic.py
```

The application will be available at `http://localhost:8501`

## 📊 System Components

### Data Ingestion Pipeline

```mermaid
flowchart LR
    subgraph "Data Sources"
        WEB[Web Scraping]
        PDF[PDF Documents]
        MD[Markdown Files]
    end
    
    subgraph "Processing"
        EXTRACT[Text Extraction]
        CHUNK[Text Chunking]
        EMBED[Generate Embeddings]
    end
    
    subgraph "Storage"
        MONGO[(MongoDB Atlas)]
        INDEX[Vector Index]
    end
    
    WEB --> EXTRACT
    PDF --> EXTRACT
    MD --> EXTRACT
    
    EXTRACT --> CHUNK
    CHUNK --> EMBED
    EMBED --> MONGO
    MONGO --> INDEX
```

### Query Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant QA as Query Analyzer
    participant RET as Retriever
    participant EVAL as Evaluator
    participant WS as Web Search
    participant AG as Answer Generator
    participant DB as MongoDB
    participant Web as Tavily API
    
    User->>UI: Submit Query
    UI->>QA: Analyze Query
    QA->>QA: Extract Intent & Entities
    QA->>RET: Query Plan
    
    RET->>DB: Vector Search
    DB->>RET: Document Chunks
    RET->>EVAL: Retrieved Chunks
    
    EVAL->>EVAL: Score Relevance
    
    alt Insufficient Knowledge
        EVAL->>WS: Trigger Web Search
        WS->>Web: Search Query
        Web->>WS: Web Results
        WS->>AG: Combined Context
    else Sufficient Knowledge
        EVAL->>AG: KB Context Only
    end
    
    AG->>AG: Generate Answer
    AG->>UI: Final Response
    UI->>User: Display Answer
```

## 🎛️ Configuration Options

### MongoDB Settings
- `MONGODB_URI`: Connection string for MongoDB Atlas
- `MONGODB_DB_NAME`: Database name (default: "hcl")
- `MONGODB_COLLECTION_NAME`: Collection name (default: "hcl-web")
- `VECTOR_INDEX_NAME`: Vector search index name

### Search Parameters
- `VECTOR_SEARCH_LIMIT`: Number of chunks to retrieve (default: 5)
- `VECTOR_SEARCH_CANDIDATES`: Search candidates pool (default: 50)
- `RELEVANCE_THRESHOLD`: Minimum relevance score (default: 7.0)
- `WEB_SEARCH_MAX_RESULTS`: Maximum web search results (default: 3)

### Model Configuration
- `EMBEDDING_MODEL`: Sentence transformer model
- `LLM_MODEL`: Gemini model version
- `LOG_LEVEL`: Logging verbosity

## 📁 Project Structure

```
agentic-rag-solution/
├── agentic.py              # Main Streamlit application
├── app.py                  # Application launcher
├── config.py               # Configuration management
├── ingestion_mongo.py      # Document ingestion pipeline
├── scrape.py              # Web scraping utility
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
├── downloaded_pdfs/       # PDF document storage
├── scraped_content/       # Scraped markdown content
└── README.md             # This file
```

## 🔧 Usage Examples

### Basic Query
```
User: "What AI services does HCL offer?"

System Response:
- Analyzes intent: factual_qa
- Retrieves relevant chunks from knowledge base
- Evaluates chunk quality and relevance
- Generates comprehensive answer with sources
```

### Complex Analysis Query
```
User: "Compare HCL's AI capabilities with current market trends"

System Response:
- Identifies complexity: complex comparison
- Retrieves HCL-specific information from KB
- Supplements with web search for market trends
- Synthesizes comparative analysis
```

## 🎨 User Interface Features

### Main Dashboard
- **Query Input**: Natural language question interface
- **Real-time Status**: Live agent status indicators
- **Answer Display**: Formatted responses with source attribution
- **Data Lineage**: Complete information source tracking

### Agent Thinking Process
- **Live Updates**: Real-time agent thoughts and decisions
- **Status Indicators**: Visual agent state monitoring
- **Transparency**: Complete workflow visibility

### Source Attribution
- **Knowledge Base Sources**: Document chunks with relevance scores
- **Web Sources**: External links with content previews
- **Data Lineage**: Step-by-step information flow tracking

## 🔍 Monitoring and Debugging

### Logging
The system provides comprehensive logging at multiple levels:
- `INFO`: General operation status
- `DEBUG`: Detailed execution information
- `ERROR`: Error conditions and exceptions

### Agent Status Tracking
Each agent reports its status in real-time:
- `IDLE`: Waiting for activation
- `THINKING`: Processing information
- `COMPLETE`: Task completed successfully
- `ERROR`: Error encountered

### Data Lineage
Complete transparency of information flow:
```json
{
    "step": "retrieval",
    "source": "MongoDB Vector Atlas",
    "num_chunks": 5,
    "files": ["hcl-ai-services.pdf", "ai-capabilities.md"]
}
```

## 🚨 Troubleshooting

### Common Issues

#### 1. MongoDB Connection Issues
```bash
# Check connection string format
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database

# Verify network access and credentials
```

#### 2. Vector Index Not Found
```bash
# Ensure vector index is created
python ingestion_mongo.py

# Check index name in MongoDB Atlas
```

#### 3. API Key Errors
```bash
# Verify API keys in .env file
GEMINI_API_KEY=your_actual_key
TAVILY_API_KEY=your_actual_key

# Check API key permissions and quotas
```

#### 4. Embedding Model Download Issues
```python
# If model download fails, try alternative model
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Performance Optimization

#### 1. Chunk Size Tuning
```python
# Adjust chunk size in ingestion_mongo.py
RecursiveCharacterTextSplitter(
    chunk_size=250,  # Increase for longer contexts
    chunk_overlap=50  # Adjust overlap for continuity
)
```

#### 2. Search Parameter Optimization
```env
# Increase search results for complex queries
VECTOR_SEARCH_LIMIT=10
VECTOR_SEARCH_CANDIDATES=100
```

#### 3. Relevance Threshold Adjustment
```env
# Lower threshold for broader results
RELEVANCE_THRESHOLD=6.0
```

## 🛡️ Security Considerations

### API Key Management
- Store API keys in environment variables
- Never commit API keys to version control
- Use secure key rotation practices

### Database Security
- Use MongoDB Atlas security features
- Enable IP whitelisting
- Use strong authentication credentials

### Input Validation
- The system includes input sanitization
- Query length limits are enforced
- Malicious input detection is implemented

## 🔮 Future Enhancements

### Planned Features
- [ ] **Multi-modal Support**: Image and video content processing
- [ ] **Advanced Analytics**: Query pattern analysis and optimization
- [ ] **Custom Agent Creation**: User-defined specialized agents
- [ ] **Integration APIs**: REST API for external system integration
- [ ] **Advanced Caching**: Intelligent result caching for performance
- [ ] **Multi-language Support**: Internationalization capabilities

### Technical Improvements
- [ ] **Async Processing**: Full asynchronous operation support
- [ ] **Distributed Deployment**: Kubernetes deployment configurations
- [ ] **Advanced Monitoring**: Comprehensive metrics and alerting
- [ ] **A/B Testing**: Experimental feature testing framework

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support and questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the configuration documentation

## 🙏 Acknowledgments

- **LangChain Team** for the LangGraph framework
- **MongoDB** for Atlas Vector Search capabilities
- **Google** for Gemini AI models
- **Tavily** for web search API
- **Streamlit** for the beautiful UI framework
- **Sentence Transformers** for embedding models

---

**Built with ❤️ for intelligent information retrieval**