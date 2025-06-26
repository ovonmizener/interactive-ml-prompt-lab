# Interactive ML & Prompting Playground 🧪

A comprehensive full-stack web application for experimenting with machine learning and prompt engineering in real-time. This playground provides an interactive environment to learn, experiment, and build ML models and LLM prompts with guided tutorials and live code snippets.

## 🎯 Project Goals

- **End-to-End ML/DL Flows**: Complete pipeline from data upload → training → explainability
- **LLM Prompt Engineering**: Interactive prompt templates, API calls, and token-level insights
- **Legal-Tech Module**: Semantic search over contracts and legal documents
- **Guided Learning**: Tutorials with live code snippets and real-time experimentation

## 🚀 Features

### 📊 Data Explorer
- Upload CSV/TXT datasets
- Interactive data visualization and statistics
- Data quality analysis and insights
- Correlation matrices and distributions

### 🤖 Model Training
- Multiple ML algorithms (Linear Regression, Random Forest, SVM, Neural Networks)
- Hyperparameter tuning and cross-validation
- Real-time training curves and metrics
- Model comparison and evaluation

### 💬 Prompt Workbench
- Live prompt editor with templates
- LLM API integration (GPT, Claude, etc.)
- Token analysis and visualization
- Prompt variations and history

### ⚖️ Legal Search
- Document upload and processing
- Semantic search with embeddings
- FAISS-based similarity search
- Legal document analytics

## 🛠️ Tech Stack

### Frontend
- **Streamlit**: Interactive web interface with modular pages
- **Plotly**: Rich data visualizations and charts
- **Pandas**: Data manipulation and analysis

### Backend
- **FastAPI**: High-performance API framework
- **Uvicorn**: ASGI server for FastAPI
- **Pydantic**: Data validation and serialization

### Machine Learning
- **Scikit-learn**: Traditional ML algorithms
- **PyTorch/TensorFlow**: Deep learning capabilities
- **SHAP/Captum**: Model explainability
- **FAISS**: Vector similarity search

### LLM & AI
- **OpenAI API**: GPT models integration
- **LangChain**: LLM orchestration
- **Sentence Transformers**: Text embeddings
- **Tiktoken**: Token counting and analysis

### Deployment
- **Docker**: Containerized deployment
- **Multi-service architecture**: FastAPI + Streamlit

## 📁 Project Structure

```
interactive-ml-prompt-lab/
├── app.py                     # FastAPI entrypoint
├── frontend/
│   ├── 1_data_explorer.py      # Data upload and visualization
│   ├── 2_model_train.py        # ML model training interface
│   ├── 3_prompt_workbench.py   # LLM prompt engineering
│   ├── 4_legal_search.py       # Legal document search
│   └── utils.py                # Shared UI components
├── core/
│   ├── ml_pipeline.py          # ML training and evaluation
│   ├── explainability.py       # SHAP and model interpretation
│   ├── llm_utils.py            # LLM integration and prompts
│   └── semantic_search.py      # Document search and embeddings
├── tutorials/                  # Learning materials
│   ├── 01_prompting_101.md     # Prompt engineering guide
│   └── 02_ml_fundamentals.md   # ML concepts tutorial
├── tests/                      # Unit tests
│   └── test_ml_pipeline.py     # ML pipeline tests
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
└── README.md                   # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- OpenAI API key (for LLM features)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/interactive-ml-prompt-lab.git
   cd interactive-ml-prompt-lab
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

5. **Run the application**
   ```bash
   # Start FastAPI backend
   python app.py
   
   # In another terminal, start Streamlit frontend
   streamlit run frontend/1_data_explorer.py
   ```

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t ml-prompt-playground .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 -p 8501:8501 ml-prompt-playground
   ```

3. **Access the application**
   - FastAPI: http://localhost:8000
   - Streamlit: http://localhost:8501
   - API Documentation: http://localhost:8000/docs

## 📖 Usage Guide

### Data Explorer

1. **Upload Data**: Use the file uploader to load CSV or TXT files
2. **Explore Statistics**: View data overview, missing values, and distributions
3. **Visualize Data**: Create histograms, correlation matrices, and charts
4. **Data Quality**: Identify issues and get recommendations

### Model Training

1. **Select Model**: Choose from available algorithms
2. **Configure Parameters**: Set hyperparameters and training options
3. **Feature Selection**: Choose relevant features for training
4. **Train Model**: Start training and monitor progress
5. **Evaluate Results**: View metrics, training curves, and model comparison

### Prompt Workbench

1. **Choose Template**: Select from predefined prompt templates
2. **Customize Variables**: Fill in template parameters
3. **Select LLM**: Choose the language model to use
4. **Generate Response**: Get LLM responses with token analysis
5. **Experiment**: Try different prompts and compare results

### Legal Search

1. **Upload Documents**: Add legal documents (PDF, DOCX, TXT)
2. **Generate Embeddings**: Create vector representations
3. **Search**: Perform semantic search with natural language queries
4. **Analyze Results**: View similarity scores and relevance

## 🧪 Tutorials

### Prompt Engineering 101
Learn the fundamentals of prompt engineering:
- Zero-shot vs. few-shot prompting
- Chain-of-thought reasoning
- Role-based prompting
- Token optimization
- Common patterns and best practices

### Machine Learning Fundamentals
Master core ML concepts:
- Supervised vs. unsupervised learning
- Model evaluation metrics
- Feature engineering
- Hyperparameter tuning
- Overfitting and underfitting

## 🧪 Testing

Run the test suite to ensure everything works correctly:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html
```

## 🔧 Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Model Configuration
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS=1000
TEMPERATURE=0.7

# Database Configuration
DATABASE_URL=sqlite:///./ml_playground.db

# Security
SECRET_KEY=your_secret_key
```

### API Configuration

The FastAPI backend provides RESTful endpoints:

- `GET /api/health` - Health check
- `POST /api/upload-data` - Upload datasets
- `POST /api/train-model` - Train ML models
- `POST /api/generate-llm` - Generate LLM responses
- `POST /api/semantic-search` - Perform semantic search

## 🚀 Future Enhancements

### Planned Features

- **Advanced ML Models**: Deep learning with PyTorch/TensorFlow
- **AutoML**: Automated model selection and hyperparameter tuning
- **Model Deployment**: Export and deploy trained models
- **Collaborative Features**: Share prompts and models
- **Real-time Collaboration**: Multi-user editing
- **Advanced Visualizations**: Interactive dashboards
- **API Integrations**: More LLM providers and services

### Technical Improvements

- **Performance**: Caching and optimization
- **Scalability**: Microservices architecture
- **Security**: Authentication and authorization
- **Monitoring**: Logging and metrics
- **CI/CD**: Automated testing and deployment

### Educational Content

- **Interactive Courses**: Step-by-step learning paths
- **Challenge Mode**: Gamified learning experiences
- **Community Features**: User-generated content
- **Certification**: Learning progress tracking

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Use type hints
- Write docstrings for functions
- Add comments for complex logic

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance API
- [OpenAI](https://openai.com/) for LLM capabilities
- [Hugging Face](https://huggingface.co/) for transformers and models
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/interactive-ml-prompt-lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/interactive-ml-prompt-lab/discussions)
- **Email**: support@mlplayground.com

## 📊 Project Status

- ✅ Core ML pipeline
- ✅ Basic prompt engineering
- ✅ Data visualization
- ✅ Legal search module
- 🔄 Advanced ML models
- 🔄 AutoML features
- 🔄 Model deployment
- 🔄 Collaborative features

---

**Happy experimenting! 🎉**

*Built with ❤️ for the ML and AI community*
