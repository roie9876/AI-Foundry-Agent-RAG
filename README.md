# AI Foundry Agent Creator

A Streamlit application for creating Azure AI Foundry agents with Azure AI Search integration. This tool allows you to easily create RAG (Retrieval-Augmented Generation) agents through a user-friendly interface.

## Features

- 🔐 Azure CLI authentication
- 📊 Browse and select AI Foundry projects
- 🔍 Configure Azure AI Search connections
- 🤖 Create agents with custom instructions
- 🎛️ Configure search parameters (query type, top-k results)

## Prerequisites

- Azure subscription
- Azure CLI installed and configured
- Access to Azure AI Foundry
- Azure AI Search service (optional, for RAG capabilities)

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/ai-foundry-agent-creator.git
    cd ai-foundry-agent-creator
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Authentication

This app uses Azure CLI authentication. Before running the app:

```bash
az login
```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```

2. The app will open in your browser at `http://localhost:8501`

3. Follow the steps in the UI:
   - Select your AI Foundry project
   - Configure your model deployment
   - Set up Azure AI Search connection
   - Define agent instructions
   - Create the agent

## Configuration Options

### Query Types
- **SIMPLE**: Basic keyword search
- **SEMANTIC**: Semantic search with language understanding
- **VECTOR**: Vector similarity search
- **VECTOR_SIMPLE_HYBRID**: Combination of vector and keyword search
- **VECTOR_SEMANTIC_HYBRID**: Combination of vector and semantic search

### Agent Instructions
Customize your agent's behavior by providing specific instructions. The default template ensures the agent:
- Always searches for information before answering
- Bases responses only on search results
- Cites sources appropriately

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit  │────▶│ Azure CLI    │────▶│ AI Foundry  │
│     UI      │     │ Credential   │     │   APIs      │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Azure Search │
                    │  Connection  │
                    └──────────────┘
```

## Troubleshooting

### Authentication Issues
- Ensure you're logged in: `az account show`
- Check your subscription: `az account list`
- Set correct subscription: `az account set --subscription <subscription-id>`

### Permission Issues
Ensure your account has the following roles:
- Contributor access to the AI Foundry project
- Cognitive Services Contributor
- Search Service Contributor (if using Azure AI Search)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This is a community project and is not officially supported by Microsoft.