# Learning Paths & Examples for Enterprise AI

A curated collection of tutorials, playbooks, example repositories, and learning resources for mastering enterprise AI tools and workflows.

---

## GPU-Accelerated Data Processing

### NVIDIA DGX Spark Playbooks
**Repository:** https://github.com/nvidia/dgx-spark-playbooks
**Platform:** NVIDIA DGX Spark (Blackwell architecture)
**Level:** Beginner to Advanced

**What It Covers:**
Step-by-step playbooks for AI/ML workloads on NVIDIA hardware, designed for DGX Spark desktop supercomputer.

**Key Learning Areas:**
- **GPU-Accelerated Data Science:**
  - RAPIDS: cuDF (pandas), cuML (scikit-learn)
  - Zero-code-change GPU acceleration
  - 250 MB datasets processed in seconds
  - Tens of millions of records with cuDF pandas

- **Machine Learning Workflows:**
  - UMAP, HDBSCAN clustering on GPU
  - Model training, fine-tuning (LLMs, vision models)
  - High-performance numerical computing with JAX

- **Production AI Applications:**
  - AI code assistants, chatbots
  - Video search, summarization
  - Image generation
  - LLM inference optimization

**Why Use This:**
- Official NVIDIA resource, production-grade examples
- 10-100x speedups demonstrated
- Complete workflow: data loading → preprocessing → training → inference
- Blackwell architecture optimization patterns
- Foundation for CUDA-X Data Science stack

**Prerequisites:**
- Basic Python
- Familiarity with pandas, scikit-learn
- NVIDIA GPU access (DGX Spark ideal, adaptable to other NVIDIA hardware)

**Related Tools:**
- RAPIDS (cuDF, cuML, cuGraph)
- CUDA-X libraries
- JAX for numerical computing
- TensorRT for inference optimization

---

## ML Systems Engineering

### Machine Learning Systems Book
**Website:** https://www.mlsysbook.ai/
**Platform:** Harvard CS249r / MIT Press (2026)
**Level:** Intermediate to Advanced

**What It Covers:**
Comprehensive textbook on building ML systems from a holistic engineering perspective—not just models, but the full systems that make them work in production.

**Key Learning Areas:**
- **Systems Foundations:**
  - Introduction to ML systems architecture
  - Deep learning primers and DNN architectures
  - Understanding the full ML engineering context

- **Design Principles:**
  - AI workflows and data engineering
  - Frameworks and training methodologies
  - End-to-end system design

- **Performance Engineering:**
  - Efficient AI and model optimization
  - Hardware acceleration strategies
  - Benchmarking and performance analysis

- **Robust Deployment:**
  - ML operations (MLOps)
  - On-device learning and edge deployment
  - Security, privacy, and reliability

- **Trustworthy Systems:**
  - Responsible AI practices
  - Sustainability considerations
  - Societal applications and impact

**Why Use This:**
- Written by Vijay Janapa Reddi (Harvard University)
- Bridges gap between "training models" and "building systems that work in production"
- Open educational resource (CC-BY-NC-SA 4.0)
- Student-contributed content from Harvard CS249r
- Publishing via MIT Press (2026)
- Focus on "seeing the forest" of ML systems engineering

**Prerequisites:**
- Basic Python and ML understanding
- Familiarity with deep learning concepts
- Interest in systems engineering

**Related Tools:**
- MLflow, Comet for MLOps
- vLLM, Ray for distributed inference
- NVIDIA NIM for deployment
- Prometheus, Grafana for monitoring

---

### Understanding Deep Learning
**Website:** https://udlbook.github.io/udlbook/
**Platform:** MIT Press / University of Bath
**Level:** Intermediate to Advanced

**What It Covers:**
Comprehensive deep learning textbook by Simon J.D. Prince covering fundamentals through cutting-edge topics like transformers and diffusion models.

**Key Learning Areas:**
- **Foundations:**
  - Supervised learning fundamentals
  - Activation functions and backpropagation
  - Loss functions and optimization
  - Batch normalization

- **Neural Network Architectures:**
  - Convolutional networks (CNNs)
  - Recurrent networks and sequence models
  - Transformers and attention mechanisms
  - Graph neural networks

- **Generative Models:**
  - Autoencoders and VAEs
  - Normalizing flows
  - Diffusion models
  - GANs

- **Advanced Topics:**
  - Reinforcement learning and Q-learning
  - Double descent and grokking phenomena
  - Lottery tickets hypothesis
  - Ethics and deep learning (Chapter 21)

**Why Use This:**
- Author: Simon J.D. Prince (University of Bath, formerly Anthropic, Borealis AI)
- Published by MIT Press (2024)
- Free PDF download available
- Python notebooks for students
- 21+ chapters progressing in complexity
- Balances theory and practical implementation
- Covers unexpected phenomena often missed in other texts

**Prerequisites:**
- Basic applied mathematics
- Linear algebra and calculus
- Some Python programming experience

**Related Tools:**
- PyTorch, TensorFlow for implementation
- Hugging Face Transformers
- JAX for research implementations

---

## AI Engineering & Production Systems

### AI Engineering by Chip Huyen
**Repository:** https://github.com/chiphuyen/aie-book
**Platform:** Foundation Models & LLM Applications
**Level:** Intermediate to Advanced

**What It Covers:**
Comprehensive guide to adapting foundation models (LLMs, multimodal) for real-world production applications. O'Reilly Media, 2025.

**Key Learning Areas:**
- **Foundation Model Adaptation:**
  - Application viability assessment
  - Model selection, fine-tuning decisions
  - Parameter-efficient fine-tuning
  - Data quality validation

- **Production LLM Techniques:**
  - Prompt engineering best practices
  - RAG strategies
  - Hallucination detection, mitigation
  - Agent design, evaluation frameworks

- **Performance Optimization:**
  - Speed optimization for production
  - Cost reduction strategies
  - Security considerations for AI apps
  - Continuous improvement feedback loops

**Why Use This:**
- Chip Huyen, author of "Designing Machine Learning Systems"
- Fundamental engineering principles over tool-specific knowledge
- 11.2k+ GitHub stars, active community
- Extensively peer-reviewed by industry experts
- Practical case studies from real implementations
- Companion materials: study guides, prompt examples
- Bridges theory and production-ready systems

**Prerequisites:**
- Foundational technical knowledge (AI/ML engineers, data scientists)
- Basic ML concepts
- LLMs/foundation models helpful but not required

**Related Tools:**
- LangChain/LangGraph for agent orchestration
- RAG frameworks (LlamaIndex, Haystack)
- LLM providers (OpenAI, Anthropic, Cohere)
- Evaluation frameworks (Ragas, DeepEval)
- Prompt management tools (Portkey, Langfuse)

---

## LLM Pre-training & Fine-tuning

### The Smol Training Playbook
**Website:** https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
**Platform:** Hugging Face
**Level:** Intermediate to Advanced

**What It Covers:**
Radically transparent 200+ page guide from Hugging Face detailing the full pipeline of building SmolLM3—a 3B-parameter model trained on 11 trillion tokens. Covers pre-training, post-training, and infrastructure.

**Key Learning Areas:**
- **Before Training:**
  - Should you even train your own model?
  - When to use existing models (Llama, Qwen, Gemma)
  - Project scoping and timeline planning

- **Ablation-First Methodology:**
  - Running hundreds of small-scale experiments
  - "Derisking" every architectural change
  - "Intuition is cheap, but GPUs are expensive"

- **Data Curation:**
  - Why data quality drives performance more than architecture
  - Training data mixture strategies
  - Quality vs quantity tradeoffs

- **Architecture Decisions:**
  - Dense vs MoE vs Hybrid considerations
  - Long context with NoPE (Rotary Position Encoding)
  - Tokenizer selection (Llama3's 128k vocabulary)
  - Intra-document masking for generalization

- **Infrastructure:**
  - GPU orchestration and distributed training
  - Edge deployment considerations
  - Memory constraints for mobile deployment

**Why Use This:**
- Official Hugging Face pre-training team resource (Oct 2025)
- 2.41k+ likes on Hugging Face
- Follows FineWeb, Ultra Scale Playbook, Evaluation Guidebook series
- Documents real failures and dead ends, not just successes
- SmolLM3 trained in ~3 months timeline
- Practical guide for teams considering LLM pre-training
- Authors include Loubna Ben Allal, Lewis Tunstall, Colin Raffel, Thomas Wolf

**Prerequisites:**
- Strong ML/deep learning background
- Familiarity with transformer architectures
- Understanding of distributed training concepts
- Experience with PyTorch and Hugging Face ecosystem

**Related Tools:**
- Hugging Face Transformers, Datasets
- NeMo Curator for data preparation
- Unsloth for efficient fine-tuning
- vLLM for inference
- W&B, Comet for experiment tracking

---

## Prompt Engineering & LLM Techniques

### Prompt Engineering Guide by DAIR.AI
**Repository:** https://github.com/dair-ai/Prompt-Engineering-Guide
**Platform:** Large Language Models (All Major Providers)
**Level:** Beginner to Advanced

**What It Covers:**
Community-driven guide to prompt engineering for LLMs, fundamentals through cutting-edge research. Active website: [promptingguide.ai](https://www.promptingguide.ai/).

**Key Learning Areas:**
- **Fundamentals:**
  - LLM settings, parameters
  - Prompt elements, design principles
  - Zero-shot, few-shot prompting
  - Context engineering

- **Advanced Techniques:**
  - Chain-of-thought (CoT) prompting
  - Tree of thoughts, reasoning strategies
  - ReAct prompting for agents
  - RAG
  - Self-consistency, ensemble methods

- **Practical Applications:**
  - Function calling, tool use
  - Code generation, debugging
  - Data synthesis, augmentation
  - Model-specific guides (ChatGPT, GPT-4, Llama, Mistral, Gemini)

- **Safety & Ethics:**
  - Adversarial prompting, jailbreaking
  - Factuality, hallucination mitigation
  - Bias detection, reduction
  - Prompt injection defenses

**Why Use This:**
- 66.3k+ GitHub stars, 193+ contributors
- 3+ million learners worldwide
- Multi-format: guides, notebooks, video lectures, research papers
- Prompt Hub with curated prompts
- Translated into 13+ languages
- Active community: Discord, Twitter, YouTube, newsletter
- Free, constantly updated with latest research
- Academy courses for structured learning

**Prerequisites:**
- Basic AI/ML understanding
- No coding for core concepts
- Python helpful for notebooks

**Estimated Time:**
- Core concepts: 2-4 hours
- Full guide: 10-15 hours
- Mastery: Ongoing

**Community:**
- [Discord](https://discord.gg/dair-ai) - Community support
- [Website](https://www.promptingguide.ai) - Interactive platform
- [YouTube](https://www.youtube.com/channel/UCyna_OxOWL7IEuOwb7WhmxQ) - Video tutorials
- [Newsletter](https://newsletter.dair.ai) - Updates, research

**Related Tools:**
- LLM providers (OpenAI, Anthropic, Cohere, Google)
- Prompt management (Portkey, Langfuse)
- LangChain/LangGraph for orchestration
- Agent frameworks (Vercel AI SDK, LlamaIndex)
- Evaluation tools (Ragas, DeepEval)

---

## Agentic Workflows & Multi-Agent Systems

### LangChain Academy
**Website:** https://academy.langchain.com/collections
**Platform:** LangChain / LangGraph
**Level:** Beginner to Advanced

**What It Covers:**
Official free courses from LangChain teaching agent development, LangGraph workflows, and production AI applications.

**Key Learning Areas:**
- **Introduction to LangGraph:**
  - State, Nodes, Edges, and Memory fundamentals
  - Building email workflows as hands-on projects
  - LangGraph architecture and control flow

- **Ambient Agents:**
  - Building autonomous email assistants from scratch
  - LangSmith integration for evaluation
  - Real-world agent deployment patterns

- **Deep Agents:**
  - Complex, long-running task handling
  - Advanced agent characteristics and patterns
  - Production-grade agent implementation

- **Agentic RAG (via Coursera/IBM):**
  - Self-improving agents with Reflection, Reflexion, ReAct
  - Query routing and retrieval-enhanced reasoning
  - Memory, iteration, and conditional logic

**Why Use This:**
- Official courses from LangChain team
- Free access to core curriculum
- Hands-on projects with real applications
- LangGraph is MIT-licensed and widely adopted
- Complementary courses on Coursera (IBM, DeepLearning.AI)
- 100+ related courses available across platforms

**Prerequisites:**
- Python programming experience
- Basic understanding of LLMs and APIs
- Familiarity with async programming helpful

**Related Tools:**
- LangChain/LangGraph for orchestration
- LangSmith for observability and evaluation
- Vercel AI SDK for frontends
- RAG frameworks (LlamaIndex, Haystack)

---

### Agentic Design Patterns
**Repository:** https://github.com/sarwarbeing-ai/Agentic_Design_Patterns
**Platform:** Python / Jupyter Notebooks
**Level:** Intermediate to Advanced

**What It Covers:**
Hands-on guide to building intelligent agentic systems by Antonio Gulli, with practical Jupyter notebooks demonstrating design patterns.

**Key Learning Areas:**
- **Agentic Architecture Patterns:**
  - Autonomous agent design principles
  - Semi-autonomous system construction
  - Agent orchestration strategies

- **Practical Implementation:**
  - Jupyter notebooks with working examples
  - PDF reference guide included
  - Pattern-based approach to agent development

**Why Use This:**
- By Antonio Gulli (known AI/ML author)
- 1.1k+ GitHub stars, growing community interest
- 100% Jupyter notebook codebase
- Practical, hands-on approach
- Companion PDF resource
- Focus on reusable design patterns

**Prerequisites:**
- Python programming experience
- Basic understanding of LLMs and agents
- Familiarity with Jupyter notebooks

**Related Tools:**
- LangChain/LangGraph for orchestration
- Vercel AI SDK for frontends
- Agent frameworks (CrewAI, AutoGPT)

---

### Agent2Agent (A2A) Protocol Samples
**Repository:** https://github.com/a2aproject/a2a-samples
**Platform:** Agent2Agent (A2A) Protocol
**Level:** Intermediate to Advanced

**What It Covers:**
Code samples and demos demonstrating the Agent2Agent (A2A) Protocol—an open standard enabling communication and interoperability between independent AI agent systems.

**Key Learning Areas:**
- **Agent Interoperability:**
  - Agent-to-agent communication patterns
  - Capability discovery using Agent Cards (JSON format)
  - Cross-framework agent collaboration (different vendors, languages)

- **Task Management & Coordination:**
  - Task lifecycle management (immediate or long-running)
  - Context sharing between agents
  - Message passing for instructions, replies, artifacts

- **Production Implementation:**
  - Secure agent communication over HTTPS
  - JSON-RPC 2.0 data exchange format
  - Authentication and authorization (OpenAPI schemes)
  - Security best practices (prompt injection defense, untrusted input handling)

- **Integration Examples:**
  - Jupyter notebooks with interactive tutorials
  - Firebase Studio templates
  - Platform extensions and connectors
  - Multi-agent purchasing concierge demos

**Why Use This:**
- Official samples from Linux Foundation A2A Project
- Production-grade security considerations
- Backed by 50+ partners: Google, Microsoft, Atlassian, Box, Cohere, IBM, Intuit, LangChain, MongoDB, PayPal, Salesforce, SAP, ServiceNow, Workday
- Enables vendor-neutral agent ecosystems
- Foundation for enterprise multi-agent applications
- Active governance under Linux Foundation (2025)

**Prerequisites:**
- Python development experience
- Understanding of distributed systems or multi-agent concepts
- Familiarity with API protocols (HTTP, JSON-RPC)
- Basic knowledge of agent-based systems

**Related Tools:**
- [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/)
- [A2A Python SDK](https://github.com/a2aproject/a2a-python-sdk)
- LangChain/LangGraph for orchestration
- Vercel AI SDK for agent frontends
- Agent frameworks (AutoGPT, CrewAI, Microsoft Semantic Kernel)

**Important Notes:**
- Strong emphasis on security: prompt injection attacks, untrusted input validation
- Designed for production systems, not experimental projects
- Requires understanding of security implications in multi-agent environments

---

## Version Information
- **Last Updated:** 2025-11-25
- **Total Resources:** 11
- **Categories Covered:** 6 (GPU-Accelerated Data Processing, ML Systems Engineering, AI Engineering & Production Systems, LLM Pre-training & Fine-tuning, Prompt Engineering & LLM Techniques, Agentic Workflows & Multi-Agent Systems)

---

## Planned Additions

Future learning resources to be added:

### High Priority
- [x] LangChain/LangGraph official examples and tutorials ✅ Added LangChain Academy
- [ ] vLLM production deployment guides
- [ ] Ray Train/Serve tutorials for distributed ML
- [ ] Unstructured.io ETL pipeline examples
- [ ] Presidio PII detection integration patterns

### Medium Priority
- [ ] RAPIDS end-to-end data science workflows
- [ ] NeMo Curator dataset preparation tutorials
- [ ] Vercel AI SDK application examples
- [ ] Multi-agent system architectures

### Community Requests
- [ ] Add your suggestions via issues or PRs

---

## Additional Learning Resources

### Official Documentation
Refer to enterprise-ai-tools.md for links to official documentation for each tool.

### Community Forums
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com)
- [LangChain Discord](https://discord.gg/langchain)
- [Ray Slack](https://ray.io/community)
- [RAPIDS Community](https://rapids.ai/community)

### Conferences & Events
- NVIDIA GTC (GPU Technology Conference)
- Ray Summit
- LangChain Webinars
- MLOps Community Events
