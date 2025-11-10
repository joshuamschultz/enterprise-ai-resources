# 🚀 Awesome Enterprise AI Tools

> Production-grade tools for building AI systems at enterprise scale

**Last Updated:** 2025-11-10

---

## 📖 Table of Contents

- [Data Validation & Type Safety](#-data-validation--type-safety)
- [NLP & Text Processing](#-nlp--text-processing)
- [Document Ingestion & ETL](#-document-ingestion--etl)
- [Data Curation](#-data-curation)
- [Distributed Computing & Processing](#-distributed-computing--processing)
- [Data Privacy & Security](#-data-privacy--security)
- [Vector Databases](#-vector-databases)
- [LLM Providers](#-llm-providers)
- [LLM Inference & Serving](#-llm-inference--serving)
- [Agentic Workflows & Orchestration](#-agentic-workflows--orchestration)
- [RAG Evaluation & Testing](#-rag-evaluation--testing)
- [Observability & Monitoring](#-observability--monitoring)
- [Cloud Platforms](#-cloud-platforms)
- [Application Development](#-application-development)

---

## 🛡️ Data Validation & Type Safety

### Pydantic
**🔗 Links:** [Website](https://docs.pydantic.dev) · [GitHub](https://github.com/pydantic/pydantic) · [Pydantic AI](https://ai.pydantic.dev)

**⚡ What:** Type-safe data validation for Python with AI agent framework

**🎯 Use When:**
- Need type safety for LLM outputs with JSON schema enforcement
- Building production AI apps requiring data integrity and consistency
- Validating API inputs/outputs across enterprise services
- Creating structured outputs from unstructured LLM responses

**💪 Why:**
- 360M+ downloads/month, used by all FAANG companies
- Model-agnostic: OpenAI, Anthropic, Gemini, Cohere, AWS, Azure, GCP
- MIT licensed for commercial/enterprise use
- Reduces runtime errors, streamlines debugging

**📊 License:** MIT | **Support:** Community + Enterprise consulting

---

## 🔤 NLP & Text Processing

### spaCy
**🔗 Links:** [Website](https://spacy.io) · [GitHub](https://github.com/explosion/spaCy)

**⚡ What:** Industrial-strength NLP with production-ready pipelines

**🎯 Use When:**
- Named entity recognition, POS tagging, dependency parsing at scale
- Preprocessing text for ML/LLM applications
- Building custom NLP pipelines for domain-specific text
- Processing large volumes (millions of documents) efficiently

**💪 Why:**
- C-like speed (Cython core), handles massive text volumes
- State-of-the-art neural models, 100+ pre-trained pipelines
- Latest release: Nov 2025, actively maintained
- Built specifically for production use, not research

**📊 License:** MIT | **Support:** Community + Commercial custom pipelines

---

## 📄 Document Ingestion & ETL

### Unstructured.io
**🔗 Links:** [Website](https://unstructured.io) · [Docs](https://docs.unstructured.io) · [GitHub](https://github.com/Unstructured-IO/unstructured)

**⚡ What:** ETL platform transforming unstructured docs → AI-ready data

**🎯 Use When:**
- Processing 25+ file types (PDFs, Word, HTML, emails, PowerPoints, images)
- Building RAG applications requiring diverse document ingestion
- GDPR/HIPAA/SOC 2 compliance needed for document processing
- High-volume automated pipelines with 50+ source/destination connectors

**💪 Why:**
- 82% Fortune 1000 adoption, SOC 2 Type 2 / HIPAA / GDPR ready
- Continuous ingestion with flexible chunking and embedding strategies
- Pythonic API + managed platform options

**📊 License:** Apache 2.0 | **Support:** Plus + Enterprise tiers

---

### Docling
**🔗 Links:** [Website](https://www.docling.ai) · [Docs](https://docling.ai/docs) · [GitHub](https://github.com/DS4SD/docling)

**⚡ What:** MIT-licensed document conversion preserving layout & structure

**🎯 Use When:**
- High-accuracy parsing for business intelligence
- Preserving complex elements: tables, equations, code blocks
- On-premise deployments with resource constraints
- Open-source alternative to commercial document AI

**💪 Why:**
- Powered by DocLayNet (layout) + TableFormer (tables) AI models
- 10k GitHub stars in <1 month, #1 trending Nov 2024
- Runs efficiently on commodity hardware
- Red Hat RHEL AI support, IBM Granite integration

**📊 License:** MIT | **Support:** IBM + Red Hat RHEL AI

---

## 📦 Data Curation

### NVIDIA NeMo Curator
**🔗 Links:** [Website](https://developer.nvidia.com/nemo-curator) · [GitHub](https://github.com/NVIDIA-NeMo/Curator)

**⚡ What:** GPU-accelerated data curation for trillion-token datasets

**🎯 Use When:**
- Pre-training data prep for foundation models (LLMs, VLMs, multimodal)
- Large-scale dataset quality improvement and deduplication (100+ PB)
- Synthetic data generation and filtering pipelines
- Processing speed critical (17x faster vs CPU)

**💪 Why:**
- Complete pipeline: download → extract → clean → dedupe → blend
- Pythonic APIs using RAPIDS (cuDF, cuGraph, cuML)
- Part of NVIDIA NeMo suite for full AI lifecycle

**📊 License:** Apache 2.0 | **Support:** NVIDIA AI Enterprise

---

## 🔧 Distributed Computing & Processing

### NVIDIA RAPIDS
**🔗 Links:** [Website](https://rapids.ai) · [Docs](https://docs.rapids.ai) · [GitHub](https://github.com/rapidsai)

**⚡ What:** GPU-accelerated pandas/scikit-learn with zero code changes

**🎯 Use When:**
- Large-scale data preprocessing and feature engineering
- Real-time analytics requiring sub-second response
- Cost optimization: replace CPU clusters with smaller GPU clusters
- EDA on billion-row datasets, graph analytics at scale

**💪 Why:**
- 50x faster end-to-end data science workflows
- Zero code change: cuDF (pandas), cuML (scikit-learn), cuGraph (NetworkX)
- PayPal 70% cost reduction, CapitalOne 100x faster training
- Spark acceleration via RAPIDS Accelerator

**📊 License:** Apache 2.0 | **Support:** NVIDIA AI Enterprise

---

### Ray
**🔗 Links:** [Website](https://www.ray.io) · [Docs](https://docs.ray.io) · [GitHub](https://github.com/ray-project/ray)

**⚡ What:** Unified framework for scaling AI/ML from laptop → cluster

**🎯 Use When:**
- Distributed training of foundation models and neural networks
- Multi-model serving with dynamic batching and autoscaling
- Hyperparameter optimization (1000s of trials)
- Any Python workload needing horizontal scaling

**💪 Why:**
- Powers OpenAI ChatGPT infrastructure
- Unified APIs: Ray Data, Train, Serve, Tune, RLlib
- Scales with minimal code changes (often 1 line)
- Azure support: fully managed first-party service (Nov 2025)
- All accelerators: NVIDIA, AMD, Intel, Google TPUs, CPUs

**📊 License:** Apache 2.0 | **Support:** Anyscale (managed on Azure/AWS)

---

## 🔐 Data Privacy & Security

### Microsoft Presidio
**🔗 Links:** [Website](https://microsoft.github.io/presidio) · [GitHub](https://github.com/microsoft/presidio)

**⚡ What:** PII detection, redaction, and anonymization for text/images

**🎯 Use When:**
- Protecting data before LLM API calls (prevent data leakage)
- GDPR/HIPAA/CCPA compliance for data anonymization
- RAG systems requiring PII removal from documents
- Real-time data masking in chatbots and agents

**💪 Why:**
- Two-engine architecture: Analyzer (detect) + Anonymizer (redact/mask/encrypt)
- Context-aware detection: NER, regex, checksums, multi-language
- Multiple anonymization strategies: redact, mask, hash, encrypt, synthetic
- LangGraph integration for PII-aware workflows

**⚠️ Note:** Automated detection, cannot guarantee 100% PII identification

**📊 License:** MIT | **Support:** Community + Microsoft backing

---

## 💾 Vector Databases

### Milvus
**🔗 Links:** [Website](https://milvus.io) · [Docs](https://milvus.io/docs) · [GitHub](https://github.com/milvus-io/milvus)

**⚡ What:** Open-source vector database built for GenAI at massive scale

**🎯 Use When:**
- Similarity search on billions of high-dimensional vectors
- RAG applications requiring fast, scalable vector search
- Mission-critical AI apps (NVIDIA, Meta, Salesforce use it)
- Need flexible deployment: Lite (prototyping) → Standalone → Distributed

**💪 Why:**
- 5,000+ enterprise users, 35,000 GitHub stars
- 72% less memory, 4x faster queries vs Elasticsearch (Milvus 2.6, June 2025)
- Enterprise security: RBAC, TLS encryption, user authentication
- Unified API across all deployment models

**📊 License:** Apache 2.0 | **Support:** Zilliz Cloud (managed service from $99/mo)

---

### PostgreSQL + pgvector
**🔗 Links:** [PostgreSQL](https://www.postgresql.org) · [pgvector](https://github.com/pgvector/pgvector)

**⚡ What:** PostgreSQL extension for high-performance vector similarity search

**🎯 Use When:**
- <100M vectors (best TCO at this scale)
- Unified relational + vector workloads (no separate DB needed)
- Need PostgreSQL ecosystem: security, backup, replication
- Cloud-managed options: AWS RDS/Aurora, GCP AlloyDB, Azure

**💪 Why:**
- 9x faster queries (pgvector 0.8.0 breakthrough)
- Supports 2,000-dim vectors (standard), 4,000-dim (halfvec)
- Binary quantization for compressed storage
- Google AlloyDB adds ScaNN index (12 years Google Research)

**📊 License:** PostgreSQL License | **Support:** Major cloud providers

---

## 🤖 LLM Providers

### OpenAI
**🔗 Links:** [Website](https://openai.com) · [API Docs](https://platform.openai.com/docs) · [Pricing](https://openai.com/api/pricing/)

**⚡ What:** GPT-5 and production LLM APIs for enterprise

**🎯 Use When:**
- Need cutting-edge reasoning and coding capabilities
- Building production apps with strict SLAs
- Require enterprise security (GDPR, CCPA, SOC 2 Type 2)
- Multi-model routing for cost/quality optimization

**💪 Why:**
- GPT-5: 80% fewer hallucinations, 50% cost reduction vs GPT-4
- 1.6% hallucination rate (healthcare), 74.9% SWE-bench accuracy
- Batch API: 50% discount for 24hr processing
- 500+ person enterprise sales team

**💰 Pricing:** $5-$25/M tokens (GPT-4.1 Sonnet), $20-$80/M (Opus)

**📊 License:** Proprietary | **Support:** Enterprise plans available

---

### Anthropic Claude
**🔗 Links:** [Website](https://www.claude.com) · [API](https://www.anthropic.com/api) · [Pricing](https://www.claude.com/pricing)

**⚡ What:** Claude 4 Sonnet/Opus with 1M token context window

**🎯 Use When:**
- Processing long documents (750k words, 75k lines of code)
- Need constitutional AI for safer, more aligned outputs
- Financial services (AIG: 5x faster underwriting, 75%→90% accuracy)
- AWS Bedrock or Google Vertex AI integration

**💪 Why:**
- 1M token context (API), 200K (web)
- Claude Code bundled in Team/Enterprise plans (Aug 2025)
- Enterprise: self-serve seat management, usage analytics, spend controls
- Available: AWS Marketplace, Bedrock, Vertex AI

**💰 Pricing:** $5-$25/M (Sonnet 4.1), $20-$80/M (Opus 4.1) + thinking tokens

**📊 License:** Proprietary | **Support:** Enterprise + AWS Marketplace

---

### Cohere
**🔗 Links:** [Website](https://cohere.com) · [API](https://cohere.com/embed) · [Pricing](https://cohere.com/pricing)

**⚡ What:** Enterprise AI with Embed 4 multimodal embeddings

**🎯 Use When:**
- Building RAG/search for regulated industries (finance, healthcare, manufacturing)
- Need multilingual support (100+ languages)
- Processing long documents (128k tokens = 200 pages)
- Cost optimization with 96% embedding compression

**💪 Why:**
- Embed 4: multimodal (text+images), 128k context window
- Optimized for agentic search and retrieval
- Available: Cohere Platform, AWS SageMaker, Azure AI Foundry
- Strong enterprise security for regulated sectors

**💰 Pricing:** $3/M input, $15/M output (Grok 3), embeddings vary

**📊 License:** Proprietary | **Support:** Enterprise support available

---

### xAI Grok
**🔗 Links:** [Website](https://x.ai) · [API](https://x.ai/api) · [Docs](https://docs.x.ai)

**⚡ What:** Grok 4 with real-time search and native tool use

**🎯 Use When:**
- Need real-time information from web/X integration
- Enterprise data extraction, programming, text summarization
- Frontier performance with exceptional token efficiency
- Cost-sensitive workloads (Grok 4 Fast)

**💪 Why:**
- Grok 4 Fast: Sept 2025 release, frontier-level performance
- Native tool use and real-time search built-in
- "Most intelligent model in the world" (xAI claim)
- Enterprise arrangements available with custom quotas

**💰 Pricing:** $3/M input, $15/M output (Grok 3) · SuperGrok Heavy $300/mo

**📊 License:** Proprietary | **Support:** Enterprise custom arrangements

---

### Hugging Face
**🔗 Links:** [Website](https://huggingface.co) · [Inference API](https://huggingface.co/inference-api) · [Endpoints](https://endpoints.huggingface.co)

**⚡ What:** 100k+ open models with unified inference infrastructure

**🎯 Use When:**
- Need access to open-source models (Llama, Mistral, Falcon, etc.)
- Building multi-model apps with consistent API
- Want deployment flexibility: serverless → dedicated endpoints
- Enterprise Hub for centralized billing and governance

**💪 Why:**
- Inference Providers: unified API to world-class inference infrastructure
- Auto-scaling: scales up/down with traffic to save costs
- Supports vLLM, TGI, SGLang, TEI, custom containers
- All data transfers SSL encrypted, no third-party access

**💰 Pricing:** Free tier → PRO → Enterprise Hub (centralized billing)

**📊 License:** Varies by model | **Support:** api-enterprise@huggingface.co

---

## ⚡ LLM Inference & Serving

### vLLM
**🔗 Links:** [Website](https://docs.vllm.ai) · [GitHub](https://github.com/vllm-project/vllm)

**⚡ What:** High-throughput, memory-efficient LLM inference engine

**🎯 Use When:**
- Production serving with strict SLAs (latency, throughput)
- Cost optimization: maximize GPU utilization
- Multi-tenant serving with isolated workloads
- Distributed inference across clusters (disaggregated prefill/decode)

**💪 Why:**
- 3-10x lower latency, 2-5x higher throughput vs standard serving
- PagedAttention algorithm eliminates memory fragmentation
- Production Stack (Jan 2025): prefix-aware routing, KV-cache sharing, autoscaling
- llm-d: Kubernetes-native (Red Hat, Google, IBM, NVIDIA, CoreWeave)
- Support for 100+ model architectures, all accelerators

**📊 License:** Apache 2.0 | **Support:** Red Hat OpenShift AI + llm-d consortium

---

### NVIDIA NIM
**🔗 Links:** [Website](https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/) · [Developer](https://developer.nvidia.com/nim)

**⚡ What:** Optimized inference microservices for AI models

**🎯 Use When:**
- Deploying AI models across cloud, data center, workstation
- Need 5-minute deployment with standard APIs
- Kubernetes scaling and enterprise support required
- Building agentic AI with guardrails (NeMo Guardrails)

**💪 Why:**
- Zero-configuration deployment, cloud-native microservices
- Deploy anywhere: NVIDIA-accelerated infrastructure (cloud, DC, workstation)
- Enterprise support: continuous validation, feature branches, NVIDIA experts
- Native integration: Azure AI Foundry, Red Hat OpenShift AI

**📊 License:** Part of NVIDIA AI Enterprise | **Support:** NVIDIA AI Enterprise

---

## 🔗 Agentic Workflows & Orchestration

### LangChain Ecosystem
**🔗 Links:** [LangChain](https://www.langchain.com) · [LangGraph](https://www.langchain.com/langgraph) · [LangSmith](https://www.langchain.com/langsmith)

**⚡ What:** Production framework for LLM apps and agents

**🎯 Use When:**
- Building RAG applications with complex retrieval logic
- Multi-agent systems requiring orchestration and state sharing
- Need provider flexibility (swap OpenAI ↔ Anthropic ↔ open-source)
- Enterprise observability, evaluation, CI/CD required
- Long-running stateful agents (hours/days/weeks)

**💪 Why:**

**LangChain:** 70M+ downloads/mo, 700+ integrations (LLMs, vectors, APIs, tools)

**LangGraph:** 400+ companies deployed agents to prod (2025 beta), stateful multi-agent workflows, human-in-the-loop

**LangSmith:** End-to-end tracing, debugging, monitoring; AWS Marketplace (2025); Cloud, Hybrid, Self-Hosted deployment

**📊 License:** MIT | **Support:** LangSmith Plus + Enterprise

---

## 🧪 RAG Evaluation & Testing

### Ragas
**🔗 Links:** [Website](https://www.ragas.io) · [Docs](https://docs.ragas.io) · [GitHub](https://github.com/explodinggradients/ragas)

**⚡ What:** Reference-free RAG evaluation with LLM-as-judge

**🎯 Use When:**
- Evaluating RAG accuracy before production deployment
- Continuous monitoring of production RAG (A/B tests, dashboards)
- Identifying weak points in retrieval or generation stages
- Optimizing retrieval strategies (chunk size, embeddings, reranking)
- Compliance: ensuring factual accuracy and relevance

**💪 Why:**
- **4 Core Metrics:** Faithfulness, Answer Relevancy, Context Precision, Context Recall
- No ground truth annotations needed (reference-free)
- Integrates with LangChain, LlamaIndex, observability platforms
- Production feedback loop for continuous improvement
- 2025 trends: GraphRAG, multi-agent evaluation, metric standardization

**📊 License:** Apache 2.0 | **Support:** Enterprise support via consultation

---

### DeepEval
**🔗 Links:** [Website](https://www.confident-ai.com) · [Docs](https://deepeval.com/docs/getting-started) · [GitHub](https://github.com/confident-ai/deepeval) · [DeepTeam](https://github.com/confident-ai/deepteam)

**⚡ What:** Open-source LLM evaluation framework with AI red teaming capabilities

**🎯 Use When:**
- Security testing LLM applications for vulnerabilities (bias, PII leakage, harmful content)
- Evaluating LLMs with 30+ plug-and-use metrics (faithfulness, hallucination, toxicity)
- Automated adversarial attack simulation (jailbreaking, prompt injection, data extraction)
- CI/CD integration for regression testing and quality gates
- OWASP Top 10 for LLMs and NIST AI Risk Management compliance

**💪 Why:**
- **Red Teaming (DeepTeam):** Detect 40+ vulnerability types, simulate 10+ attack methods, no dataset required
- **Evaluation Metrics:** 30+ research-backed metrics for end-to-end and component-level testing
- **Confident AI Platform:** Cloud platform for monitoring, tracing, A/B testing, real-time insights
- Synthetic dataset generation with state-of-the-art evolution techniques
- Integrates with CI/CD, LangChain, AWS Bedrock, Azure AI Foundry
- Data residency options: US (North Carolina) or EU (Frankfurt)
- SOC 2 Type 2 compliant with custom permissions and PII masking

**⚠️ Note:** DeepTeam (red teaming) dynamically simulates attacks at runtime; DeepEval (evaluation) requires prepared datasets

**📊 License:** Apache 2.0 | **Support:** Confident AI Enterprise + Community

---

## 📊 Observability & Monitoring

### Prometheus
**🔗 Links:** [Website](https://prometheus.io) · [Docs](https://prometheus.io/docs) · [GitHub](https://github.com/prometheus/prometheus)

**⚡ What:** De facto standard for Kubernetes monitoring (90% CNCF adoption)

**🎯 Use When:**
- Monitoring Kubernetes clusters and applications at scale
- Need multi-dimensional time-series metrics with labels
- Service discovery for ephemeral workloads
- Multi-cluster management (Thanos/Mimir/Cortex for 3-300 clusters)

**💪 Why:**
- Multi-dimensional data model matching Kubernetes labels
- Auto-discovery of scrape targets (perfect for K8s)
- Prometheus Operator: declarative full ecosystem management
- Federation, remote storage, GitOps for enterprise scale

**📊 License:** Apache 2.0 | **Support:** Community + CNCF

---

### Grafana
**🔗 Links:** [Website](https://grafana.com) · [Docs](https://grafana.com/docs) · [GitHub](https://github.com/grafana/grafana)

**⚡ What:** Observability platform for visualizations and dashboards

**🎯 Use When:**
- Unified observability across metrics, logs, traces, profiles
- Need dynamic dashboards with observability-as-code (Grafana 12)
- Multi-data source visualization (Prometheus, Loki, Splunk, Datadog, etc.)
- AI-powered incident resolution (Grafana Assistant)

**💪 Why:**
- **Grafana 12 (May 2025):** Git Sync, dynamic dashboards, native alert management
- Observability as code: version, validate, deploy dashboards like code
- Enterprise plugins: Splunk, ServiceNow, Datadog integrations
- Usage insights: user behavior, dashboard utilization, data source metrics

**📊 License:** AGPL 3.0 (OSS) | **Support:** Grafana Enterprise + Cloud

---

### Grafana Loki
**🔗 Links:** [Website](https://grafana.com/oss/loki) · [Docs](https://grafana.com/docs/loki) · [GitHub](https://github.com/grafana/loki)

**⚡ What:** Cost-effective log aggregation inspired by Prometheus

**🎯 Use When:**
- Need cost-effective log aggregation at massive scale
- Familiar with Prometheus (uses same labels/service discovery)
- Kubernetes logs with OpenTelemetry/Grafana Alloy
- Want horizontal scalability with low-cost object storage

**💪 Why:**
- Only indexes metadata (labels), not log contents → huge cost savings
- LogQL query language familiar to PromQL users
- 2025: Grafana Alloy ingestion (OTel Collector distribution)
- Deployment modes: Monolithic (simple) → Distributed (scalable)

**📊 License:** AGPL 3.0 | **Support:** Grafana Cloud Logs + Enterprise Logs

---

## ☁️ Cloud Platforms

### AWS Bedrock
**🔗 Links:** [Website](https://aws.amazon.com/bedrock) · [Docs](https://docs.aws.amazon.com/bedrock)

**⚡ What:** Managed service for building GenAI applications with foundation models

**🎯 Use When:**
- Need access to multiple FMs (Anthropic, Meta, Cohere, Mistral, Amazon Titan)
- Building production AI agents at scale with governance
- Require enterprise security, compliance, and private endpoints
- Want unified tool server for multi-agent applications

**💪 Why:**
- **AgentCore (Preview 2025):** Runtime, Memory, Gateway for AI agents
- Supports any framework/model, works with LangChain, LlamaIndex, etc.
- Enterprise governance: encryption, access mgmt, model governance
- Success: Robinhood 5B tokens/day, 80% cost reduction, 50% dev time cut

**💰 Pricing:** Free preview until Sept 16, 2025 → pay-per-use after

**📊 License:** Proprietary | **Support:** AWS Enterprise Support

---

### Azure OpenAI Service
**🔗 Links:** [Website](https://azure.microsoft.com/products/ai-services/openai-service) · [Docs](https://learn.microsoft.com/azure/ai-services/openai) · [Pricing](https://azure.microsoft.com/pricing/details/cognitive-services/openai-service)

**⚡ What:** Enterprise OpenAI models on Azure with data privacy guarantees

**🎯 Use When:**
- Need OpenAI models (GPT-5, o1, DALL-E, Whisper) with enterprise SLAs
- Require data residency and compliance (no data leaves Azure)
- Integration with Fabric, Cosmos DB, Azure AI Search
- Microsoft partnership extended through 2032

**💪 Why:**
- **New Partnership (Oct 2025):** Microsoft 27% stake, $250B Azure commitment, IP rights through 2032
- GPT-5 GA on Azure AI Foundry (Aug 2025)
- gpt-oss: OpenAI's first open-weight release since GPT-2
- Regional flexibility, private endpoints, compliance built-in

**💰 Pricing:** Pay-per-use, same as OpenAI API + Azure infra costs

**📊 License:** Proprietary | **Support:** Azure Enterprise Support

---

## 🎨 Application Development

### Vercel AI SDK
**🔗 Links:** [Website](https://ai-sdk.dev) · [Docs](https://ai-sdk.dev/docs) · [GitHub](https://github.com/vercel/ai)

**⚡ What:** TypeScript toolkit for AI-powered frontends

**🎯 Use When:**
- Building AI chat interfaces with streaming responses
- Next.js applications requiring server-side AI integration
- Need multi-model support (30+ LLM providers)
- Edge runtime deployments for low-latency global inference

**💪 Why:**
- Unified API across OpenAI, Anthropic, Google, AWS Bedrock, open-source
- First-class streaming with React Server Components (RSC)
- **AI SDK 6 (beta 2025):** Agent abstraction, tool execution approval, human-in-the-loop
- Vercel AI Cloud: AI Gateway, DDoS/bot protection, WAF, Fluid Compute

**💰 Pricing:** Free tier → Pro → Enterprise with custom DDoS/IP blocking

**📊 License:** Apache 2.0 | **Support:** Vercel Enterprise

---

## 📚 Quick Reference

### By Data Flow

```
1. Data Validation → Pydantic
2. Text Processing → spaCy
3. Document Ingestion → Unstructured.io, Docling
4. Data Curation → NeMo Curator
5. Distributed Processing → RAPIDS, Ray
6. Privacy/PII Removal → Presidio
7. Vector Storage → Milvus, PostgreSQL+pgvector
8. LLM Providers → OpenAI, Claude, Cohere, Grok, Hugging Face
9. LLM Inference → vLLM, NIM, Ray
10. Agent Orchestration → LangChain/LangGraph
11. RAG Evaluation → Ragas, DeepEval
12. Monitoring → Prometheus, Grafana, Loki
13. Cloud Platforms → AWS Bedrock, Azure OpenAI
14. Frontend Integration → Vercel AI SDK
```

### By Use Case

**Building RAG Application:**
```
Documents → Unstructured/Docling → Presidio → Milvus/pgvector
Query → LangChain → vLLM/NIM/Ray → Response
Evaluate → Ragas, DeepEval
Red Team → DeepEval/DeepTeam
Monitor → LangSmith, Prometheus, Grafana
Frontend → Vercel AI SDK
```

**Training Foundation Model:**
```
Raw Data → NeMo Curator → RAPIDS (processing) → Ray (distributed training)
Model → vLLM/NIM/Ray (serving) → Production
Monitor → Prometheus, Grafana
```

**Production LLM App:**
```
Inputs → Pydantic validation → spaCy preprocessing
Distributed Processing → RAPIDS, Ray
LLM → OpenAI/Claude/Cohere via vLLM/NIM/Ray
Vector Search → Milvus/pgvector
Monitor → Prometheus, Grafana, Loki
Cloud → AWS Bedrock or Azure OpenAI
```

---

## 🏢 Enterprise Support Summary

| Tool | License | Enterprise Support |
|------|---------|-------------------|
| Pydantic | MIT | Community + Consulting |
| spaCy | MIT | Community + Commercial pipelines |
| Unstructured.io | Apache 2.0 | Plus + Enterprise |
| Docling | MIT | IBM + Red Hat RHEL AI |
| NeMo Curator | Apache 2.0 | NVIDIA AI Enterprise |
| RAPIDS | Apache 2.0 | NVIDIA AI Enterprise |
| Presidio | MIT | Community + Microsoft |
| Milvus | Apache 2.0 | Zilliz Cloud ($99/mo+) |
| PostgreSQL+pgvector | PostgreSQL | Cloud providers (AWS, GCP, Azure) |
| OpenAI | Proprietary | Enterprise plans |
| Anthropic Claude | Proprietary | Enterprise + AWS Marketplace |
| Cohere | Proprietary | Enterprise support |
| xAI Grok | Proprietary | Enterprise custom |
| Hugging Face | Varies | Enterprise Hub |
| vLLM | Apache 2.0 | Red Hat OpenShift AI + llm-d |
| NVIDIA NIM | NVIDIA AI Enterprise | NVIDIA AI Enterprise |
| Ray | Apache 2.0 | Anyscale (Azure/AWS managed) |
| LangChain | MIT | LangSmith Plus + Enterprise |
| Ragas | Apache 2.0 | Enterprise consultation |
| DeepEval | Apache 2.0 | Confident AI Enterprise + Community |
| Prometheus | Apache 2.0 | CNCF Community |
| Grafana | AGPL 3.0 | Grafana Enterprise + Cloud |
| Loki | AGPL 3.0 | Grafana Enterprise + Cloud |
| AWS Bedrock | Proprietary | AWS Enterprise Support |
| Azure OpenAI | Proprietary | Azure Enterprise Support |
| Vercel AI SDK | Apache 2.0 | Vercel Enterprise |

---

## 🔗 All Links

**Documentation:**
- [Pydantic](https://docs.pydantic.dev) · [spaCy](https://spacy.io) · [Unstructured](https://docs.unstructured.io) · [Docling](https://docling.ai/docs)
- [NeMo Curator](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/) · [RAPIDS](https://docs.rapids.ai) · [Presidio](https://microsoft.github.io/presidio)
- [Milvus](https://milvus.io/docs) · [pgvector](https://github.com/pgvector/pgvector) · [OpenAI](https://platform.openai.com/docs)
- [Claude](https://www.anthropic.com/api) · [Cohere](https://cohere.com/embed) · [Grok](https://docs.x.ai) · [Hugging Face](https://huggingface.co/docs)
- [vLLM](https://docs.vllm.ai) · [NIM](https://developer.nvidia.com/nim) · [Ray](https://docs.ray.io)
- [LangChain](https://python.langchain.com) · [Ragas](https://docs.ragas.io) · [DeepEval](https://deepeval.com/docs/getting-started)
- [Prometheus](https://prometheus.io/docs) · [Grafana](https://grafana.com/docs) · [Loki](https://grafana.com/docs/loki)
- [AWS Bedrock](https://docs.aws.amazon.com/bedrock) · [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai)
- [Vercel AI SDK](https://ai-sdk.dev/docs)

**GitHub Repositories:**
- [Pydantic](https://github.com/pydantic/pydantic) · [spaCy](https://github.com/explosion/spaCy) · [Unstructured](https://github.com/Unstructured-IO/unstructured)
- [Docling](https://github.com/DS4SD/docling) · [NeMo](https://github.com/NVIDIA/NeMo) · [RAPIDS](https://github.com/rapidsai)
- [Presidio](https://github.com/microsoft/presidio) · [Milvus](https://github.com/milvus-io/milvus) · [pgvector](https://github.com/pgvector/pgvector)
- [vLLM](https://github.com/vllm-project/vllm) · [Ray](https://github.com/ray-project/ray) · [LangChain](https://github.com/langchain-ai/langchain)
- [Ragas](https://github.com/explodinggradients/ragas) · [DeepEval](https://github.com/confident-ai/deepeval) · [DeepTeam](https://github.com/confident-ai/deepteam)
- [Prometheus](https://github.com/prometheus/prometheus)
- [Grafana](https://github.com/grafana/grafana) · [Loki](https://github.com/grafana/loki) · [Vercel AI SDK](https://github.com/vercel/ai)

---

## 🤝 Contributing

See `.claude/CLAUDE.md` for guidelines on adding new tools to this list.

**Criteria for inclusion:**
✅ Production-ready and enterprise-proven
✅ Active maintenance (2025 updates)
✅ Clear enterprise value proposition
✅ Official support channels

---

**Made with ❤️ for the AI Engineering Community**
