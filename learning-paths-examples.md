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

## Version Information
- **Last Updated:** 2025-11-12
- **Total Resources:** 3
- **Categories Covered:** 3 (GPU-Accelerated Data Processing, AI Engineering & Production Systems, Prompt Engineering & LLM Techniques)

---

## Planned Additions

Future learning resources to be added:

### High Priority
- [ ] LangChain/LangGraph official examples and tutorials
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
