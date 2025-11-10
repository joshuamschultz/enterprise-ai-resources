# Learning Paths & Examples for Enterprise AI

A curated collection of tutorials, playbooks, example repositories, and learning resources for mastering enterprise AI tools and workflows.

---

## GPU-Accelerated Data Processing

### NVIDIA DGX Spark Playbooks
**Repository:** https://github.com/nvidia/dgx-spark-playbooks
**Platform:** NVIDIA DGX Spark (Blackwell architecture)
**Level:** Beginner to Advanced

**What It Covers:**
Step-by-step playbooks for setting up and running AI/ML workloads on NVIDIA hardware, specifically designed for the DGX Spark desktop supercomputer.

**Key Learning Areas:**
- **GPU-Accelerated Data Science:**
  - RAPIDS (cuDF for pandas acceleration, cuML for scikit-learn)
  - Zero-code-change GPU acceleration
  - Processing 250 MB datasets in seconds
  - Handling tens of millions of records with cuDF pandas

- **Machine Learning Workflows:**
  - UMAP and HDBSCAN clustering on GPU
  - Model training and fine-tuning (LLMs and vision models)
  - High-performance numerical computing with JAX

- **Production AI Applications:**
  - AI code assistants and chatbots
  - Video search and summarization
  - Image generation
  - LLM inference optimization

**Why Use This:**
- Official NVIDIA resource with production-grade examples
- Demonstrates real-world GPU acceleration benefits (10-100x speedups)
- Covers complete workflow: data loading → preprocessing → training → inference
- Blackwell architecture optimization patterns
- Foundation for understanding CUDA-X Data Science stack

**Prerequisites:**
- Basic Python knowledge
- Familiarity with pandas and scikit-learn (for RAPIDS modules)
- Access to NVIDIA GPU (DGX Spark ideal, but adaptable to other NVIDIA hardware)

**Related Tools:**
- RAPIDS (cuDF, cuML, cuGraph)
- CUDA-X libraries
- JAX for numerical computing
- TensorRT for inference optimization

---

## Format Guidelines

Each learning resource entry should include:

### Required Fields:
1. **Title/Name** - Clear, descriptive name
2. **Repository/Link** - Direct URL to resource
3. **Platform/Framework** - What technology it focuses on
4. **Level** - Beginner, Intermediate, Advanced, or combination

### Recommended Fields:
5. **What It Covers** - Brief overview (1-2 sentences)
6. **Key Learning Areas** - Bulleted list of main topics/skills
7. **Why Use This** - Value proposition and use cases
8. **Prerequisites** - Required knowledge or setup
9. **Related Tools** - Links to other tools in enterprise-ai-tools.md

### Optional Fields:
10. **Estimated Time** - How long to complete
11. **Hands-On Projects** - Practical exercises included
12. **Community** - Link to forums, Discord, Slack
13. **Certification** - If completion leads to credential

---

## Categories

Learning resources are organized by:

1. **Data Ingestion & ETL**
   - Document processing pipelines
   - Multi-source data integration
   - Real-time data streaming

2. **Data Processing & Curation**
   - Dataset quality improvement
   - GPU-accelerated analytics
   - Feature engineering at scale

3. **Distributed Computing**
   - Ray ecosystem tutorials
   - Multi-node training
   - Cluster management

4. **LLM Inference & Serving**
   - Production inference optimization
   - Multi-model serving
   - Latency and throughput tuning

5. **Agentic Workflows**
   - LangChain/LangGraph examples
   - Multi-agent orchestration
   - Human-in-the-loop patterns

6. **Data Privacy & Security**
   - PII detection and anonymization
   - Compliance workflows
   - Secure LLM applications

7. **Application Development**
   - End-to-end AI application building
   - Frontend integration
   - Production deployment

8. **MLOps & Production**
   - CI/CD for ML
   - Monitoring and observability
   - Model versioning and governance

---

## Contributing New Resources

When adding new learning paths or examples:

1. **Verify Quality:**
   - Official or well-maintained repositories preferred
   - Active community (recent commits, responsive maintainers)
   - Clear documentation

2. **Ensure Relevance:**
   - Focus on enterprise-scale applications
   - Production-grade patterns and practices
   - Avoid toy examples or outdated tutorials

3. **Check Compatibility:**
   - Tools should align with enterprise-ai-tools.md
   - Version compatibility clearly stated
   - Dependencies documented

4. **Structure Consistently:**
   - Follow the format guidelines above
   - Use markdown formatting for readability
   - Include direct, stable URLs

---

## Version Information
- **Last Updated:** 2025-11-09
- **Total Resources:** 1
- **Categories Covered:** 1 (GPU-Accelerated Data Processing)

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
