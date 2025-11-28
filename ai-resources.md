# AI Resources & Development Tools

> Curated prompt libraries, code repositories, frameworks, and developer resources for building AI applications

**Last Updated:** 2025-11-28

---

## Table of Contents

- [Prompt Libraries](#-prompt-libraries)
- [AI Code Repositories](#-ai-code-repositories)
- [Agent Frameworks](#-agent-frameworks)
- [Production Templates](#-production-templates)
- [Quick Reference](#quick-reference)

---

## 📝 Prompt Libraries

### Gamma.app Prompts
**🔗 Links:** [Website](https://gamma.app) · [Prompts Library](https://gamma.app/prompts)

**⚡ What:** 100+ prompts and API workflows for presentations, documents, and websites

**🎯 Use When:**
- Creating educational presentations (literary analysis, lesson plans)
- Building business demos (discovery calls, product showcases)
- Need structured two-step process (outline → slides)
- Want to control presentation density and style

**💪 Why:**
- **100+ Prompts:** Educational, business, creative categories
- **Best Practices Built-In:** Outcome + audience, outline approval, density constraints
- **Anti-Cliché Features:** Negative prompts to forbid overused phrases
- **Template Examples:** College presentations, HR demos, lesson plans
- **Two-Step Process:** Approve outlines before slide generation

**📊 License:** Proprietary | **Access:** Free tier available

---

### Anthropic Claude Prompt Library
**🔗 Links:** [Prompt Library](https://docs.anthropic.com/claude/docs/intro-to-prompting) · [System Prompts](https://docs.claude.com/en/release-notes/system-prompts) · [Cookbook](https://github.com/anthropics/anthropic-cookbook)

**⚡ What:** Official prompts from Claude's creators covering work and creative use cases

**🎯 Use When:**
- Learning prompt engineering fundamentals
- Need production-grade system prompts (Claude Opus 4, Sonnet 4)
- Building Claude-powered applications
- Want interactive tutorials and code examples

**💪 Why:**
- **Official Authority:** Direct from Anthropic engineering team
- **Published System Prompts:** Full prompts used in Claude.ai and mobile apps
- **Interactive Tutorial:** Google Sheets extension for hands-on learning
- **Anthropic Cookbook:** Code examples and implementation patterns
- **Community Collection:** [awesome-claude-prompts](https://github.com/langgptai/awesome-claude-prompts) - 50+ curated prompts
- **Recent Updates:** Opus 4 and Sonnet 4 system prompts (Jan 2025)

**📊 License:** Open Source (Cookbook) | **Access:** Free

---

### OpenAI Prompt Resources
**🔗 Links:** [Cookbook](https://cookbook.openai.com/) · [Best Practices](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api) · [Prompt Examples](https://platform.openai.com/docs/examples)

**⚡ What:** Official examples and guides organized by use case with production patterns

**🎯 Use When:**
- Building agents, multimodal apps, or text processing pipelines
- Need guardrails and optimization patterns
- Want specialized GPT guides (GPT-5.1, GPT-5-Codex, Sora 2)
- Implementing enterprise patterns (regulatory drafting, supply-chain automation)

**💪 Why:**
- **Organized by Category:** Agents, Multimodal, Text Processing, Guardrails, Optimization
- **Recent Examples (Nov 2025):** Self-evolving agents, voice agents, GPT-5.1 coding assistants
- **Specialized Guides:** Model-specific prompting (GPT-4.1, GPT-5, Codex, Sora 2)
- **Production Patterns:** Regulatory documents, automation, evaluation, self-healing workflows
- **Reproducible:** Code snippets, datasets, step-by-step walkthroughs

**📊 License:** MIT (Cookbook) | **Access:** Free

---

### DAIR.AI Prompt Engineering Guide
**🔗 Links:** [GitHub](https://github.com/dair-ai/Prompt-Engineering-Guide) · [Web Version](https://www.promptingguide.ai/)

**⚡ What:** Comprehensive academic and practical guide (50k+ stars, 3M+ learners)

**🎯 Use When:**
- Learning advanced techniques (Chain-of-Thought, Tree of Thoughts, ReAct)
- Need multilingual support (13 languages)
- Building RAG systems or AI agents
- Want research-backed methods with Jupyter notebooks

**💪 Why:**
- **50,000+ GitHub Stars:** Most comprehensive community resource
- **3M+ Learners Worldwide:** Proven educational value
- **Advanced Techniques:** Zero-shot, few-shot, CoT, ToT, ReAct, RAG
- **Practical Applications:** Data generation, code generation, function calling
- **Learning Resources:** 1-hour video lecture, Jupyter notebooks, research papers
- **Model Coverage:** ChatGPT, GPT-4, Claude, LLaMA, Mistral, Gemini
- **DAIR.AI Academy:** Structured courses

**📊 License:** MIT | **Access:** Free | **Last Updated:** Nov 17, 2025

---

### Community Prompt Collections

#### GitHub Collections
**🔗 Links:**
- **[Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts)** - 134k stars, largest community collection
- **[awesome-copilot](https://github.com/github/awesome-copilot)** - GitHub's official collection for Copilot
- **[ai-boost/awesome-prompts](https://github.com/ai-boost/awesome-prompts)** - Curated from GPT Store top GPTs

**⚡ What:** Community-curated prompt collections with production examples

**💪 Why:**
- Largest community collections (100k+ stars combined)
- Real-world use cases from production applications
- Regularly updated by active communities
- Free and open source

---

#### DocsBot AI Prompts
**🔗 Links:** [DocsBot Prompts](https://docsbot.ai/prompts)

**⚡ What:** Free ready-to-use prompts for ChatGPT, Claude, Gemini

**🎯 Use When:** Need quick templates without registration

**📊 Access:** Free

---

#### God of Prompt
**🔗 Links:** [Library](https://www.godofprompt.ai/prompt-library)

**⚡ What:** 30,000+ prompts for marketing, SEO, e-commerce

**💰 Pricing:** Free to $150 lifetime access

---

#### Hugging Face Datasets
**🔗 Links:** [Hub](https://huggingface.co/datasets)

**⚡ What:** Machine learning prompt datasets

- **P3 (Public Pool of Prompts):** Collection for T0* models, diverse NLP tasks
- **fka/awesome-chatgpt-prompts:** ChatGPT prompts dataset
- **eltorio/ai-prompts:** Email, translation, summarization, code review
- **Chat Templates:** Official preprocessing examples

**📊 License:** Various | **Access:** Free

---

## 💻 AI Code Repositories

### Claude Code Resources

#### Official Repository
**🔗 Links:** [GitHub](https://github.com/anthropics/claude-code)

**⚡ What:** Agentic coding tool for terminal

**💪 Features:**
- Understands entire codebase
- Executes tasks via natural language
- Explains code architecture
- Handles git workflows
- Command-line integration

**📊 License:** Anthropic | **Access:** Free

---

#### Awesome Claude Code
**🔗 Links:** [GitHub](https://github.com/hesreallyhim/awesome-claude-code)

**⚡ What:** Most comprehensive Claude Code community resource (12.1k stars, 658 forks)

**🎯 Use When:**
- Need agent skills for code generation
- Want workflow templates (Laravel, n8n, design review)
- Building custom slash commands
- Integrating with GitHub/version control
- Setting up monitoring and automation

**💪 Categories:**
- **Agent Skills:** Code generation, web assets
- **Workflows & Knowledge Guides:** Laravel, n8n, design review checklists
- **Tooling:** Session managers, log viewers, template browsers, GitHub integration
- **Status Lines & Hooks:** Monitoring, automation triggers
- **Slash-Commands:** Version control, testing, documentation, CI/CD
- **CLAUDE.md Files:** Language/domain-specific configurations

**📊 License:** MIT | **Access:** Free

---

#### Comprehensive Guides
**🔗 Links:**
- **[claude-code-guide](https://github.com/Cranot/claude-code-guide)** - Version 2025.0 (Jan 2025), verified features
- **[Claude Code Everything](https://github.com/wesammustafa/Claude-Code-Everything-You-Need-to-Know)** - All-in-one guide with BMAD method
- **[ClaudeLog](https://claudelog.com/claude-code-mcps/awesome-claude-code/)** - Docs, guides, tutorials, best practices

**⚡ What:** Complete learning resources for Claude Code

**📊 Access:** Free

---

### Vercel AI SDK Templates

**🔗 Links:** [Templates](https://vercel.com/templates/ai) · [GitHub](https://github.com/vercel/ai) · [Docs](https://ai-sdk.dev/docs/introduction) · [Academy Course](https://vercel.com/academy/ai-sdk)

**⚡ What:** 32+ production-ready AI application templates from Next.js creators

**🎯 Use When:**
- Building chatbots (Next.js, Gemini, Nuxt)
- Creating generative UI applications
- Need RAG chatbot (Next.js + AI SDK + Drizzle + PostgreSQL)
- Implementing Model Context Protocol (MCP)
- Deploying to production quickly (under 60 seconds)

**💪 Template Categories:**

**Chatbots:**
- Next.js AI Chatbot (full-featured, hackable)
- Gemini AI Chatbot
- Nuxt AI Chatbot

**Specialized Applications:**
- Morphic (answer engine with generative UI)
- qrGPT (AI QR code generation)
- AI Headshot Generator
- PDF to Quiz Generator
- Alt Text Generator

**Advanced Use Cases:**
- RAG chatbot with PostgreSQL
- Generative UI chatbot (streams React Server Components)
- ChatGPT app with MCP
- LangChain projects (chat, agents, retrieval)
- Express app integration

**📊 License:** MIT | **Access:** Free | **Deployment:** One-click Vercel deployment

---

### LangChain & LangGraph

#### LangGraph Framework
**🔗 Links:** [GitHub](https://github.com/langchain-ai/langgraph) · [Examples](https://langchain-ai.github.io/langgraph/examples/) · [Docs](https://langchain-ai.github.io/langgraph/)

**⚡ What:** Industry-standard orchestration framework for stateful multi-agent systems

**🎯 Use When:**
- Building long-running, stateful agents
- Need human-in-the-loop workflows
- Implementing multi-agent coordination
- Require durable execution across failures
- Production deployment with observability

**💪 Why:**
- **Trusted by Enterprises:** Klarna, Replit, Elastic
- **Durable Execution:** State persists across failures
- **Human-in-the-Loop:** Inspect/modify state mid-execution
- **Comprehensive Memory:** Working + long-term memory
- **LangSmith Integration:** Full tracing and metrics
- **Production Infrastructure:** Battle-tested deployment patterns
- **Version 1.0:** Commitment to stability until 2.0

**Getting Started:**
```bash
pip install -U langgraph
```

**Example Workflows:**
- ReAct agents
- Memory management
- Retrieval systems
- Agentic RAG
- Multi-agent workflows

**📊 License:** MIT | **Access:** Free

---

#### LangChain Ecosystem
**🔗 Links:**
- **[LangChain](https://github.com/langchain-ai/langchain)** - Components library
- **[LangSmith](https://www.langchain.com/langsmith)** - Observability platform
- **[LangGraph Studio](https://github.com/langchain-ai/langgraph-studio)** - Visual prototyping
- **[awesome-LangGraph](https://github.com/von-development/awesome-LangGraph)** - Community index

**Production Examples:**
- Open SWE (async coding agent)
- LinkedIn SQL Bot (multi-agent system)

**📊 License:** MIT | **Access:** Free (LangSmith has paid tiers)

---

## 🤖 Agent Frameworks

### CrewAI
**🔗 Links:** [GitHub](https://github.com/crewAIInc/crewAI) · [Docs](https://docs.crewai.com/)

**⚡ What:** Fast-growing enterprise framework (100,000+ certified developers, 20% market share)

**🎯 Use When:**
- Building role-based autonomous agent teams
- Need event-driven workflows with conditional logic
- Want lean, lightning-fast Python framework
- Implementing complex data analysis or research automation
- Production business process orchestration

**💪 Why:**
- **Dual Architecture:**
  - **Crews:** Teams of autonomous agents with role-based collaboration
  - **Flows:** Event-driven workflows with conditional logic
- **Independent:** Not dependent on LangChain (lean & fast)
- **100,000+ Certified Developers:** Largest certification program
- **20% Market Adoption:** 3rd most popular framework
- **Python 3.10+ Required:** Modern Python features

**Getting Started:**
```bash
pip install crewai
# or with tools
pip install 'crewai[tools]'

# Generate project
crewai create crew <project_name>
```

**Example Types:**
- Landing Page Generator
- Trip Planner
- Stock Analysis
- Job Posting
- Human Input Integration

**Learning:** deeplearning.ai courses, community documentation

**📊 License:** MIT | **Access:** Free

---

### Other Leading Frameworks

#### AutoGPT
**⚡ What:** Autonomous agent framework (25% market share)

**💪 Features:**
- Self-planning goal-driven assistant
- Breaks jobs into subtasks
- Autonomous execution

**🎯 Use:** Experimentation, prototyping, learning

---

#### Microsoft AutoGen
**⚡ What:** Multi-agent conversation framework

**🎯 Use Cases:** Complex workflows requiring agent coordination

---

#### Microsoft Semantic Kernel
**⚡ What:** SDK for AI orchestration

**🎯 Integration:** Enterprise Microsoft stack

---

#### LlamaIndex
**⚡ What:** Data framework for LLM applications

**🎯 Use Cases:** RAG, document retrieval, knowledge bases

---

**Market Forecast:** Gartner predicts 33% of enterprise software will incorporate agentic AI by 2028 (vs <1% in 2024)

---

## 🏗️ Production Templates

### Microsoft AI Templates
**🔗 Links:** [Azure AI Templates](https://learn.microsoft.com/en-us/azure/developer/ai/intelligent-app-templates) · [.NET AI Template](https://devblogs.microsoft.com/dotnet/announcing-dotnet-ai-template-preview1/)

**⚡ What:** Enterprise-grade AI application templates

**Templates:**
- **Contoso Chat:** Retail copilot using RAG pattern, GenAIOps workflow with Azure AI and Prompty
- **.NET AI Chat Template:** Quick build for AI chat apps using Microsoft.Extensions.AI and Microsoft.Extensions.VectorData

**Pattern Focus:** Retrieval Augmented Generation (RAG)

**📊 License:** Microsoft | **Access:** Free

---

### Production Design Patterns
**🔗 Links:** [InfoQ - Design Patterns](https://www.infoq.com/articles/practical-design-patterns-modern-ai-systems/) · [AI Assisted Development 2025](https://www.infoq.com/minibooks/ai-assisted-development-2025/)

**Categories:**

1. **Prompting & Context Patterns:** Crafting effective instructions, providing relevant context
2. **Agent Interaction Patterns:** Multi-agent system management
3. **Cost Optimization Patterns:** Reducing compute costs
4. **Safety Patterns:** Preventing misleading content, building user trust
5. **Reliability Patterns:** Guide model output consistency

**Best Practices:**
- **Beyond PoC:** Architecture, process, accountability matter
- **Responsible Integration:** AI in delivery pipelines with human judgment
- **Foundation:** Prompt engineering determines reliability, efficiency, cost

**Recommended Workflow:**
1. Pick one high-value use case
2. Define accuracy, faithfulness, latency targets
3. Create zero-shot and few-shot variants with JSON schema
4. Compare outputs across 2-3 models
5. Attach evaluators (faithfulness, format adherence, domain checks)

**Mitigations:**
- Retrieval-augmented generation
- Citation/source pinning
- Evaluator loops
- Grounding on authoritative corpora

**📊 Access:** Free

---

## Quick Reference

### By Resource Type

**Learning Prompt Engineering:**
```
Fundamentals → DAIR.AI Prompt Engineering Guide (50k+ stars)
Claude-Specific → Anthropic Prompt Library + Cookbook
OpenAI-Specific → OpenAI Cookbook + Best Practices
Interactive → Anthropic Google Sheets Tutorial
Research-Backed → DAIR.AI (papers, notebooks, courses)
```

**Building AI Applications:**
```
Next.js/React → Vercel AI SDK Templates (32+ templates)
Multi-Agent → LangGraph (stateful, production-ready)
Role-Based Agents → CrewAI (100k+ certified devs)
RAG Systems → LlamaIndex, LangChain
Microsoft Stack → Semantic Kernel, Azure AI Templates
```

**Claude Code Development:**
```
Official → anthropics/claude-code
Community → awesome-claude-code (12.1k stars)
Workflows → claude-code-guide, ClaudeLog
Skills → Agent skills, slash commands, CLAUDE.md configs
```

**Prompt Collections:**
```
Community → Awesome ChatGPT Prompts (134k stars)
Quick Templates → DocsBot AI Prompts (free)
Marketing/SEO → God of Prompt (30k+ prompts)
Dataset Training → Hugging Face datasets
Platform-Specific → Gamma (presentations), Copilot (GitHub)
```

### By Use Case

**Building Chatbots:**
```
Next.js → Vercel AI SDK (Next.js AI Chatbot template)
RAG → Vercel template (Next.js + Drizzle + PostgreSQL)
Generative UI → Vercel template (React Server Components)
Multi-Agent → LangGraph
```

**Production AI Systems:**
```
Stateful Agents → LangGraph (durable execution, memory)
Autonomous Teams → CrewAI (role-based collaboration)
Microsoft → Azure AI Templates (Contoso Chat)
Workflows → CrewAI Flows (event-driven, conditional)
```

**Learning & Experimentation:**
```
Tutorials → DAIR.AI Guide (video + notebooks)
Interactive → Anthropic Google Sheets
Examples → OpenAI Cookbook (production patterns)
Prototyping → LangGraph Studio (visual)
```

### Framework Comparison

| Framework | Best For | Market Share | Key Feature |
|-----------|----------|--------------|-------------|
| **LangGraph** | Stateful multi-agent | Industry standard | Durable execution |
| **CrewAI** | Role-based teams | 20% | Event-driven flows |
| **AutoGPT** | Autonomous tasks | 25% | Self-planning |
| **LlamaIndex** | RAG systems | - | Data framework |
| **Semantic Kernel** | Microsoft stack | - | Enterprise SDK |

---

## Planned Additions

- [ ] Additional prompt libraries (PromptBase, AIPRM, PromptHero)
- [ ] LangChain Hub integration examples
- [ ] More Claude Code workflows
- [ ] AutoGen examples and templates
- [ ] Semantic Kernel tutorials
- [ ] LlamaIndex production patterns
- [ ] Prompt engineering evaluation tools
- [ ] Multi-modal prompt examples
- [ ] Voice agent templates
- [ ] Video generation prompts (Sora 2)

---

## Integration Patterns

### RAG Application with AI SDK
```
Documents → Unstructured/Docling (from enterprise-ai-tools.md)
         → Vector Database (Milvus/pgvector/Chroma)
         → Vercel AI SDK Template (RAG chatbot)
         → Cohere Rerank (precision)
         → LangSmith (observability)
```

### Multi-Agent Workflow
```
User Request → LangGraph (orchestration)
            → CrewAI Crews (role-based agents)
            → Claude Code (coding tasks)
            → LangSmith (tracing)
            → Production deployment
```

### Prompt Engineering Pipeline
```
Use Case → DAIR.AI Guide (techniques)
        → Anthropic Cookbook (Claude examples)
        → OpenAI Cookbook (GPT examples)
        → Evaluators (faithfulness, format)
        → Production deployment
```

---

## Sources

- [Gamma Prompt Library](https://gamma.app/prompts)
- [Best Prompt Patterns for Gamma AI](https://skywork.ai/blog/best-prompt-patterns-gamma-ai-presentation-2025/)
- [Anthropic Prompt Library](https://docs.anthropic.com/claude/docs/intro-to-prompting)
- [Anthropic System Prompts](https://docs.claude.com/en/release-notes/system-prompts)
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook)
- [Awesome Claude Prompts](https://github.com/langgptai/awesome-claude-prompts)
- [OpenAI Cookbook](https://cookbook.openai.com/)
- [OpenAI Best Practices](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)
- [DAIR.AI Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts)
- [awesome-copilot](https://github.com/github/awesome-copilot)
- [ai-boost/awesome-prompts](https://github.com/ai-boost/awesome-prompts)
- [DocsBot AI Prompts](https://docsbot.ai/prompts)
- [God of Prompt Library](https://www.godofprompt.ai/prompt-library)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Awesome Claude Code](https://github.com/hesreallyhim/awesome-claude-code)
- [Claude Code Guide](https://github.com/Cranot/claude-code-guide)
- [Vercel AI SDK](https://github.com/vercel/ai)
- [Vercel AI Templates](https://vercel.com/templates/ai)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [awesome-LangGraph](https://github.com/von-development/awesome-LangGraph)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)
- [Microsoft AI Templates](https://learn.microsoft.com/en-us/azure/developer/ai/intelligent-app-templates)
- [InfoQ Design Patterns](https://www.infoq.com/articles/practical-design-patterns-modern-ai-systems/)
- [AI Assisted Development 2025](https://www.infoq.com/minibooks/ai-assisted-development-2025/)

---

**Made for the AI Developer Community**
