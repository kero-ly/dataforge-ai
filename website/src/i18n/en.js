export default {
  nav: {
    home: 'Home',
    research: 'Research',
    publications: 'Publications',
    news: 'News',
    resources: 'Resources',
    team: 'Team',
  },
  footer: {
    tagline: 'High-Concurrency LLM Data Synthesis Engine',
    contact: 'Contact',
  },

  home: {
    hero: {
      title: 'DataForge',
      subtitle: 'High-Concurrency LLM Data Synthesis Engine',
      slogan: 'Provide seed data, and synthesize rich, high-quality, massive datasets for you',
      cta_start: 'Get Started',
      cta_github: 'View on GitHub',
      install_tip: 'Click to copy',
      install_copied: 'Copied!',
    },
    overview: {
      title: 'Overview',
      body: 'DataForge is a production-ready async data synthesis framework that addresses four core challenges in LLM training data synthesis: high-concurrency scheduling, multi-cloud API routing, fault-tolerant recovery, and quality filtering. Through declarative YAML configuration, users can build end-to-end data synthesis pipelines in minutes without managing concurrency or retry logic. DataForge has been validated on models ranging from 1.5B to 72B parameters, achieving 5,000+ RPM throughput on a single machine.',
    },
    highlights: {
      title: 'Key Features',
      items: [
        {
          icon: '⚡',
          title: 'Async Pipeline',
          desc: 'Single-queue N-worker asyncio architecture with 50+ concurrent coroutines. 5,000+ RPM on a single machine. Scale horizontally with Ray or Dask.',
        },
        {
          icon: '🛡️',
          title: 'Fault Tolerance',
          desc: 'WAL-based checkpoint recovery + exponential backoff with full jitter. 99.8% task completion under 30% fault injection. Zero data loss on service interruption.',
        },
        {
          icon: '🔌',
          title: 'Plugin System',
          desc: 'Extend via BaseStrategy / BaseEvaluator in 5 lines. Built-in EvolInstruct, LLMJudge, and RegexFilter. Unified client interface for OpenAI, Anthropic, vLLM, Ollama, and Aliyun.',
        },
      ],
    },
    stats: {
      items: [
        { value: '+3.82%', label: 'Throughput over NaiveAsync (p=0.013)' },
        { value: '99.8%', label: 'Completion under 30% fault injection' },
        { value: '5×', label: 'Eval speedup with eval_max_tokens' },
      ],
    },
    mission: 'Democratizing high-quality LLM training data synthesis for every researcher.',
  },

  research: {
    title: 'Research',
    subtitle: 'We focus on the core challenges of LLM data engineering, advancing the systematic study of data synthesis methodology.',
    areas: [
      {
        icon: '🧪',
        title: 'Data Synthesis',
        desc: 'We study LLM-based automatic instruction data synthesis, including EvolInstruct mutation (constraints, deepening, concretization), Self-Play dialogue expansion, and SeedToQA seed transformation — exploring low-cost, high-diversity synthesis paradigms.',
      },
      {
        icon: '🔍',
        title: 'Data Quality Evaluation',
        desc: 'We explore multi-dimensional quality evaluation systems covering LLM-as-a-Judge scoring, regex filtering, length-window constraints, semantic completeness detection, and similarity-based deduplication to build trustworthy quality assurance pipelines.',
      },
      {
        icon: '⚡',
        title: 'High-Concurrency Scheduling',
        desc: 'We study high-concurrency async scheduling theory for LLM APIs, including dual-bucket rate limiting (RPM+TPM), prefix-aware KV cache scheduling, heterogeneous two-stage gen-eval overlap execution, and exponential backoff with full jitter.',
      },
      {
        icon: '🌐',
        title: 'Distributed Data Processing',
        desc: 'We explore Ray Actor and Dask-based distributed data synthesis frameworks achieving linear multi-machine scaling, studying cross-node checkpoint coordination, load balancing, and task scheduling optimization.',
      },
    ],
  },

  publications: {
    title: 'Publications',
    subtitle: 'We publish data synthesis research at top-tier conferences and journals.',
    placeholder_title: 'Paper Title (Coming Soon)',
    placeholder_authors: 'Author List',
    placeholder_venue: 'Conference / Journal · Year',
    placeholder_abstract: 'Abstract coming soon. Stay tuned.',
    btn_pdf: 'PDF',
    btn_arxiv: 'arXiv',
    btn_bibtex: 'BibTeX',
  },

  news: {
    title: 'News',
    subtitle: 'Project releases and major milestones.',
    items: [
      {
        date: '2026-04-17',
        title: 'DataForge Official Website Launched',
        desc: 'Official website goes live with bilingual support (Chinese/English), covering project overview, research directions, publications, resources, and team.',
      },
      {
        date: '2026-03-20',
        title: 'DataForge v1.0 MVP Released',
        desc: 'v1.0 MVP released with declarative YAML configuration, full CLI commands, and 85% test coverage. Now open source on GitHub.',
      },
      {
        date: '2026-02-15',
        title: 'v0.5 Beta — EvolInstruct & LLMJudge',
        desc: 'Introduced EvolInstruct mutation strategy, LLMJudge scoring evaluator, RegexFilter, and automatic JSON repair mechanism.',
      },
      {
        date: '2026-01-10',
        title: 'v0.1 Alpha — Core Engine',
        desc: 'First Alpha release with async scheduler, unified LLM routing, dual-bucket rate limiter, and WAL checkpoint engine.',
      },
    ],
  },

  resources: {
    title: 'Resources',
    subtitle: 'Datasets, configuration templates, and related tools to get you started quickly.',
    groups: [
      {
        icon: '📦',
        title: 'Datasets',
        items: [
          { name: 'seeds_1k.jsonl', desc: '1,000 seed instruction dataset for quick experiments', link: 'https://github.com/kero-ly/dataforge-ai' },
          { name: 'seeds_10k.jsonl', desc: '10,000 high-quality seed instruction dataset', link: 'https://github.com/kero-ly/dataforge-ai' },
        ],
      },
      {
        icon: '📝',
        title: 'Config Templates',
        items: [
          { name: 'openai_evol.yaml', desc: 'Standard config for OpenAI + EvolInstruct', link: 'https://github.com/kero-ly/dataforge-ai/tree/main/configs' },
          { name: 'vllm_local.yaml', desc: 'Local vLLM server configuration template', link: 'https://github.com/kero-ly/dataforge-ai/tree/main/configs' },
          { name: 'cloud_api.yaml', desc: 'Aliyun Bailian / Anthropic cloud API config', link: 'https://github.com/kero-ly/dataforge-ai/tree/main/configs' },
        ],
      },
      {
        icon: '🔗',
        title: 'Related Projects',
        items: [
          { name: 'vLLM', desc: 'High-performance LLM inference engine', link: 'https://github.com/vllm-project/vllm' },
          { name: 'OpenAI SDK', desc: 'Official Python client for OpenAI', link: 'https://github.com/openai/openai-python' },
          { name: 'Data-Juicer', desc: 'Alibaba open-source data processing toolkit', link: 'https://github.com/modelscope/data-juicer' },
        ],
      },
    ],
  },

  team: {
    title: 'Team',
    subtitle: 'Researchers and engineers from academia and industry driving DataForge forward.',
    members: [
      { name: 'luoyang', role: 'Lead Developer', github: 'https://github.com/kero-ly' },
      { name: 'Contributor', role: 'Research Engineer', github: 'https://github.com/kero-ly/dataforge-ai' },
    ],
    join_title: 'Join Us',
    join_desc: 'We welcome researchers and engineers interested in LLM data engineering to contribute.',
    join_btn: 'Contributing Guide',
  },
}
