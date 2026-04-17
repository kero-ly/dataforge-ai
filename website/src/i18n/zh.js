export default {
  nav: {
    home: '首页',
    research: '研究',
    publications: '论文',
    news: '动态',
    resources: '资源',
    team: '团队',
  },
  footer: {
    tagline: '面向 LLM 时代的高并发数据合成引擎',
    contact: '联系我们',
  },

  home: {
    hero: {
      title: 'DataForge',
      subtitle: '面向 LLM 时代的高并发数据合成引擎',
      slogan: '"炼油厂" — 将原始种子数据提炼为高质量 LLM 训练集',
      cta_start: '快速开始',
      cta_github: '查看源码',
      install_tip: '点击复制',
      install_copied: '已复制！',
    },
    overview: {
      title: '项目简介',
      body: 'DataForge 是一个生产级异步数据合成框架，旨在解决 LLM 训练数据合成中的高并发调度、多云 API 路由、容错恢复与质量过滤四大核心问题。通过声明式 YAML 配置，用户可在分钟级内构建端到端数据合成流水线，无需关心底层并发控制与错误重试逻辑。DataForge 已在 7B、14B、72B 等多个规模的模型上完成验证，单机吞吐可达 5000+ rpm。',
    },
    highlights: {
      title: '核心特性',
      items: [
        {
          icon: '⚡',
          title: '异步高并发管道',
          desc: '基于 asyncio 的单队列多 worker 架构，50+ 并发协程，单机 5000+ rpm，支持 Ray/Dask 横向扩展。',
        },
        {
          icon: '🛡️',
          title: '容错与断点恢复',
          desc: 'WAL 日志追加 + 指数退避重试，30% 故障注入下仍保持 99.8% 任务完成率，服务中断后零数据丢失续跑。',
        },
        {
          icon: '🔌',
          title: '插件化扩展',
          desc: '继承 BaseStrategy / BaseEvaluator，5 行代码接入自定义合成策略或质量评估器，内置 EvolInstruct、LLMJudge、RegexFilter。',
        },
      ],
    },
    stats: {
      items: [
        { value: '+3.82%', label: '吞吐量领先 NaiveAsync (p=0.013)' },
        { value: '99.8%', label: '30% 故障注入下任务完成率' },
        { value: '5×', label: 'eval_max_tokens 评估加速比' },
      ],
    },
    mission: '让每个研究者都能便捷地合成高质量 LLM 训练数据。',
  },

  research: {
    title: '研究方向',
    subtitle: '我们聚焦于 LLM 数据工程的核心挑战，推动数据合成方法论的系统化发展。',
    areas: [
      {
        icon: '🧪',
        title: '数据合成',
        desc: '研究基于大语言模型的指令数据自动合成方法，包括 EvolInstruct 指令变异（约束化、深化、具体化）、Self-Play 对话扩充、SeedToQA 种子转化等策略，探索低成本高多样性的合成范式。',
      },
      {
        icon: '🔍',
        title: '数据质量评估',
        desc: '探索多维度数据质量评估体系，涵盖 LLM-as-a-Judge 自动评分、正则过滤、长度窗口约束、语义完整性检测与相似度去重，构建可信赖的合成数据质量保证链路。',
      },
      {
        icon: '⚡',
        title: '高并发调度',
        desc: '研究面向 LLM API 的高并发异步调度理论，包括双桶限流（RPM+TPM）、前缀感知 KV 缓存调度、异质两阶段生成-评估重叠执行，以及指数退避抖动容错机制。',
      },
      {
        icon: '🌐',
        title: '分布式数据处理',
        desc: '探索基于 Ray Actor 与 Dask 的分布式数据合成执行框架，实现多机多卡线性扩展，研究跨节点检查点协调、负载均衡与任务调度优化。',
      },
    ],
  },

  publications: {
    title: '论文发表',
    subtitle: '我们在顶级会议与期刊发表数据合成相关研究成果。',
    placeholder_title: '论文标题（即将发布）',
    placeholder_authors: '作者列表',
    placeholder_venue: 'Conference / Journal · Year',
    placeholder_abstract: '摘要即将公开，敬请期待。',
    btn_pdf: 'PDF',
    btn_arxiv: 'arXiv',
    btn_bibtex: 'BibTeX',
  },

  news: {
    title: '最新动态',
    subtitle: '项目版本发布与重要里程碑。',
    items: [
      {
        date: '2026-04-17',
        title: 'DataForge 官网正式上线',
        desc: '官方网站上线，支持中英双语，涵盖项目介绍、研究方向、论文发表、资源下载与团队信息。',
      },
      {
        date: '2026-03-20',
        title: 'DataForge v1.0 MVP 正式发布',
        desc: '发布 v1.0 MVP 版本，支持 YAML 声明式配置、完整 CLI 命令、85% 测试覆盖率，开源于 GitHub。',
      },
      {
        date: '2026-02-15',
        title: 'v0.5 Beta — EvolInstruct & LLMJudge',
        desc: '引入 EvolInstruct 指令变异策略、LLMJudge 评分评估器、RegexFilter 过滤器，以及 JSON 自动修复机制。',
      },
      {
        date: '2026-01-10',
        title: 'v0.1 Alpha — 核心引擎',
        desc: '发布首个 Alpha 版本，包含异步调度器、统一 LLM 路由、双桶限流器和 WAL 断点续传引擎。',
      },
    ],
  },

  resources: {
    title: '资源下载',
    subtitle: '数据集、配置模板与相关工具，助力快速上手。',
    groups: [
      {
        icon: '📦',
        title: '数据集',
        items: [
          { name: 'seeds_1k.jsonl', desc: '1000 条种子指令数据集，适用于快速实验', link: 'https://github.com/kero-ly/dataforge-ai' },
          { name: 'seeds_10k.jsonl', desc: '10000 条高质量种子数据集', link: 'https://github.com/kero-ly/dataforge-ai' },
        ],
      },
      {
        icon: '📝',
        title: '配置模板',
        items: [
          { name: 'openai_evol.yaml', desc: 'OpenAI + EvolInstruct 标准配置', link: 'https://github.com/kero-ly/dataforge-ai/tree/main/configs' },
          { name: 'vllm_local.yaml', desc: '本地 vLLM 服务端配置模板', link: 'https://github.com/kero-ly/dataforge-ai/tree/main/configs' },
          { name: 'cloud_api.yaml', desc: '阿里云百炼 / Anthropic 云端配置', link: 'https://github.com/kero-ly/dataforge-ai/tree/main/configs' },
        ],
      },
      {
        icon: '🔗',
        title: '相关项目',
        items: [
          { name: 'vLLM', desc: '高性能 LLM 推理引擎', link: 'https://github.com/vllm-project/vllm' },
          { name: 'OpenAI SDK', desc: '官方 Python 客户端', link: 'https://github.com/openai/openai-python' },
          { name: 'Data-Juicer', desc: '阿里巴巴开源数据处理工具集', link: 'https://github.com/modelscope/data-juicer' },
        ],
      },
    ],
  },

  team: {
    title: '团队成员',
    subtitle: '来自学术界与工业界的研究者共同推动 DataForge 的发展。',
    members: [
      { name: 'luoyang', role: 'Lead Developer', github: 'https://github.com/kero-ly' },
      { name: 'Contributor', role: 'Research Engineer', github: 'https://github.com/kero-ly/dataforge-ai' },
    ],
    join_title: '加入我们',
    join_desc: '我们欢迎对 LLM 数据工程感兴趣的研究者与工程师参与贡献。',
    join_btn: '查看贡献指南',
  },
}
