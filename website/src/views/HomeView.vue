<template>
  <div>
    <!-- Hero -->
    <section class="hero">
      <div class="container hero-inner">
        <h1 class="hero-title">{{ t('home.hero.title') }}</h1>
        <h2 class="hero-subtitle">{{ t('home.hero.subtitle') }}</h2>
        <p class="hero-slogan">{{ t('home.hero.slogan') }}</p>

        <div class="hero-badges">
          <a href="https://github.com/kero-ly/dataforge-ai" target="_blank" rel="noopener">
            <img src="https://img.shields.io/github/stars/kero-ly/dataforge-ai?style=social" alt="GitHub Stars" />
          </a>
          <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+" />
          <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License" />
          <img src="https://img.shields.io/badge/version-1.0.0-orange" alt="v1.0.0" />
        </div>

        <div class="hero-actions">
          <RouterLink to="/resources" class="btn btn-primary">{{ t('home.hero.cta_start') }}</RouterLink>
          <a href="https://github.com/kero-ly/dataforge-ai" target="_blank" rel="noopener" class="btn btn-outline">
            {{ t('home.hero.cta_github') }}
          </a>
        </div>

        <div class="install-block" @click="copyInstall">
          <code>pip install dataforge</code>
          <span class="install-hint">{{ copied ? t('home.hero.install_copied') : t('home.hero.install_tip') }}</span>
        </div>
      </div>
    </section>

    <!-- Overview -->
    <section class="section overview-section">
      <div class="container">
        <h2 class="section-title">{{ t('home.overview.title') }}</h2>
        <hr class="divider" />
        <p class="overview-body">{{ t('home.overview.body') }}</p>
      </div>
    </section>

    <!-- Highlights -->
    <section class="section highlights-section">
      <div class="container">
        <h2 class="section-title">{{ t('home.highlights.title') }}</h2>
        <hr class="divider" />
        <div class="grid-3">
          <div v-for="item in t('home.highlights.items')" :key="item.title" class="card">
            <div class="card-header">{{ item.icon }} {{ item.title }}</div>
            <p>{{ item.desc }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- Stats bar -->
    <section class="stats-bar">
      <div class="container stats-inner">
        <div v-for="stat in t('home.stats.items')" :key="stat.value" class="stat-item">
          <span class="stat-value">{{ stat.value }}</span>
          <span class="stat-label">{{ stat.label }}</span>
        </div>
      </div>
    </section>

    <!-- Mission -->
    <section class="section mission-section">
      <div class="container">
        <p class="mission-text">{{ t('home.mission') }}</p>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
const copied = ref(false)

function copyInstall() {
  navigator.clipboard.writeText('pip install dataforge').then(() => {
    copied.value = true
    setTimeout(() => { copied.value = false }, 2000)
  })
}
</script>

<style scoped>
/* Hero */
.hero {
  background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
  color: #fff;
  padding: 5rem 0 4rem;
  text-align: center;
}

.hero-title {
  font-size: 4rem;
  font-weight: 800;
  letter-spacing: -1px;
  margin-bottom: 0.75rem;
}

.hero-subtitle {
  font-size: 1.5rem;
  font-weight: 400;
  opacity: 0.9;
  margin-bottom: 1rem;
}

.hero-slogan {
  font-style: italic;
  font-size: 1.1rem;
  opacity: 0.75;
  margin-bottom: 2rem;
}

.hero-badges {
  display: flex;
  gap: 0.75rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 2rem;
}

.hero-badges img {
  height: 20px;
}

.hero-actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 2rem;
}

.hero .btn-outline {
  color: #fff;
  border-color: rgba(255,255,255,0.6);
}

.hero .btn-outline:hover {
  background: rgba(255,255,255,0.15);
  color: #fff;
}

.install-block {
  display: inline-flex;
  align-items: center;
  gap: 1rem;
  background: rgba(0,0,0,0.25);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 6px;
  padding: 0.6rem 1.25rem;
  cursor: pointer;
  transition: background 0.2s;
  user-select: none;
}

.install-block:hover {
  background: rgba(0,0,0,0.35);
}

.install-block code {
  font-family: 'Courier New', monospace;
  font-size: 0.95rem;
  color: var(--accent-color);
}

.install-hint {
  font-size: 0.8rem;
  opacity: 0.6;
  white-space: nowrap;
}

/* Overview */
.overview-section {
  background: #fff;
}

.overview-body {
  font-size: 1.05rem;
  line-height: 1.8;
  color: #444;
  max-width: 860px;
}

/* Highlights */
.highlights-section {
  background: #f5f7fa;
}

/* Stats bar */
.stats-bar {
  background: var(--primary-color);
  padding: 2.5rem 0;
}

.stats-inner {
  display: flex;
  justify-content: space-around;
  flex-wrap: wrap;
  gap: 2rem;
  text-align: center;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.stat-value {
  font-size: 2.2rem;
  font-weight: 700;
  color: var(--accent-color);
}

.stat-label {
  font-size: 0.9rem;
  color: rgba(255,255,255,0.75);
}

/* Mission */
.mission-section {
  text-align: center;
  background: #fff;
}

.mission-text {
  font-size: 1.4rem;
  font-style: italic;
  color: var(--secondary-color);
  max-width: 700px;
  margin: 0 auto;
  line-height: 1.6;
}

@media (max-width: 600px) {
  .hero-title { font-size: 2.5rem; }
  .hero-subtitle { font-size: 1.1rem; }
}
</style>
