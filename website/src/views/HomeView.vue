<template>
  <div>
    <!-- Full-screen carousel hero -->
    <HeroCarousel />

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
import { useI18n } from 'vue-i18n'
import HeroCarousel from '../components/HeroCarousel.vue'

const { t } = useI18n()
</script>

<style scoped>
/* Overview */
.overview-section { background: #fff; }

.overview-body {
  font-size: 1.05rem;
  line-height: 1.8;
  color: #444;
  max-width: 860px;
  white-space: pre-line;
}

/* Highlights */
.highlights-section { background: #f5f7fa; }

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
</style>
