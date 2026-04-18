<template>
  <div class="section">
    <div class="container">
      <h1 class="section-title">{{ t('news.title') }}</h1>
      <p class="section-subtitle">{{ t('news.subtitle') }}</p>
      <hr class="divider" />

      <div class="timeline">
        <div v-for="item in tm('news.items')" :key="item.date" class="timeline-item">
          <div class="timeline-dot"></div>
          <div class="timeline-content">
            <span class="timeline-date">{{ item.date }}</span>
            <h3 class="timeline-title">{{ item.title }}</h3>
            <p class="timeline-desc">{{ item.desc }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useI18n } from 'vue-i18n'
const { t, tm } = useI18n()
</script>

<style scoped>
.timeline {
  position: relative;
  padding-left: 2.5rem;
  margin-top: 1rem;
}

.timeline::before {
  content: '';
  position: absolute;
  left: 0.55rem;
  top: 0;
  bottom: 0;
  width: 2px;
  background: var(--accent-color);
}

.timeline-item {
  position: relative;
  margin-bottom: 2.5rem;
}

.timeline-dot {
  position: absolute;
  left: -2.5rem;
  top: 0.35rem;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--primary-color);
  border: 3px solid var(--accent-color);
}

.timeline-date {
  font-size: 0.85rem;
  color: #888;
  display: block;
  margin-bottom: 0.25rem;
}

.timeline-title {
  font-size: 1.1rem;
  color: var(--primary-color);
  font-weight: 600;
  margin-bottom: 0.4rem;
}

.timeline-desc {
  color: #555;
  line-height: 1.65;
}
</style>
