<template>
  <div class="section">
    <div class="container">
      <h1 class="section-title">{{ t('research.title') }}</h1>
      <p class="section-subtitle">{{ t('research.subtitle') }}</p>
      <hr class="divider" />
      <div class="grid-2 research-grid">
        <div v-for="area in tm('research.areas')" :key="area.title" class="card research-card">
          <div class="card-header">{{ area.icon }} {{ area.title }}</div>
          <p>{{ area.desc }}</p>
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
.research-grid {
  margin-top: 1rem;
}

.research-card p {
  line-height: 1.75;
  color: #555;
}
</style>
