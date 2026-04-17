<template>
  <div class="section">
    <div class="container">
      <h1 class="section-title">{{ t('publications.title') }}</h1>
      <p class="section-subtitle">{{ t('publications.subtitle') }}</p>
      <hr class="divider" />

      <div class="pub-list">
        <div v-for="i in 3" :key="i" class="pub-card">
          <div class="pub-venue">
            <span class="tag">{{ t('publications.placeholder_venue') }}</span>
          </div>
          <h3 class="pub-title">{{ t('publications.placeholder_title') }}</h3>
          <p class="pub-authors">{{ t('publications.placeholder_authors') }}</p>
          <p class="pub-abstract">{{ t('publications.placeholder_abstract') }}</p>
          <div class="pub-actions">
            <a href="#" class="btn btn-outline btn-sm">{{ t('publications.btn_pdf') }}</a>
            <a href="#" class="btn btn-outline btn-sm">{{ t('publications.btn_arxiv') }}</a>
            <a href="#" class="btn btn-outline btn-sm">{{ t('publications.btn_bibtex') }}</a>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
</script>

<style scoped>
.pub-list {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.pub-card {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 1.75rem;
  background: #fff;
  transition: box-shadow 0.2s;
}

.pub-card:hover {
  box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}

.pub-venue {
  margin-bottom: 0.75rem;
}

.pub-title {
  font-size: 1.2rem;
  color: var(--primary-color);
  font-weight: 600;
  margin-bottom: 0.4rem;
}

.pub-authors {
  font-size: 0.9rem;
  color: #666;
  margin-bottom: 0.75rem;
}

.pub-abstract {
  color: #555;
  font-size: 0.95rem;
  line-height: 1.65;
  margin-bottom: 1.25rem;
}

.pub-actions {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
}
</style>
