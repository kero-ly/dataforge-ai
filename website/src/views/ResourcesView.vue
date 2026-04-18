<template>
  <div class="section">
    <div class="container">
      <h1 class="section-title">{{ t('resources.title') }}</h1>
      <p class="section-subtitle">{{ t('resources.subtitle') }}</p>
      <hr class="divider" />

      <div class="resource-groups">
        <div v-for="group in tm('resources.groups')" :key="group.title" class="resource-group">
          <h2 class="group-title">{{ group.icon }} {{ group.title }}</h2>
          <div class="resource-cards">
            <a
              v-for="item in group.items"
              :key="item.name"
              :href="item.link"
              target="_blank"
              rel="noopener"
              class="resource-card"
            >
              <span class="resource-name">{{ item.name }}</span>
              <span class="resource-desc">{{ item.desc }}</span>
            </a>
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
.resource-groups {
  display: flex;
  flex-direction: column;
  gap: 2.5rem;
}

.group-title {
  font-size: 1.3rem;
  color: var(--primary-color);
  font-weight: 600;
  margin-bottom: 1rem;
}

.resource-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 1rem;
}

.resource-card {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  padding: 1.25rem 1.5rem;
  background: var(--card-bg);
  border-radius: 8px;
  border: 1px solid #e8e8e8;
  text-decoration: none;
  transition: box-shadow 0.2s, transform 0.2s;
}

.resource-card:hover {
  box-shadow: 0 4px 16px rgba(0,0,0,0.1);
  transform: translateY(-3px);
}

.resource-name {
  font-weight: 600;
  color: var(--secondary-color);
  font-size: 0.95rem;
}

.resource-desc {
  font-size: 0.85rem;
  color: #666;
  line-height: 1.5;
}
</style>
