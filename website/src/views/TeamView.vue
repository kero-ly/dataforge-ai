<template>
  <div class="section">
    <div class="container">
      <h1 class="section-title">{{ t('team.title') }}</h1>
      <p class="section-subtitle">{{ t('team.subtitle') }}</p>
      <hr class="divider" />

      <div class="grid-4 team-grid">
        <a
          v-for="member in tm('team.members')"
          :key="member.name"
          :href="member.github"
          target="_blank"
          rel="noopener"
          class="member-card"
        >
          <div class="member-avatar">{{ member.name[0].toUpperCase() }}</div>
          <span class="member-name">{{ member.name }}</span>
          <span class="member-role">{{ member.role }}</span>
        </a>
      </div>

      <div class="join-box">
        <h2 class="join-title">{{ t('team.join_title') }}</h2>
        <p class="join-desc">{{ t('team.join_desc') }}</p>
        <a
          href="https://github.com/kero-ly/dataforge-ai/blob/main/CONTRIBUTING.md"
          target="_blank"
          rel="noopener"
          class="btn btn-primary"
        >{{ t('team.join_btn') }}</a>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useI18n } from 'vue-i18n'
const { t, tm } = useI18n()
</script>

<style scoped>
.team-grid {
  margin-bottom: 3rem;
}

.member-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.6rem;
  padding: 2rem 1rem;
  background: var(--card-bg);
  border-radius: 10px;
  border: 1px solid #e8e8e8;
  text-decoration: none;
  transition: box-shadow 0.2s, transform 0.2s;
}

.member-card:hover {
  box-shadow: 0 6px 20px rgba(0,0,0,0.1);
  transform: translateY(-4px);
}

.member-avatar {
  width: 72px;
  height: 72px;
  border-radius: 50%;
  background: var(--primary-color);
  color: #fff;
  font-size: 1.8rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
}

.member-name {
  font-weight: 600;
  color: var(--primary-color);
  font-size: 1rem;
}

.member-role {
  font-size: 0.85rem;
  color: #777;
}

.join-box {
  background: #f5f7fa;
  border-radius: 10px;
  padding: 2.5rem;
  text-align: center;
  border: 1px solid #e0e0e0;
}

.join-title {
  font-size: 1.4rem;
  color: var(--primary-color);
  font-weight: 700;
  margin-bottom: 0.6rem;
}

.join-desc {
  color: #555;
  margin-bottom: 1.5rem;
}
</style>
