<template>
  <footer class="footer">
    <div class="container footer-inner">
      <div class="footer-brand">
        <span class="footer-logo">DataForge</span>
        <p>{{ t('footer.tagline') }}</p>
      </div>
      <div class="footer-links">
        <a href="https://github.com/kero-ly/dataforge-ai" target="_blank" rel="noopener">GitHub</a>
        <RouterLink to="/publications">{{ t('nav.publications') }}</RouterLink>
        <RouterLink to="/resources">{{ t('nav.resources') }}</RouterLink>
        <RouterLink to="/team">{{ t('nav.team') }}</RouterLink>
      </div>
    </div>
    <div class="footer-bottom container">
      <span>© 2026 DataForge Team. MIT License.</span>
      <span>{{ t('footer.contact') }}: <a href="mailto:contact@dataforge.ai">contact@dataforge.ai</a></span>
    </div>
  </footer>
</template>

<script setup>
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
</script>

<style scoped>
.footer {
  background: var(--footer-color);
  color: rgba(255,255,255,0.85);
  padding: 3rem 0 1.5rem;
}

.footer-inner {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 2rem;
  flex-wrap: wrap;
  margin-bottom: 2rem;
}

.footer-logo {
  font-size: 1.3rem;
  font-weight: 700;
  color: #fff;
  display: block;
  margin-bottom: 0.5rem;
}

.footer-brand p {
  font-size: 0.9rem;
  max-width: 280px;
  opacity: 0.8;
}

.footer-links {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  align-items: center;
}

.footer-links a {
  color: rgba(255,255,255,0.8);
  font-size: 0.9rem;
  transition: color 0.2s;
}

.footer-links a:hover {
  color: var(--accent-color);
}

.footer-bottom {
  border-top: 1px solid rgba(255,255,255,0.15);
  padding-top: 1rem;
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.5rem;
  font-size: 0.85rem;
  opacity: 0.7;
}

.footer-bottom a {
  color: var(--accent-color);
}
</style>
