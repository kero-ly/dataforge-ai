<template>
  <nav class="navbar">
    <div class="nav-inner container">
      <RouterLink to="/" class="nav-logo">DataForge</RouterLink>

      <button class="nav-toggle" @click="menuOpen = !menuOpen" aria-label="toggle menu">
        <span></span><span></span><span></span>
      </button>

      <ul class="nav-links" :class="{ open: menuOpen }">
        <li v-for="link in links" :key="link.path">
          <RouterLink :to="link.path" @click="menuOpen = false">{{ t(link.label) }}</RouterLink>
        </li>
      </ul>

      <button class="lang-btn" @click="toggleLang">
        {{ locale === 'zh' ? 'EN' : '中文' }}
      </button>
    </div>
  </nav>
</template>

<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

const { t, locale } = useI18n()
const menuOpen = ref(false)

const links = [
  { path: '/',             label: 'nav.home' },
  { path: '/research',     label: 'nav.research' },
  { path: '/publications', label: 'nav.publications' },
  { path: '/news',         label: 'nav.news' },
  { path: '/resources',    label: 'nav.resources' },
  { path: '/team',         label: 'nav.team' },
]

function toggleLang() {
  locale.value = locale.value === 'zh' ? 'en' : 'zh'
  localStorage.setItem('df-lang', locale.value)
}
</script>

<style scoped>
.navbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: var(--nav-height);
  background: var(--primary-color);
  z-index: 1000;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.nav-inner {
  display: flex;
  align-items: center;
  height: 100%;
  gap: 1rem;
}

.nav-logo {
  font-size: 1.4rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

.nav-links {
  display: flex;
  list-style: none;
  gap: 0.25rem;
  flex: 1;
  justify-content: center;
}

.nav-links a {
  color: rgba(255,255,255,0.88);
  padding: 0.4rem 0.9rem;
  border-radius: 4px;
  font-size: 0.95rem;
  transition: color 0.2s, background 0.2s;
}

.nav-links a:hover,
.nav-links a.router-link-active {
  color: var(--accent-color);
  background: rgba(255,255,255,0.08);
}

.lang-btn {
  background: rgba(255,255,255,0.15);
  color: #fff;
  border: 1px solid rgba(255,255,255,0.3);
  padding: 0.3rem 0.8rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85rem;
  white-space: nowrap;
  transition: background 0.2s;
}

.lang-btn:hover {
  background: rgba(255,255,255,0.25);
}

.nav-toggle {
  display: none;
  flex-direction: column;
  gap: 5px;
  background: none;
  border: none;
  cursor: pointer;
  padding: 0.25rem;
  margin-left: auto;
}

.nav-toggle span {
  display: block;
  width: 24px;
  height: 2px;
  background: #fff;
  border-radius: 2px;
}

@media (max-width: 768px) {
  .nav-toggle { display: flex; }

  .nav-links {
    display: none;
    position: absolute;
    top: var(--nav-height);
    left: 0;
    right: 0;
    background: var(--primary-color);
    flex-direction: column;
    padding: 1rem 2rem;
    gap: 0.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  }

  .nav-links.open { display: flex; }
}
</style>
