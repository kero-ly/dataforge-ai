<template>
  <section class="hero-carousel" aria-label="hero">
    <!-- Slide layers -->
    <div
      v-for="(img, i) in images"
      :key="i"
      class="slide"
      :class="{
        'slide--active':  i === current,
        'slide--leaving': i === leaving,
      }"
    >
      <div
        class="slide-bg"
        :ref="el => { if (el) bgRefs[i] = el }"
        :style="{ backgroundImage: `url(${img})` }"
      />
    </div>

    <!-- Dark gradient overlay -->
    <div class="overlay" />

    <!-- Hero content -->
    <div class="hero-content container">
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
        <a href="https://github.com/kero-ly/dataforge-ai" target="_blank" rel="noopener" class="btn btn-ghost">
          {{ t('home.hero.cta_github') }}
        </a>
      </div>

      <div class="install-block" @click="copyInstall">
        <code>pip install dataforge</code>
        <span class="install-hint">{{ copied ? t('home.hero.install_copied') : t('home.hero.install_tip') }}</span>
      </div>
    </div>

    <!-- Dot navigation -->
    <div class="carousel-dots">
      <button
        v-for="(_, i) in images"
        :key="i"
        class="dot"
        :class="{ 'dot--active': i === current }"
        :aria-label="`Slide ${i + 1}`"
        @click="goTo(i)"
      />
    </div>

    <!-- Scroll hint -->
    <div class="scroll-hint">
      <span class="scroll-arrow" />
    </div>
  </section>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import img1 from '../assets/1.jpg'
import img2 from '../assets/2.jpg'
import img3 from '../assets/3.jpg'
import img4 from '../assets/4.jpg'

const { t } = useI18n()

const images  = [img1, img2, img3, img4]
const current = ref(0)
const leaving = ref(-1)
const bgRefs  = ref([])
const copied  = ref(false)

const SLIDE_MS      = 6000   // how long each slide shows
const FADE_MS       = 1200   // crossfade duration (must match CSS)
const KENBURNS_MS   = SLIDE_MS + FADE_MS   // zoom over full visible window

function startKenBurns(index) {
  const el = bgRefs.value[index]
  if (!el) return
  el.style.animation = 'none'
  void el.offsetWidth            // force reflow → restarts animation
  el.style.animation = `kenburns ${KENBURNS_MS}ms linear forwards`
}

let leaveTimer = null
let slideTimer = null

function advance() {
  const next = (current.value + 1) % images.length
  leaving.value = current.value
  current.value  = next
  startKenBurns(next)

  clearTimeout(leaveTimer)
  leaveTimer = setTimeout(() => { leaving.value = -1 }, FADE_MS)
}

function goTo(index) {
  if (index === current.value) return
  clearInterval(slideTimer)
  leaving.value = current.value
  current.value  = index
  startKenBurns(index)
  clearTimeout(leaveTimer)
  leaveTimer = setTimeout(() => { leaving.value = -1 }, FADE_MS)
  slideTimer = setInterval(advance, SLIDE_MS)
}

function copyInstall() {
  navigator.clipboard.writeText('pip install dataforge').then(() => {
    copied.value = true
    setTimeout(() => { copied.value = false }, 2000)
  })
}

onMounted(() => {
  startKenBurns(0)
  slideTimer = setInterval(advance, SLIDE_MS)
})

onUnmounted(() => {
  clearInterval(slideTimer)
  clearTimeout(leaveTimer)
})
</script>

<style scoped>
/* ─── Container ───────────────────────────────── */
.hero-carousel {
  position: relative;
  width: 100%;
  height: 100vh;
  min-height: 500px;
  overflow: hidden;
}

/* ─── Slide layer ─────────────────────────────── */
.slide {
  position: absolute;
  inset: 0;
  opacity: 0;
  z-index: 0;
  transition: opacity 1200ms ease-in-out;
  will-change: opacity;
}

.slide--active  { opacity: 1; z-index: 1; }
.slide--leaving { opacity: 0; z-index: 2; }  /* sits above incoming, fades out */

/* ─── Background image ────────────────────────── */
.slide-bg {
  position: absolute;
  inset: 0;
  background-size: cover;
  background-position: center center;
  background-repeat: no-repeat;
  transform-origin: center center;
  will-change: transform;
}

@keyframes kenburns {
  from { transform: scale(1);    }
  to   { transform: scale(1.35); }
}

/* ─── Dark overlay ────────────────────────────── */
.overlay {
  position: absolute;
  inset: 0;
  z-index: 3;
  background: linear-gradient(
    180deg,
    rgba(0, 0, 0, 0.42) 0%,
    rgba(0, 10, 30, 0.55) 60%,
    rgba(0, 10, 30, 0.72) 100%
  );
}

/* ─── Hero content ────────────────────────────── */
.hero-content {
  position: absolute;
  inset: 0;
  z-index: 4;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 2rem;
  color: #fff;
}

.hero-title {
  font-size: clamp(2.8rem, 7vw, 5.5rem);
  font-weight: 800;
  letter-spacing: -1px;
  margin-bottom: 0.6rem;
  text-shadow: 0 2px 12px rgba(0,0,0,0.4);
}

.hero-subtitle {
  font-size: clamp(1rem, 2.5vw, 1.5rem);
  font-weight: 400;
  opacity: 0.92;
  margin-bottom: 0.75rem;
  max-width: 680px;
}

.hero-slogan {
  font-style: italic;
  font-size: clamp(0.9rem, 1.8vw, 1.1rem);
  opacity: 0.75;
  margin-bottom: 1.75rem;
}

.hero-badges {
  display: flex;
  gap: 0.6rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 1.75rem;
}

.hero-badges img { height: 20px; }

.hero-actions {
  display: flex;
  gap: 0.9rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 1.5rem;
}

.btn-ghost {
  display: inline-block;
  padding: 0.6rem 1.4rem;
  border-radius: 4px;
  font-size: 0.95rem;
  font-weight: 500;
  color: #fff;
  border: 2px solid rgba(255,255,255,0.6);
  transition: background 0.2s;
  cursor: pointer;
}

.btn-ghost:hover { background: rgba(255,255,255,0.15); color: #fff; }

.install-block {
  display: inline-flex;
  align-items: center;
  gap: 1rem;
  background: rgba(0,0,0,0.35);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 6px;
  padding: 0.55rem 1.2rem;
  cursor: pointer;
  transition: background 0.2s;
  user-select: none;
  backdrop-filter: blur(6px);
}

.install-block:hover { background: rgba(0,0,0,0.5); }

.install-block code {
  font-family: 'Courier New', monospace;
  font-size: 0.92rem;
  color: var(--accent-color);
}

.install-hint {
  font-size: 0.78rem;
  opacity: 0.55;
  white-space: nowrap;
}

/* ─── Dot navigation ──────────────────────────── */
.carousel-dots {
  position: absolute;
  bottom: 3rem;
  left: 50%;
  transform: translateX(-50%);
  z-index: 5;
  display: flex;
  gap: 0.6rem;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  border: 2px solid rgba(255,255,255,0.6);
  background: transparent;
  cursor: pointer;
  transition: background 0.3s, transform 0.3s;
  padding: 0;
}

.dot--active {
  background: #fff;
  transform: scale(1.3);
}

/* ─── Scroll hint ─────────────────────────────── */
.scroll-hint {
  position: absolute;
  bottom: 1.2rem;
  left: 50%;
  transform: translateX(-50%);
  z-index: 5;
}

.scroll-arrow {
  display: block;
  width: 20px;
  height: 20px;
  border-right: 2px solid rgba(255,255,255,0.5);
  border-bottom: 2px solid rgba(255,255,255,0.5);
  transform: rotate(45deg);
  animation: bounce 1.6s ease-in-out infinite;
}

@keyframes bounce {
  0%, 100% { transform: rotate(45deg) translateY(0);    opacity: 0.5; }
  50%       { transform: rotate(45deg) translateY(6px);  opacity: 1; }
}

/* ─── Responsive ──────────────────────────────── */
@media (max-width: 600px) {
  .hero-badges { display: none; }
  .install-hint { display: none; }
  .carousel-dots { bottom: 2rem; }
}
</style>
