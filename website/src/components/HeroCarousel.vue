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
      <!--
        activationCounts[i] increments each time slide i becomes active.
        Changing :key forces Vue to unmount+remount this element,
        which restarts the CSS animation from scratch (no JS animation needed).
      -->
      <div
        :key="activationCounts[i]"
        class="slide-bg"
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
        <a
          href="https://github.com/kero-ly/dataforge-ai"
          target="_blank"
          rel="noopener"
          class="btn btn-ghost"
        >
          <!-- GitHub Octocat icon -->
          <svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" style="vertical-align:-3px;margin-right:6px">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
              0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
              -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
              .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
              -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27
              .68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
              .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
              0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
          </svg>
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

const images = [img1, img2, img3, img4]
const current = ref(0)
const leaving = ref(-1)
const copied  = ref(false)

// Each entry tracks how many times slide i has been activated.
// Changing the value causes Vue to remount the .slide-bg element,
// which restarts the CSS kenburns animation automatically.
const activationCounts = ref(images.map(() => 0))

const SLIDE_MS = 6000
const FADE_MS  = 1200

let leaveTimer = null
let slideTimer = null

function activate(index) {
  activationCounts.value[index]++
}

function advance() {
  const next = (current.value + 1) % images.length
  leaving.value = current.value
  current.value  = next
  activate(next)
  clearTimeout(leaveTimer)
  leaveTimer = setTimeout(() => { leaving.value = -1 }, FADE_MS)
}

function goTo(index) {
  if (index === current.value) return
  clearInterval(slideTimer)
  leaving.value = current.value
  current.value  = index
  activate(index)
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
  // slide 0 starts active; its CSS animation fires automatically
  // because .slide--active .slide-bg { animation: kenburns ... }
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
  margin-top: calc(-1 * var(--nav-height));
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
.slide--leaving { opacity: 0; z-index: 2; }

/* ─── Background image ────────────────────────── */
.slide-bg {
  position: absolute;
  inset: 0;
  background-size: cover;
  background-position: center center;
  background-repeat: no-repeat;
  transform-origin: center center;
  will-change: transform;
  /*
   * Animation always declared here so the browser tracks progress.
   * Default: paused — freezes at current position (no snap-back on leave).
   * Active:  running — plays forward from current position.
   * On remount (activationCounts key change): restarts from scale(1).
   */
  animation: kenburns 7200ms linear forwards;
  animation-play-state: paused;
}

.slide--active .slide-bg {
  animation-play-state: running;
}

/* ─── Dark overlay ────────────────────────────── */
.overlay {
  position: absolute;
  inset: 0;
  z-index: 3;
  background: linear-gradient(
    180deg,
    rgba(0, 0, 0, 0.38) 0%,
    rgba(0, 10, 30, 0.52) 60%,
    rgba(0, 10, 30, 0.68) 100%
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
  text-shadow: 0 2px 16px rgba(0,0,0,0.5);
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
  display: inline-flex;
  align-items: center;
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
  0%, 100% { transform: rotate(45deg) translateY(0);   opacity: 0.5; }
  50%       { transform: rotate(45deg) translateY(6px); opacity: 1; }
}

/* ─── Responsive ──────────────────────────────── */
@media (max-width: 600px) {
  .hero-badges  { display: none; }
  .install-hint { display: none; }
  .carousel-dots { bottom: 2rem; }
}
</style>
