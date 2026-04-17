<template>
  <NavBar />
  <main class="page">
    <RouterView />
  </main>
  <SiteFooter />
</template>

<script setup>
import NavBar from './components/NavBar.vue'
import SiteFooter from './components/SiteFooter.vue'
</script>
