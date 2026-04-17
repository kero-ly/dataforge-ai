import { createApp } from 'vue'
import App from './App.vue'
import router from './router/index.js'
import i18n from './i18n/index.js'
import './assets/main.css'

createApp(App).use(router).use(i18n).mount('#app')
