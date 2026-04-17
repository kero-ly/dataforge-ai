import { createI18n } from 'vue-i18n'
import zh from './zh.js'
import en from './en.js'

const savedLang = localStorage.getItem('df-lang') || 'zh'

export default createI18n({
  legacy: false,
  locale: savedLang,
  fallbackLocale: 'en',
  messages: { zh, en },
})
