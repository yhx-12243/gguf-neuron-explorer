// 画面の言語(日本語 / English / 中文)。文言は日本語の文をそのまま鍵にし、英語・中国語は EN / ZH の辞書から引く(無ければ日本語のまま)。
// 差し込む値は {n} のような目印で渡す: t("{n} 層", { n: 32 })。英語の単数・複数は {n|layer|layers}(n が 1 なら前)。
// 言語はこのブラウザの localStorage に覚え、切り替えたらページを読み直す(開いているニューロンは URL の # に残る)。
// サーバが返すエラーや警告の文(GneError など)は日本語のまま。
import { EN } from "./i18n_en.js";
import { ZH } from "./i18n_zh.js";

const KEY = "gne.lang";

// URL の ?lang= → このブラウザに覚えた言語 → ブラウザの言語、の順
function initial() {
  const q = new URLSearchParams(location.search).get("lang")?.toLowerCase();
  if (q === "ja" || q === "en" || q === "zh") {
    try { localStorage.setItem(KEY, q); } catch { /* 覚えられなくても URL で決まる */ }
    return q;
  }
  try {
    const v = localStorage.getItem(KEY).toLowerCase();
    if (v === "ja" || v === "en" || v === "zh") return v;
  } catch { /* 読めなければ既定 */ }
  const v = (navigator.language ?? "").toLowerCase().split("-")[0];
  return v === "ja" || v === "zh" ? v : "en";
}

export const LANG = initial();
document.documentElement.lang = LANG;

// 翻訳辞書に無かった日本語の文(テストと確認用、window.gneI18n.missing)
const missing = new Set();
const JA = /[ぁ-んァ-ヶ一-龥]/;
window.gneI18n = { lang: LANG, missing };

export function t(ja, vars) {
  let s = ja;
  if (LANG === "en") {
    if (EN[ja] !== undefined) s = EN[ja];
    else if (JA.test(ja)) missing.add(ja);
  } else if (LANG === "zh") {
    if (ZH[ja] !== undefined) s = ZH[ja];
    else if (JA.test(ja)) missing.add(ja);
  }
  if (vars) {
    s = s.replace(/\{(\w+)\|([^|}]*)\|([^}]*)\}/g, (m, k, one, other) => (vars[k] === undefined ? m : String(vars[k]) === "1" ? one : other));
    s = s.replace(/\{(\w+)\}/g, (m, k) => (vars[k] === undefined ? m : String(vars[k])));
  }
  return s;
}

export function setLang(lang) {
  try { localStorage.setItem(KEY, lang); } catch { /* 覚えられなくても今回は切り替える */ }
  const u = new URL(location.href);
  u.searchParams.set("lang", lang);
  location.replace(u.toString());
}

// index.html の静的な文言: data-i18n(中身)、data-i18n-title、data-i18n-placeholder
export function translateDOM(root = document) {
  for (const el of root.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of root.querySelectorAll("[data-i18n-title]")) el.title = t(el.dataset.i18nTitle);
  for (const el of root.querySelectorAll("[data-i18n-placeholder]")) el.placeholder = t(el.dataset.i18nPlaceholder);
}
