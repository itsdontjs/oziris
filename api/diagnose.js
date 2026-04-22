import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';

const FETCH_TIMEOUT  = 8000;
const MAX_HTML_CHARS = 15000;

// ── Scraping helpers ─────────────────────────────────────────────────────────

async function fetchWithTimeout(url, timeout = FETCH_TIMEOUT) {
  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
      },
    });
    return (await res.text()).slice(0, MAX_HTML_CHARS);
  } catch { return null; }
  finally { clearTimeout(tid); }
}

function extractWebsiteData(html, url) {
  if (!html) return null;
  html = html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
    .replace(/<noscript[^>]*>[\s\S]*?<\/noscript>/gi, '');

  const get = rx => (html.match(rx) || [])[1]?.trim() || '';
  const title   = get(/<title[^>]*>([^<]+)<\/title>/i);
  const desc    = get(/<meta[^>]*name=["']description["'][^>]*content=["']([^"']{1,200})["']/i)
               || get(/<meta[^>]*content=["']([^"']{1,200})["'][^>]*name=["']description["']/i);
  const ogTitle = get(/<meta[^>]*property=["']og:title["'][^>]*content=["']([^"']+)["']/i);
  const ogDesc  = get(/<meta[^>]*property=["']og:description["'][^>]*content=["']([^"']{1,200})["']/i);
  const h1s = [...html.matchAll(/<h1[^>]*>([^<]{3,80})<\/h1>/gi)].map(m => m[1].trim()).slice(0, 2);
  const h2s = [...html.matchAll(/<h2[^>]*>([^<]{3,80})<\/h2>/gi)].map(m => m[1].trim()).slice(0, 4);
  const ctaMatches = [...html.matchAll(/<(?:button|a)[^>]*>([^<]{4,60})<\/(?:button|a)>/gi)]
    .map(m => m[1].replace(/<[^>]+>/g, '').trim())
    .filter(t => /comprar|contratar|agendar|entrar|fale|contato|whatsapp|grátis|gratuito|começar|acesso|clique|saiba/i.test(t))
    .slice(0, 5);
  return {
    url,
    title: title || ogTitle || '',
    description: desc || ogDesc || '',
    headings: [...h1s, ...h2s].filter(Boolean).slice(0, 5),
    ctas: ctaMatches,
    tracking: {
      ga:       /gtag\(|analytics\.js|google-analytics|googletagmanager/.test(html),
      fbPixel:  /fbq\(|facebook\.net\/en_US\/fbevents/.test(html),
      gtm:      /googletagmanager\.com\/gtm/.test(html),
      whatsapp: /whatsapp|wa\.me/.test(html),
      chatbot:  /intercom|drift\.com|hubspot|tawk\.to|crisp\.chat/.test(html),
    },
  };
}

function normalizeUrl(raw) {
  if (!raw) return null;
  raw = raw.trim();
  return /^https?:\/\//i.test(raw) ? raw : 'https://' + raw;
}

function normalizeInstagram(raw) {
  if (!raw) return null;
  const clean = raw.replace(/https?:\/\/(?:www\.)?instagram\.com\/?/, '').replace(/\//g, '').replace('@', '').trim();
  return clean ? `https://www.instagram.com/${clean}/` : null;
}

function buildDigitalCtx(digital) {
  let ctx = '';
  if (digital.website) {
    const w = digital.website;
    ctx += `\n=== SITE (${w.url}) ===
Título: ${w.title || 'Não encontrado'}
Descrição/Meta: ${w.description || 'Não encontrado'}
Headings principais: ${w.headings?.join(' | ') || 'Nenhum claro'}
CTAs encontradas: ${w.ctas?.join(', ') || 'Nenhuma clara identificada'}
Rastreamento: GA=${w.tracking.ga}, FacebookPixel=${w.tracking.fbPixel}, GTM=${w.tracking.gtm}, WhatsApp=${w.tracking.whatsapp}, Chatbot=${w.tracking.chatbot}`;
  }
  if (digital.instagram) {
    const ig = digital.instagram;
    ctx += `\n=== INSTAGRAM ===\nNome/Perfil: ${ig.title || 'Não encontrado'}\nBio: ${ig.description || 'Não encontrado'}`;
  }
  return ctx;
}

// ── Motor 1: Claude — Diagnóstico principal ──────────────────────────────────

async function runClaude({ seg, budget, canal, pain, impact, digitalCtx }) {
  const hasDigital = digitalCtx.trim().length > 0;
  const prompt = `Você é o Oziris, especialista sênior em marketing digital e automação para negócios brasileiros.

DADOS COMPLETOS DO LEAD (funil SPIN):
- Segmento: ${seg}
- Investimento mensal em tráfego pago: ${budget || 'Não informado'}
- Principal canal de aquisição: ${canal || 'Não informado'}
- Maior gargalo/dor [SPIN — Problema]: ${pain}
- Impacto se nada mudar [SPIN — Implicação]: ${impact || 'Não informado'}
${hasDigital ? `\nPRESENÇA DIGITAL ANALISADA:${digitalCtx}` : '\nPresença digital: não fornecida.'}

REGRAS ABSOLUTAS:
1. Cada frase deve ser específica para ${seg} com canal ${canal || 'informado'} e dor "${pain}"
2. NUNCA generalize — se servir para outro negócio, reescreva
3. Gargalos explicam POR QUÊ esse problema acontece nesse segmento
4. Próximos passos são executáveis essa semana
5. Score de 1-100 reflete a saúde real do marketing com base nos dados fornecidos
${hasDigital ? '6. Cite elementos CONCRETOS encontrados no site (pixel, CTA, heading, URL real)' : ''}

Responda SOMENTE com JSON válido, sem markdown:
{
  "nivel": "diagnóstico em 1-2 frases — específico para ${seg} com canal ${canal || 'informado'} e dor '${pain}'",
  "score": <inteiro 1-100>,
  "gargalos": [
    "por que '${pain}' acontece especificamente em ${seg}",
    "consequência direta no dia a dia desse negócio",
    "o que está sendo perdido por causa disso"
  ],
  "perda_estimada": "valor mensal em R$ considerando ${budget || 'ausência de investimento'} no segmento ${seg}",
  "boa_noticia": "oportunidade real e concreta para ${seg} com ${canal || 'canal informado'} — o que pode mudar rápido",
  "proximos_passos": [
    "ação específica para ${seg}, executável essa semana",
    "resolve diretamente '${pain}'",
    "resultado mensurável em até 30 dias"
  ],
  "insights_digitais": ${hasDigital ? '"análise do que foi encontrado: cite elementos reais como pixel ausente, CTA fraco, heading vago, etc."' : 'null'}
}`;

  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  const message = await client.messages.create({
    model: 'claude-opus-4-6',
    max_tokens: 1500,
    messages: [{ role: 'user', content: prompt }],
  });
  const raw = message.content.find(b => b.type === 'text')?.text || '';
  const match = raw.match(/\{[\s\S]+\}/);
  if (!match) throw new Error('Claude: no JSON');
  return JSON.parse(match[0]);
}

// ── Motor 2: GPT-4o — Diagnóstico completo + benchmarks ─────────────────────

async function runGPT({ seg, budget, canal, pain, impact, digitalCtx }) {
  const hasDigital = digitalCtx.trim().length > 0;
  const prompt = `Você é o Oziris, especialista sênior em marketing digital e automação para negócios brasileiros.

DADOS COMPLETOS DO LEAD (funil SPIN):
- Segmento: ${seg}
- Investimento mensal em tráfego pago: ${budget || 'Não informado'}
- Principal canal de aquisição: ${canal || 'Não informado'}
- Maior gargalo/dor [SPIN — Problema]: ${pain}
- Impacto se nada mudar [SPIN — Implicação]: ${impact || 'Não informado'}
${hasDigital ? `\nPRESENÇA DIGITAL ANALISADA:${digitalCtx}` : '\nPresença digital: não fornecida.'}

REGRAS ABSOLUTAS:
1. Cada frase deve ser específica para ${seg} com canal ${canal || 'informado'} e dor "${pain}"
2. NUNCA generalize — se servir para outro negócio, reescreva
3. Gargalos explicam POR QUÊ esse problema acontece nesse segmento
4. Próximos passos são executáveis essa semana
5. Score de 1-100 reflete a saúde real do marketing com base nos dados fornecidos
${hasDigital ? '6. Cite elementos CONCRETOS encontrados no site (pixel, CTA, heading real)' : ''}

Responda SOMENTE com JSON válido, sem markdown:
{
  "nivel": "diagnóstico em 1-2 frases — específico para ${seg} com canal ${canal || 'informado'} e dor '${pain}'",
  "score": <inteiro 1-100>,
  "gargalos": [
    "por que '${pain}' acontece especificamente em ${seg}",
    "consequência direta no dia a dia desse negócio",
    "o que está sendo perdido por causa disso"
  ],
  "perda_estimada": "valor mensal em R$ considerando ${budget || 'ausência de investimento'} no segmento ${seg}",
  "boa_noticia": "oportunidade real e concreta para ${seg} com ${canal || 'canal informado'} — o que pode mudar rápido",
  "proximos_passos": [
    "ação específica para ${seg}, executável essa semana",
    "resolve diretamente '${pain}'",
    "resultado mensurável em até 30 dias"
  ],
  "insights_digitais": ${hasDigital ? '"análise do que foi encontrado: cite elementos reais como pixel ausente, CTA fraco, heading vago, etc."' : 'null'},
  "benchmarks": [
    "o que os líderes de ${seg} fazem de diferente em relação a ${canal || 'aquisição de clientes'}",
    "métrica ou padrão de mercado que esse segmento deveria atingir",
    "estratégia comprovada que está funcionando agora em ${seg} no Brasil"
  ]
}`;

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const res = await client.chat.completions.create({
    model: 'gpt-4o',
    max_tokens: 1500,
    messages: [{ role: 'user', content: prompt }],
  });
  const raw = res.choices[0].message.content || '';
  const match = raw.match(/\{[\s\S]+\}/);
  if (!match) throw new Error('GPT: no JSON');
  return JSON.parse(match[0]);
}

// ── Motor 3: Gemini — Quick wins ─────────────────────────────────────────────

async function runGemini({ seg, canal, pain, digitalCtx }) {
  const hasDigital = digitalCtx.trim().length > 0;
  const prompt = `Você é um consultor de marketing digital para pequenas e médias empresas brasileiras.

Lead:
- Segmento: ${seg}
- Canal principal: ${canal || 'Não informado'}
- Principal dor: ${pain}
${hasDigital ? `\nPresença digital:\n${digitalCtx}` : ''}

Gere quick wins — ações concretas que esse negócio pode executar nas próximas 48 horas para sentir resultado rápido. Seja específico para ${seg}.

Responda SOMENTE com JSON válido, sem markdown:
{
  "quick_wins": [
    "ação que pode ser feita hoje, específica para ${seg} usando ${canal || 'o canal principal'}",
    "resolve parte de '${pain}' em menos de 48h",
    "configuração ou ajuste simples com impacto imediato"
  ]
}`;

  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  const model = genAI.getGenerativeModel({ model: 'gemini-1.5-pro' });
  const result = await model.generateContent(prompt);
  const raw = result.response.text();
  const match = raw.match(/\{[\s\S]+\}/);
  if (!match) throw new Error('Gemini: no JSON');
  return JSON.parse(match[0]);
}

// ── Handler principal ─────────────────────────────────────────────────────────

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { seg, budget, canal, pain, impact, website, instagram, youtube } = req.body || {};
  if (!seg || !pain) return res.status(400).json({ error: 'Missing required fields: seg, pain' });

  // ── Scraping em paralelo ────────────────────────────────────────────────────
  const fetches = [];
  const keys = [];
  const websiteUrl   = normalizeUrl(website);
  const instagramUrl = normalizeInstagram(instagram);
  const youtubeUrl   = normalizeUrl(youtube);
  if (websiteUrl)   { fetches.push(fetchWithTimeout(websiteUrl));   keys.push('website'); }
  if (instagramUrl) { fetches.push(fetchWithTimeout(instagramUrl)); keys.push('instagram'); }
  if (youtubeUrl)   { fetches.push(fetchWithTimeout(youtubeUrl));   keys.push('youtube'); }

  const scraped = fetches.length ? await Promise.allSettled(fetches) : [];
  const digital = {};
  scraped.forEach((r, i) => {
    const html = r.status === 'fulfilled' ? r.value : null;
    if (keys[i] === 'website')   digital.website   = extractWebsiteData(html, websiteUrl);
    else if (keys[i] === 'instagram') digital.instagram = { platform: 'instagram', title: '', description: '' };
    else if (keys[i] === 'youtube')   digital.youtube   = extractWebsiteData(html, youtubeUrl);
  });

  const digitalCtx = buildDigitalCtx(digital);
  const data = { seg, budget, canal, pain, impact, digitalCtx };

  // ── 3 motores em paralelo ───────────────────────────────────────────────────
  const [claudeRes, gptRes, geminiRes] = await Promise.allSettled([
    runClaude(data),
    runGPT(data),
    runGemini(data),
  ]);

  const claude = claudeRes.status === 'fulfilled' ? claudeRes.value : null;
  const gpt    = gptRes.status    === 'fulfilled' ? gptRes.value    : null;
  const gemini = geminiRes.status === 'fulfilled' ? geminiRes.value : null;

  if (claudeRes.status === 'rejected') {
    console.error('[oziris] Claude falhou:', claudeRes.reason?.message);
  }
  if (gptRes.status === 'rejected') {
    console.error('[oziris] GPT falhou:', gptRes.reason?.message);
  }
  if (geminiRes.status === 'rejected') {
    console.error('[oziris] Gemini falhou:', geminiRes.reason?.message);
  }

  // Primário: Claude. Fallback automático: GPT-4o.
  const primary = claude || gpt;

  if (!primary) {
    return res.status(500).json({ error: 'All engines failed', details: {
      claude: claudeRes.reason?.message,
      gpt:    gptRes.reason?.message,
    }});
  }

  // ── Merge dos resultados ────────────────────────────────────────────────────
  const diagnosis = {
    ...primary,
    // Se Claude rodou, pega benchmarks do GPT (ele gerou benchmarks separados)
    // Se GPT é primário, benchmarks já estão em primary
    benchmarks:  (!claude && gpt) ? (gpt.benchmarks || []) : (gpt?.benchmarks || primary.benchmarks || []),
    quick_wins:  gemini?.quick_wins || [],
    engines_used: [
      claude ? 'Claude Opus' : null,
      gpt    ? 'GPT-4o'      : null,
      gemini ? 'Gemini 1.5 Pro' : null,
    ].filter(Boolean),
  };

  return res.status(200).json({ success: true, diagnosis });
}
