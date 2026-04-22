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

// Extrai dados reais do perfil público do Instagram via meta tags og:
function extractInstagramData(html, url, username) {
  const get = rx => (html.match(rx) || [])[1]?.trim() || '';
  const ogTitle = get(/<meta[^>]*property=["']og:title["'][^>]*content=["']([^"']+)["']/i)
               || get(/<meta[^>]*content=["']([^"']+)["'][^>]*property=["']og:title["']/i);
  const ogDesc  = get(/<meta[^>]*property=["']og:description["'][^>]*content=["']([^"']{1,500})["']/i)
               || get(/<meta[^>]*content=["']([^"']{1,500})["'][^>]*property=["']og:description["']/i);
  // og:description do Instagram: "12,3K Followers, 450 Following, 89 Posts – Bio text aqui"
  const followersM = ogDesc.match(/([\d.,]+\s*[KkMm]?)\s*Followers?/i);
  const followingM = ogDesc.match(/([\d.,]+)\s*Following/i);
  const postsM     = ogDesc.match(/([\d.,]+)\s*Posts?/i);
  const bioM       = ogDesc.match(/Posts?\s*[-–]\s*(.+)$/is);
  const blocked    = !ogTitle && (html.includes('Log in') || html.includes('login_page') || html.length < 500);
  return {
    url, username: username || '',
    title:     ogTitle || '',
    bio:       bioM ? bioM[1].trim() : (ogDesc || ''),
    followers: followersM ? followersM[1].trim() : null,
    following: followingM ? followingM[1].trim() : null,
    posts:     postsM     ? postsM[1].trim()     : null,
    blocked,
  };
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
    if (ig.blocked || (!ig.title && !ig.bio)) {
      ctx += `\n=== INSTAGRAM (@${ig.username || 'informado'}) ===\nStatus: Perfil existe mas meta-dados não acessíveis publicamente. Use username para análise contextual.`;
    } else {
      ctx += `\n=== INSTAGRAM (@${ig.username}) ===
Nome/Perfil: ${ig.title}
Bio: ${ig.bio || 'Não encontrado'}
Seguidores: ${ig.followers || 'Não disponível'}
Seguindo: ${ig.following || 'Não disponível'}
Posts publicados: ${ig.posts || 'Não disponível'}
URL: ${ig.url}`;
    }
  }
  if (digital.youtube) {
    const yt = digital.youtube;
    ctx += `\n=== YOUTUBE (${yt.url}) ===
Título: ${yt.title || 'Não encontrado'}
Descrição: ${yt.description || 'Não encontrado'}
Headings: ${yt.headings?.join(' | ') || 'Nenhum'}`;
  }
  return ctx;
}

// ── Prompt compartilhado — 15 pontos de análise ──────────────────────────────

function buildAnalysisPrompt({ seg, budget, canal, pain, impact, digitalCtx }) {
  const hasDigital = digitalCtx.trim().length > 0;
  return `Você é o Oziris, especialista sênior em marketing digital e análise de negócios brasileiros.

PERFIL DO LEAD:
- Segmento: ${seg}
- Investimento mensal em tráfego pago: ${budget || 'Não informado'}
- Principal canal de aquisição: ${canal || 'Não informado'}
- Principal dor: ${pain}
- Impacto percebido: ${impact || 'Não informado'}
${hasDigital ? `\nPRESENÇA DIGITAL ANALISADA:${digitalCtx}` : '\nPresença digital: não fornecida.'}

REGRAS ABSOLUTAS:
1. CADA item deve ser específico para ${seg} com canal ${canal || 'informado'} — NUNCA generalize
2. Concorrentes devem ser marcas/empresas brasileiras reais e conhecidas em ${seg}
3. Score de 1-100 reflete a saúde real do marketing com os dados fornecidos
${hasDigital ? `4. OBRIGATÓRIO: cite dados REAIS encontrados (número de seguidores, bio exata, CTAs, pixels, headings)
5. insights_digitais DEVE mencionar elementos concretos: ex. "bio diz X", "sem pixel do Facebook", "${(digitalCtx.match(/@[\w.]+/) || [''])[0]} tem Y seguidores"` : ''}

Responda SOMENTE com JSON válido, sem markdown:
{
  "score": <inteiro 1-100>,
  "overview": "2-3 frases sobre posicionamento atual citando dados REAIS encontrados (bio, seguidores, título do site, etc.) — específico para ${seg}",
  "insights_digitais": ${hasDigital ? `"análise da presença digital com dados CONCRETOS: cite bio real, número de seguidores, pixel presente/ausente, CTA encontrada, heading do site — seja específico"` : 'null'},
  "publico_alvo": "quem é o público-alvo percebido com base no canal, segmento e dor relatada — idade, perfil, dores",
  "dores_exploradas": [
    "primeira dor que esse negócio já trabalha bem",
    "segunda dor explorada",
    "terceira dor que poderia ser mais explorada"
  ],
  "clareza_oferta": "avaliação direta: o que está claro na oferta e o que está confuso ou ausente para ${seg}",
  "pontos_fortes": [
    "ponto forte 1 — específico e concreto para esse perfil",
    "ponto forte 2",
    "ponto forte 3"
  ],
  "gargalos_conversao": [
    "gargalo 1 — por que acontece especificamente em ${seg} com ${canal || 'esse canal'}",
    "gargalo 2 — consequência direta no faturamento",
    "gargalo 3 — o que está sendo perdido"
  ],
  "qualidade_conteudo": "avaliação da qualidade, relevância e consistência do conteúdo produzido para ${seg}",
  "consistencia_frequencia": "análise da consistência e frequência de publicação/comunicação no canal ${canal || 'principal'}",
  "estrutura_funil": "como está o funil atual: topo → meio → fundo — onde está quebrando para ${seg}",
  "concorrentes": [
    "Concorrente brasileiro real 1 em ${seg}",
    "Concorrente 2",
    "Concorrente 3",
    "Concorrente 4",
    "Concorrente 5",
    "Concorrente 6",
    "Concorrente 7",
    "Concorrente 8",
    "Concorrente 9",
    "Concorrente 10"
  ],
  "diferenciais": "o que diferencia esse negócio dos 10 concorrentes — o que está sendo desperdiçado ou não comunicado",
  "lacunas_mercado": [
    "lacuna 1 que nenhum concorrente está explorando bem em ${seg}",
    "lacuna 2",
    "lacuna 3"
  ],
  "novos_produtos": [
    "ideia de produto/serviço 1 viável para ${seg} baseada nas lacunas",
    "ideia 2",
    "ideia 3"
  ],
  "crescimento_ia": [
    "estratégia 1 de crescimento usando automação ou IA específica para ${seg}",
    "estratégia 2",
    "estratégia 3"
  ],
  "melhorias_funil": [
    "melhoria 1 no funil atual — executável essa semana para ${seg}",
    "melhoria 2",
    "melhoria 3"
  ]
}`;
}

// ── Motor 1: Claude — Análise principal ──────────────────────────────────────

async function runClaude(data) {
  const prompt = buildAnalysisPrompt(data);
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  const message = await client.messages.create({
    model: 'claude-opus-4-6',
    max_tokens: 2500,
    messages: [{ role: 'user', content: prompt }],
  });
  const raw = message.content.find(b => b.type === 'text')?.text || '';
  const match = raw.match(/\{[\s\S]+\}/);
  if (!match) throw new Error('Claude: no JSON');
  return JSON.parse(match[0]);
}

// ── Motor 2: GPT-4o — Análise completa (fallback + benchmarks extra) ─────────

async function runGPT(data) {
  const prompt = buildAnalysisPrompt(data);

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const res = await client.chat.completions.create({
    model: 'gpt-4o',
    max_tokens: 2500,
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
    else if (keys[i] === 'instagram') {
      const username = instagramUrl.replace('https://www.instagram.com/', '').replace(/\/$/, '');
      digital.instagram = html ? extractInstagramData(html, instagramUrl, username) : { url: instagramUrl, username, blocked: true };
    }
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
    // Gemini complementa com quick_wins (ações 48h)
    quick_wins: gemini?.quick_wins || [],
    engines_used: [
      claude ? 'Claude Opus' : null,
      gpt    ? 'GPT-4o'      : null,
      gemini ? 'Gemini 1.5 Pro' : null,
    ].filter(Boolean),
  };

  return res.status(200).json({ success: true, diagnosis });
}
