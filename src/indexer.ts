import * as vscode from 'vscode';
import fg from 'fast-glob';
import * as path from 'path';
import * as fs from 'fs/promises';
import OpenAI from 'openai';

// Lazy import pdf-parse to keep activation light
async function parsePdfText(pdfPath?: string): Promise<string | undefined> {
if (!pdfPath) return undefined;
try {
// eslint-disable-next-line @typescript-eslint/no-var-requires
const pdf = require('pdf-parse');
const data = await fs.readFile(pdfPath);
const out = await pdf(data);
return typeof out?.text === 'string' ? out.text : undefined;
} catch {
return undefined;
}
}
export type IndexChunk = {
path: string;
startLine: number;
endLine: number;
text: string;
embedding?: number[];
};


export type IndexBlob = {
repoPath: string;
chunks: IndexChunk[];
meta: {
madeAt: number;
embeddingModel?: string;
maxChunkLines: number;
paperPresent: boolean;
};
};
function tokenizeLines(text: string, maxLines: number): { start: number; end: number; text: string }[] {
const lines = text.split(/\r?\n/);
const chunks: { start: number; end: number; text: string }[] = [];
for (let i = 0; i < lines.length; i += maxLines) {
const start = i;
const end = Math.min(lines.length - 1, i + maxLines - 1);
const slice = lines.slice(start, end + 1).join('');
chunks.push({ start, end, text: slice });
}
return chunks;
}


async function embedTexts(client: OpenAI, model: string, inputs: string[]): Promise<number[][]> {
const resp = await client.embeddings.create({ model, input: inputs });
return resp.data.map(d => d.embedding as number[]);
}
function cosineSim(a: number[], b: number[]): number {
let dot = 0, na = 0, nb = 0;
const n = Math.min(a.length, b.length);
for (let i = 0; i < n; i++) {
const x = a[i], y = b[i];
dot += x * y; na += x * x; nb += y * y;
}
return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

function sanitize(s: string): string {
  if (!s) return '';
  // drop control chars and collapse whitespace
  return s
    .replace(/[\u0000-\u001F\u007F]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
export async function buildIndex(opts: {
repoPath: string;
paperPath?: string;
model: string;
maxChunkLines: number;
apiKey?: string;
embeddingApiKey?: string;
apiBase?: string;
embeddingModel?: string;
}): Promise<IndexBlob> {
const { repoPath, paperPath,apiKey,embeddingApiKey, model,maxChunkLines, apiBase, embeddingModel } = opts;
const patterns = [
  '**/*.{ts,tsx,js,jsx,py,java,kt,go,rs,cpp,c,hpp,h,cs,scala,rb,php,swift,sh,txt}', // everything else
  '**/[Rr][Ee][Aa][Dd][Mm][Ee].[Mm][Dd]'                                           // README.md (any case)
];
const ignore = ['**/node_modules/**', '**/.git/**', '**/dist/**', '**/build/**', '**/.venv/**'];
const abs = await fg(patterns, { cwd: repoPath, ignore, dot: false, absolute: true });
//vscode.window.showInformationMessage(`Found ${repoPath} files in paperpath ${paperPath}.`);

const chunks: IndexChunk[] = [];
for (const p of abs) {
    try {
const buf = await fs.readFile(p);
const text = new TextDecoder().decode(buf);
for (const c of tokenizeLines(text, Math.max(1, maxChunkLines))) {
chunks.push({ path: p, startLine: c.start, endLine: c.end, text: c.text });
}
} catch { 
  vscode.window.showInformationMessage(`Found ${repoPath} files in paperpath ${paperPath}.`);
  /* ignore file read errors */ }
}
//vscode.window.showInformationMessage(`Indexed ${chunks.length} chunks from ${abs.length} files.`);
//const apiKey='ollama';
 //baseURL:model.includes('claude')?'https://api.openai.com/v1':
const client = apiKey ? new OpenAI({ apiKey: embeddingApiKey,baseURL:model.includes('claude')?'https://api.openai.com/v1':apiBase }) : undefined;
//vscode.window.showInformationMessage(`Using embedding model: ${client}`);
const embModel = embeddingModel;
//vscode.window.showInformationMessage(`apibase and embedding model: ${apiBase}, ${embModel}`);
if ( embModel) {
  try {
    // Build a cleaned list of non-empty texts with their original indices
    const cleaned: { idx: number; text: string }[] = [];
    for (let i = 0; i < chunks.length; i++) {
      const s = (chunks[i].text ?? '').toString().trim();
      if (s.length) cleaned.push({ idx: i, text: s });
    }
    
    if (cleaned.length === 0) {
      console.warn('[index] no non-empty texts; skipping embeddings');
    } else {
         // vscode.window.showInformationMessage(`Computing embeddings for  non-empty chunks...`);

     const B = 64; // modest batch size for most providers
      for (let i = 0; i < cleaned.length; i += B) {
        const slice = cleaned.slice(i, i + B);
        const res = await client!.embeddings.create({
          model: embModel,
          input: slice.map(x => x.text) // never empty
        });
       // vscode.window.showInformationMessage(`Computing and stroing to the correct index for  non-empty chunks...`);

        // write back to the correct chunk indices
        res.data.forEach((d, j) => {
          chunks[slice[j].idx].embedding = d.embedding as number[];
              //    vscode.window.showInformationMessage(`Computing and stroing to the correct index for  non-empty chunks...${d.embedding}`);

        });
      }
     
}
    
  } catch (e: any) {
     vscode.window.showInformationMessage(`embeddings disabled during indexing:${e?.message || e}`);
    console.warn('[index] embeddings disabled during indexing:', e?.message || e);
    // (Optional) vscode.window.showWarningMessage('Embeddings unavailable; using text-only ranking.');
    // Continue without embeddings; askWithLLM should handle this case.
  }
}

else{
  vscode.window.showInformationMessage(`Skipping embeddings; no API key or model.`);
}

return {
repoPath,
chunks,
meta: { madeAt: Date.now(), embeddingModel: embModel, maxChunkLines, paperPresent: !!paperPath }
};
}
/**
 * BM25 ranker for NL→code without prebuilding state.
 * Returns [{ i, score }] where i is chunk index in index.chunks.
 */
 function bm25RankOnce(
  index: { chunks: { text: string }[] },
  question: string,
  opts: { k1?: number; b?: number; minWordLen?: number } = {}
): { i: number; score: number }[] {
  const chunks = index?.chunks || [];
  if (!chunks.length) return [];

  const k1 = opts.k1 ?? 1.6;
  const b  = opts.b  ?? 0.75;
  const MIN = opts.minWordLen ?? 2;

  // --- tiny code-aware tokenizers (local) ---
  const splitId = (s: string) =>
    s.replace(/([a-z])([A-Z])/g, '$1 $2')
     .toLowerCase()
     .split(/[^a-z0-9]+/i)
     .filter(w => w.length > MIN);

  const tokenizeCode = (text: string) => {
    const t = text || '';
    const toks: string[] = [];
    // comments & docstrings
    const comments = t.match(/\/\/.*|#.*|\/\*[\s\S]*?\*\/|"""[\s\S]*?"""|'''[\s\S]*?'''/g) || [];
    comments.forEach(c => toks.push(...splitId(c)));
    // strings
    const strings = t.match(/(["'`])(?:\\.|(?!\1).)*\1/g) || [];
    strings.forEach(s => toks.push(...splitId(s)));
    // identifiers
    const ids = t.match(/[A-Za-z_][A-Za-z0-9_]{2,}/g) || [];
    ids.forEach(id => toks.push(...splitId(id)));
    return toks;
  };

  // --- build per-doc bags + DF, avgdl ---
  const df = new Map<string, number>();
  const bags: Array<Record<string, number>> = new Array(chunks.length);
  let totalLen = 0;

  for (let i = 0; i < chunks.length; i++) {
    const toks = tokenizeCode(chunks[i].text);
    totalLen += toks.length;
    const bag: Record<string, number> = {};
    for (const t of toks) bag[t] = (bag[t] || 0) + 1;
    bags[i] = bag;

    const uniq = new Set(Object.keys(bag));
    for (const t of uniq) df.set(t, (df.get(t) || 0) + 1);
  }

  const N = Math.max(1, chunks.length);
  const avgdl = Math.max(1, totalLen / N);

  // --- query terms ---
  const qTerms = splitId(question);

  // --- score each chunk with BM25 ---
  const scored = chunks.map((_, i) => {
    const bag = bags[i];
    const dl  = Object.values(bag).reduce((a, n) => a + n, 0) || 1;
    let score = 0;

    for (const q of qTerms) {
      const f = bag[q] || 0;
      if (!f) continue;
      const n = df.get(q) || 0;
      const idf = Math.log((N - n + 0.5) / (n + 0.5) + 1e-6);
      const tf  = (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / avgdl)));
      score += idf * tf;
    }
    return { i, score };
  });

  // higher first
  scored.sort((a, b) => b.score - a.score);
  return scored;
}
export function extractJsonObject(s: string): any | null {
  if (!s) return null;
  // strip ```json ... ```
  s = s.trim().replace(/^```(?:json)?\s*/i, "").replace(/```$/i, "");
  return s;
  vscode.window.showInformationMessage(`Extracting JSON from LLM response.. ${s}`);
  // fast path
  try { return JSON.parse(s); } catch {}

  // fallback: grab first balanced {...}
  const start = s.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  for (let i = start; i < s.length; i++) {
    const ch = s[i];
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        const cand = s.slice(start, i + 1);
        try { return JSON.parse(cand); } catch {}
      }
    }
  }
  return null;
}

export async function askWithLLM(index: IndexBlob, opts: {
question: string;
topK: number;
model: string;
apiKey: string;
embeddingApiKey?: string;
apiBase?: string;
temperature?: number;
}): Promise<{ path: string; startLine?: number; endLine?: number;reason?:string; bestLine?: number; score?: number } | undefined> {
const { question, topK, model, apiKey,embeddingApiKey, apiBase, temperature } = opts;
//const client = new OpenAI({ apiKey, baseURL: apiBase });
//vscode.window.showInformationMessage(`LLM asked: ${question} `);


// If we have embeddings, rank by cosine; else fallback to simple tf (length) heuristic
let scored = index.chunks.map((c, i) => ({ i, score: 0 }));
try {
const embModel = index.meta.embeddingModel;
//vscode.window.showInformationMessage(`embmodel for ${embModel}, ${index.chunks[0]?.embedding} `);

if (embModel && index.chunks[0]?.embedding) {
  //const apiKey='ollama';
  const client2 = new OpenAI({ apiKey:embeddingApiKey, baseURL:model.includes('claude')?'https://api.openai.com/v1': apiBase}) ;
const qv = (await client2.embeddings.create({ model: embModel, input: [question] })).data[0].embedding as number[];
//  const response = await fetch('http://localhost:11434/api/embeddings', {
//     method: 'POST',
//     headers: { 'Content-Type': 'application/json' },
//     body: JSON.stringify({
//       model: 'mxbai-embed-large',
//       input: sanitize(question), // Make sure this is an array of strings
//     }),
//   });
// const raw:any = await response.json();
// const emb: number[] = raw.embedding; 
scored = index.chunks.map((c, i) => ({ i, score: c.embedding ? cosineSim(qv, c.embedding) : 0 }));
//vscode.window.showInformationMessage(`Fell back to cosine similarity for ranking.${scored}`);
}
else{
  scored = bm25RankOnce(index, question);

vscode.window.showInformationMessage(`Fell back to lexical similarity for ranking.`);
}
} 
catch (e: any) {
console.warn('[ask] ranking fallback:', e?.message || e);
// (Optional) vscode.window.showWarningMessage('Embeddings unavailable; using text-only ranking.');
// scored stays as initialized (all zero)
}

scored.sort((a,b) => b.score - a.score);
const pick = scored.slice(0, Math.max(1,topK));
//vscode.window.showInformationMessage(`Top chunk score: ${pick.length}`);
const ctx = pick.map(({ i, score }, rank) => ({
rank, score, path: index.chunks[i].path, start: index.chunks[i].startLine, end: index.chunks[i].endLine, text: index.chunks[i].text
}));
//vscode.window.showInformationMessage(`Top chunk path: ${ctx} `);

const sys  = `
Task: Map the question to the most relevant code region among candidates.

Must Return ONLY  JSON with EXACTLY these keys:
- path (string)
- startLine (int)
- endLine (int)
- reason (string < 200 words)
- score( how confident the LLM is about its answer, float between 0 and 5 only integer)
`;
const user = [
{ type: 'text', text: `Question: ${question}` },
{ type: 'text', text: `Candidates (top-${ctx.length}):` },
{ type: 'text', text: ctx.map(c => `#${c.rank} score=${c.score.toFixed(3)} file=${c.path} [${c.start}-${c.end}]
${c.text}`).join('---') }
];

//vscode.window.showInformationMessage(`LLM chose  ${model}`);


const openai = new OpenAI({
  baseURL: apiBase,
  apiKey: apiKey, // required but unused
})
const completion = await openai.chat.completions.create({
  model: model,
  messages: [
{ role: 'system', content: sys },
{ role: 'user', content: user as any }
]
})

//vscode.window.showInformationMessage(`output  ${completion.choices[0].message.content}`)
const content = completion.choices[0]?.message?.content || '{}';
//let content_trimmed: any = extractJsonObject(content);
//vscode.window.showInformationMessage(`LLM outputs  ${content}`);
let json: any = {};

try { json = JSON.parse(extractJsonObject(content)); } catch { 
  //vscode.window.showInformationMessage(``);

  /* ignore */ }

//vscode.window.showInformationMessage(`LLM chose ${json.reason} `);
if (!json?.path) return undefined;

return { path: json.path, startLine: json.startLine, endLine: json.endLine,reason:json.reason, score: ctx[0]?.score, bestLine: json.startLine };
}
export async function saveIndex(ctx: vscode.ExtensionContext, idx: IndexBlob) {
const target = vscode.Uri.joinPath(ctx.globalStorageUri, 'index.json');
const payload = JSON.stringify(idx);
await vscode.workspace.fs.writeFile(target, new TextEncoder().encode(payload));
}


export async function loadIndex(ctx: vscode.ExtensionContext): Promise<IndexBlob | undefined> {
try {
const target = vscode.Uri.joinPath(ctx.globalStorageUri, 'index.json');
const data = await vscode.workspace.fs.readFile(target);
const json = JSON.parse(new TextDecoder().decode(data)) as IndexBlob;
return json;
} catch {
return undefined;
}
}
export async function deleteIndex(ctx: vscode.ExtensionContext): Promise<boolean> {
  const target = vscode.Uri.joinPath(ctx.globalStorageUri, 'index.json');

  // If it doesn't exist, treat as "nothing to delete"
  try {
    await vscode.workspace.fs.stat(target);
  } catch {
    return false;
  }

  // Try to delete; don't send to OS trash
  try {
    await vscode.workspace.fs.delete(target, { useTrash: false });
    return true;
  } catch {
    return false;
  }
}
export async function suggestQuestions(
  index: IndexBlob,
  opts: { apiKey: string; apiBase?: string; model: string; paperPath?: string }
): Promise<string[]> {
  const client = new OpenAI({ apiKey: opts.apiKey, baseURL: opts.apiBase });

  // Try reading a small excerpt of the paper (optional)
  let paperText = '';
  if (opts.paperPath) {
    try {
      paperText = (await parsePdfText(opts.paperPath))?.slice(0, 2000) || '';
    } catch { /* ignore */ }
  }

  // Give the LLM a sense of the repo contents (limit for token safety)
  const files = Array.from(new Set(index.chunks.map(c => c.path))).slice(0, 40);

  const system = 'You are helping a newcomer read a research paper and its codebase. Propose five short, beginner-friendly questions that help them explore the implementation based on the paper and files. Return strictly a JSON object: { "questions": string[5] }.';
  const user = `Paper excerpt (may be empty):${paperText}

Repo files (sample):
${files.join('\n')}`;

  const chat = await client.chat.completions.create({
    model: opts.model,
    temperature: 1,
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: user }
    ]
  });

  try {
    const content = chat.choices[0]?.message?.content || '{}';
    const parsed = JSON.parse(content);
    const arr = Array.isArray(parsed?.questions) ? parsed.questions : [];
    return arr.filter((s: any) => typeof s === 'string').slice(0, 5);
  } catch {
    return [];
  }
}
// indexer.ts

async function embedAll(
  client: OpenAI,
  model: string,
  texts: string[]
): Promise<number[][]> {
  // Clean input and keep index mapping
  const cleaned: { idx: number; text: string }[] = [];
  texts.forEach((t, i) => {
    const s = (t ?? "").toString().trim();
    if (s.length > 0) cleaned.push({ idx: i, text: s });
  });

  if (cleaned.length === 0) return []; // nothing to embed (avoid 400)

  const BATCH = 128; // keep it modest for most providers
  const out: number[][] = new Array(texts.length); // aligned to original length (sparse)
  for (let i = 0; i < cleaned.length; i += BATCH) {
    const slice = cleaned.slice(i, i + BATCH);
    const res = await client.embeddings.create({
      model,
      input: slice.map(x => x.text)
    });
    // write back embeddings aligned to original positions
    res.data.forEach((d, j) => { out[slice[j].idx] = d.embedding as number[]; });
  }
  return out;
}

export async function suggestFollowupsFromPrompt(
  prompt: string,
  opts: { apiKey: string; apiBase?: string; model: string ,paperPath?:string}
): Promise<string[]> {
  const client = new OpenAI({ apiKey: opts.apiKey, baseURL: opts.apiBase });
  let paperText = '';
  if (opts.paperPath) {
    try {
      paperText = (await parsePdfText(opts.paperPath))?.slice(0, 2000) || '';
    } catch { /* ignore */ }
  }

  // Give the LLM a sense of the repo contents (limit for token safety)
  //const files = Array.from(new Set(index.chunks.map(c => c.path))).slice(0, 40);

  const sys = 'You are helping a newcomer read a research paper and its codebase. Propose five short, beginner-friendly questions that help them explore the implementation based on the paper and files. Return strictly a JSON object: { "questions": string[5] }.';
  const user = `Paper excerpt (may be empty):${paperText}
and the user prompt is: ${prompt}
}`;
  // const sys =
  //   'You generate 5 short follow-up questions based ONLY on the user prompt. Return JSON: {"questions": string[5]}. No preamble.';
  // const user = `User prompt:\n${prompt}`;

  const chat = await client.chat.completions.create({
    model: opts.model,
    temperature: opts.model==='gpt-5'?1:0,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: sys },
      { role: "user", content: user }
    ]
  });

  try {
    const content = chat.choices[0]?.message?.content || "{}";
    const parsed = JSON.parse(extractJsonObject(content));
    const arr = Array.isArray(parsed?.questions) ? parsed.questions : [];
    return arr.filter((s: any) => typeof s === "string").slice(0, 5);
  } catch {
    return [];
  }
}


