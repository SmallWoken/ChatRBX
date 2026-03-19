export interface ModelConfig {
	vocab_size: number;
	n_embd: number;
	n_head: number;
	n_layer: number;
	block_size: number;
}

export interface WeightsData {
	_placeholder?: boolean;
	config: ModelConfig;
	chars: string;
	stoi?: Record<string, number>;
	weights: Record<string, number[] | number[][]>;
}

type Vec = number[];
type Mat = number[][];

function zeros(n: number): Vec {
	const v: Vec = [];
	for (let i = 0; i < n; i++) v.push(0);
	return v;
}

function vecAddInPlace(a: Vec, b: Vec): void {
	for (let i = 0; i < a.size(); i++) a[i] += b[i];
}

function linear(W: Mat, x: Vec, bias: Vec): Vec {
	const out = zeros(W.size());
	for (let i = 0; i < W.size(); i++) {
		let s = bias[i];
		const row = W[i];
		for (let j = 0; j < x.size(); j++) s += row[j] * x[j];
		out[i] = s;
	}
	return out;
}

function linearNoBias(W: Mat, x: Vec): Vec {
	const out = zeros(W.size());
	for (let i = 0; i < W.size(); i++) {
		let s = 0;
		const row = W[i];
		for (let j = 0; j < x.size(); j++) s += row[j] * x[j];
		out[i] = s;
	}
	return out;
}

function layerNorm(x: Vec, w: Vec, b: Vec): Vec {
	const n = x.size();
	let mean = 0;
	for (let i = 0; i < n; i++) mean += x[i];
	mean /= n;

	let variance = 0;
	for (let i = 0; i < n; i++) {
		const d = x[i] - mean;
		variance += d * d;
	}
	variance /= n;

	const inv = 1 / math.sqrt(variance + 1e-5);
	const out = zeros(n);
	for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
	return out;
}

function gelu(x: number): number {
	const k = math.sqrt(2 / math.pi);
	return 0.5 * x * (1 + math.tanh(k * (x + 0.044715 * x * x * x)));
}

function softmax(x: Vec): Vec {
	let maxV = x[0];
	for (let i = 1; i < x.size(); i++) {
		if (x[i] > maxV) maxV = x[i];
	}

	const out = zeros(x.size());
	let s = 0;
	for (let i = 0; i < x.size(); i++) {
		out[i] = math.exp(x[i] - maxV);
		s += out[i];
	}

	if (s <= 0) {
		const uniform = 1 / x.size();
		for (let i = 0; i < x.size(); i++) out[i] = uniform;
		return out;
	}

	for (let i = 0; i < x.size(); i++) out[i] /= s;
	return out;
}

function sampleCategorical(probs: Vec): number {
	const r = math.random();
	let cumulative = 0;
	for (let i = 0; i < probs.size(); i++) {
		cumulative += probs[i];
		if (r <= cumulative) return i;
	}
	return probs.size() - 1;
}

function cloneVec(v: Vec): Vec {
	const out: Vec = [];
	for (let i = 0; i < v.size(); i++) out.push(v[i]);
	return out;
}

export class TinyGPT {
	private config: ModelConfig;
	private w: Record<string, Vec | Mat>;
	private stoi: Record<string, number>;
	private itos: string[];

	constructor(data: WeightsData) {
		this.config = data.config;
		this.w = data.weights;
		this.stoi = {};
		this.itos = [];

		for (let i = 0; i < data.chars.size(); i++) {
			const ch = data.chars.sub(i + 1, i + 1);
			this.stoi[ch] = i;
			this.itos.push(ch);
		}

		// Optional debug check against exported stoi
		if (data.stoi !== undefined) {
			for (const [ch, id] of pairs(data.stoi)) {
				if (this.stoi[ch] !== id) {
					warn(`[TinyGPT] stoi mismatch for ${ch}: local=${this.stoi[ch]} exported=${id}`);
				}
			}
		}
	}

	private g(key: string): Vec {
		return this.w[key] as Vec;
	}

	private gm(key: string): Mat {
		return this.w[key] as Mat;
	}

	private encode(text: string): number[] {
		const tokens: number[] = [];
		for (let i = 0; i < text.size(); i++) {
			const ch = text.sub(i + 1, i + 1);
			const id = this.stoi[ch];
			if (id !== undefined) tokens.push(id);
		}
		return tokens;
	}

	private decode(tokens: number[]): string {
		let out = "";
		for (let i = 0; i < tokens.size(); i++) {
			const id = tokens[i];
			if (id >= 0 && id < this.itos.size()) {
				out += this.itos[id];
			}
		}
		return out;
	}

	private forward(tokens: number[]): Vec {
		const { n_embd, n_head, n_layer, block_size } = this.config;
		const head_size = n_embd / n_head;
		const scale = 1 / math.sqrt(head_size);

		const start = tokens.size() > block_size ? tokens.size() - block_size : 0;
		const T = tokens.size() - start;

		const wte = this.gm("wte.weight");
		const wpe = this.gm("wpe.weight");

		const x: Mat = [];
		for (let t = 0; t < T; t++) {
			const tok = tokens[start + t];
			const row = zeros(n_embd);
			const te = wte[tok];
			const pe = wpe[t];
			for (let d = 0; d < n_embd; d++) {
				row[d] = te[d] + pe[d];
			}
			x.push(row);
		}

		for (let l = 0; l < n_layer; l++) {
			const ln1_w = this.g(`blocks.${l}.ln1.weight`);
			const ln1_b = this.g(`blocks.${l}.ln1.bias`);
			const ln2_w = this.g(`blocks.${l}.ln2.weight`);
			const ln2_b = this.g(`blocks.${l}.ln2.bias`);

			const c_attn_w = this.gm(`blocks.${l}.attn.c_attn.weight`);
			const c_attn_b = this.g(`blocks.${l}.attn.c_attn.bias`);
			const c_proj_w = this.gm(`blocks.${l}.attn.c_proj.weight`);
			const c_proj_b = this.g(`blocks.${l}.attn.c_proj.bias`);

			const fc_w = this.gm(`blocks.${l}.mlp.fc.weight`);
			const fc_b = this.g(`blocks.${l}.mlp.fc.bias`);
			const mlp_w = this.gm(`blocks.${l}.mlp.proj.weight`);
			const mlp_b = this.g(`blocks.${l}.mlp.proj.bias`);

			const qkv: Mat = [];
			for (let t = 0; t < T; t++) {
				const norm = layerNorm(x[t], ln1_w, ln1_b);
				qkv.push(linear(c_attn_w, norm, c_attn_b));
			}

			const attn_out: Mat = [];
			for (let t = 0; t < T; t++) attn_out.push(zeros(n_embd));

			for (let h = 0; h < n_head; h++) {
				const hs = h * head_size;

				for (let t1 = 0; t1 < T; t1++) {
					const scores = zeros(t1 + 1);

					for (let t2 = 0; t2 <= t1; t2++) {
						let dot = 0;
						for (let d = 0; d < head_size; d++) {
							const q = qkv[t1][hs + d];
							const k = qkv[t2][n_embd + hs + d];
							dot += q * k;
						}
						scores[t2] = dot * scale;
					}

					const att = softmax(scores);

					for (let d = 0; d < head_size; d++) {
						let sum = 0;
						for (let t2 = 0; t2 <= t1; t2++) {
							const v = qkv[t2][2 * n_embd + hs + d];
							sum += att[t2] * v;
						}
						attn_out[t1][hs + d] = sum;
					}
				}
			}

			for (let t = 0; t < T; t++) {
				const proj = linear(c_proj_w, attn_out[t], c_proj_b);
				vecAddInPlace(x[t], proj);
			}

			for (let t = 0; t < T; t++) {
				const norm = layerNorm(x[t], ln2_w, ln2_b);
				const hRaw = linear(fc_w, norm, fc_b);
				for (let i = 0; i < hRaw.size(); i++) {
					hRaw[i] = gelu(hRaw[i]);
				}
				const proj = linear(mlp_w, hRaw, mlp_b);
				vecAddInPlace(x[t], proj);
			}

			task.wait();
		}

		const ln_f_w = this.g("ln_f.weight");
		const ln_f_b = this.g("ln_f.bias");
		const xFinal = layerNorm(x[T - 1], ln_f_w, ln_f_b);
		return linearNoBias(this.gm("lm_head.weight"), xFinal);
	}

	generate(
		prompt: string,
		maxTokens = 100,
		temperature = 0.8,
		topK = 40,
		onToken?: (partial: string) => void,
	): string {
		// Match training format better
		const seed = `<BOS>User: ${prompt}\nBot:`;
		const tokens = this.encode(seed);

		if (tokens.size() === 0) {
			return "[encode error]";
		}

		let generated = "";

		for (let step = 0; step < maxTokens; step++) {
			const logits = cloneVec(this.forward(tokens));

			if (temperature <= 0) {
				let bestIdx = 0;
				for (let i = 1; i < logits.size(); i++) {
					if (logits[i] > logits[bestIdx]) bestIdx = i;
				}
				tokens.push(bestIdx);

				const ch = this.itos[bestIdx] ?? "";
				generated += ch;
				if (onToken !== undefined) onToken(generated);

				if (generated.find("<EOS>")[0] !== undefined) break;
				continue;
			}

			for (let i = 0; i < logits.size(); i++) {
				logits[i] /= temperature;
			}

			if (topK > 0 && topK < logits.size()) {
				const sorted = cloneVec(logits);
				sorted.sort((a, b) => a > b);
				const threshold = sorted[topK - 1];
				for (let i = 0; i < logits.size(); i++) {
					if (logits[i] < threshold) logits[i] = -1e30;
				}
			}

			const probs = softmax(logits);
			const nextTok = sampleCategorical(probs);
			tokens.push(nextTok);

			const ch = this.itos[nextTok] ?? "";
			generated += ch;
			if (onToken !== undefined) onToken(generated);

			if (generated.find("<EOS>")[0] !== undefined) break;
		}

		const eosIdx = generated.find("<EOS>")[0];
		if (eosIdx !== undefined) {
			generated = generated.sub(1, eosIdx - 1);
		}

		return generated.gsub("^%s+", "")[0].gsub("%s+$", "")[0];
	}

	isReady(): boolean {
		return this.w["wte.weight"] !== undefined;
	}

	debugEncode(text: string): number[] {
		return this.encode(text);
	}
}