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
	weights: { [key: string]: number[] | number[][] };
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
	for (let i = 1; i < x.size(); i++) if (x[i] > maxV) maxV = x[i];
	const out = zeros(x.size());
	let s = 0;
	for (let i = 0; i < x.size(); i++) {
		out[i] = math.exp(x[i] - maxV);
		s += out[i];
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

export class TinyGPT {
	private config: ModelConfig;
	private w: { [key: string]: Vec | Mat };
	private stoi: { [key: string]: number };
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
	}

	private g(key: string): Vec {
		return this.w[key] as Vec;
	}

	private gm(key: string): Mat {
		return this.w[key] as Mat;
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
			for (let d = 0; d < n_embd; d++) row[d] = te[d] + pe[d];
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
				qkv.push(linear(c_attn_w, layerNorm(x[t], ln1_w, ln1_b), c_attn_b));
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
							dot += qkv[t1][hs + d] * qkv[t2][n_embd + hs + d];
						}
						scores[t2] = dot * scale;
					}
					const att = softmax(scores);

					for (let d = 0; d < head_size; d++) {
						let sum = 0;
						for (let t2 = 0; t2 <= t1; t2++) {
							sum += att[t2] * qkv[t2][2 * n_embd + hs + d];
						}
						attn_out[t1][hs + d] = sum;
					}
				}
			}

			for (let t = 0; t < T; t++) {
				vecAddInPlace(x[t], linear(c_proj_w, attn_out[t], c_proj_b));
			}

			for (let t = 0; t < T; t++) {
				const h_raw = linear(fc_w, layerNorm(x[t], ln2_w, ln2_b), fc_b);
				for (let i = 0; i < h_raw.size(); i++) h_raw[i] = gelu(h_raw[i]);
				vecAddInPlace(x[t], linear(mlp_w, h_raw, mlp_b));
			}

			task.wait();
		}

		const ln_f_w = this.g("ln_f.weight");
		const ln_f_b = this.g("ln_f.bias");
		const x_final = layerNorm(x[T - 1], ln_f_w, ln_f_b);
		return linearNoBias(this.gm("lm_head.weight"), x_final);
	}

	generate(prompt: string, maxTokens = 100, temperature = 0.8, topK = 40, onToken?: (partial: string) => void): string {
		const tokens: number[] = [];
		for (let i = 0; i < prompt.size(); i++) {
			const ch = prompt.sub(i + 1, i + 1);
			const id = this.stoi[ch];
			if (id !== undefined) tokens.push(id);
		}
		if (tokens.size() === 0) tokens.push(0);

		let result = "";
		for (let step = 0; step < maxTokens; step++) {
			const logits = this.forward(tokens);

			for (let i = 0; i < logits.size(); i++) logits[i] /= temperature;

			if (topK > 0 && topK < logits.size()) {
				const copy: Vec = [];
				for (let i = 0; i < logits.size(); i++) copy.push(logits[i]);
				copy.sort((a, b) => a > b);
				const threshold = copy[topK - 1];
				for (let i = 0; i < logits.size(); i++) {
					if (logits[i] < threshold) logits[i] = -1e38;
				}
			}

			const nextTok = sampleCategorical(softmax(logits));
			tokens.push(nextTok);
			result += this.itos[nextTok];
			if (onToken !== undefined) onToken(result);
		}

		return result;
	}

	isReady(): boolean {
		return this.w["wte.weight"] !== undefined;
	}
}
