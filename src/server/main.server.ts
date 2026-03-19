import { TinyGPT, WeightsData } from "shared/TinyGPT";
import data from "shared/weights.json";

const ReplicatedStorage = game.GetService("ReplicatedStorage");

const model = new TinyGPT(data as unknown as WeightsData);
print("[chatrbx] isReady:", model.isReady());

if (!model.isReady()) {
	warn("[chatrbx] weights.json is placeholder, train the model first");
}

const chatRemote = new Instance("RemoteFunction");
chatRemote.Name = "ChatRequest";
chatRemote.Parent = ReplicatedStorage;

chatRemote.OnServerInvoke = (_player: Player, ...args: unknown[]) => {
	const prompt = args[0];
	if (!typeIs(prompt, "string")) return "[error: prompt must be a string]";
	if (!model.isReady()) return "[model not trained yet]";

	const trimmed = prompt.sub(1, 128);
	const response = model.generate(trimmed, 150, 0.8, 40, (partial) => {
		print(`[gen] ${partial}`);
	});
	return response;
};

print("[chatrbx] ready");
