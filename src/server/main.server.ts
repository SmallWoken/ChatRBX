import { TinyGPT, WeightsData } from "shared/TinyGPT";
import data from "shared/weights.json";

const ReplicatedStorage = game.GetService("ReplicatedStorage");

const model = new TinyGPT(data as WeightsData);
print("[chatrbx] isReady:", model.isReady());

if (!model.isReady()) {
	warn("[chatrbx] weights.json is placeholder or invalid");
}

let chatRemote = ReplicatedStorage.FindFirstChild("ChatRequest") as RemoteFunction | undefined;
if (!chatRemote) {
	chatRemote = new Instance("RemoteFunction");
	chatRemote.Name = "ChatRequest";
	chatRemote.Parent = ReplicatedStorage;
}

chatRemote.OnServerInvoke = (_player: Player, prompt: unknown) => {
	if (!typeIs(prompt, "string")) return "[error: prompt must be a string]";
	if (!model.isReady()) return "[model not trained yet]";

	const trimmed = prompt.sub(1, 120);

	// Greedy first is easier to debug.
	const response = model.generate(trimmed, 120, 0.7, 20, (partial) => {
		print(`[gen] ${partial}`);
	});

	return response.size() > 0 ? response : "[no response]";
};

print("[chatrbx] ready");