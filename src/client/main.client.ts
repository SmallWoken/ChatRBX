// Client — calls the server-side ChatRequest RemoteFunction.
const ReplicatedStorage = game.GetService("ReplicatedStorage");
const Players = game.GetService("Players");

const chatRemote = ReplicatedStorage.WaitForChild("ChatRequest") as RemoteFunction;
const localPlayer = Players.LocalPlayer;

// Test: fire a prompt when the character loads
localPlayer.CharacterAdded.Connect(() => {
	task.wait(1); // brief delay so the server finishes setting up

	const prompt = "Once upon a time";
	print(`[chatrbx] Sending prompt: "${prompt}"`);

	const response = chatRemote.InvokeServer(prompt) as string;
	print(`[chatrbx] Got response: "${response}"`);
});
