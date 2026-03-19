// Client — calls the server-side ChatRequest RemoteFunction.
const ReplicatedStorage = game.GetService("ReplicatedStorage");
const ChatService = game.GetService("Chat");
const Players = game.GetService("Players");
const ChatRBX = game.Workspace.WaitForChild("ChatRBX") as Model;
const Head = ChatRBX.WaitForChild("Head");

const chatRemote = ReplicatedStorage.WaitForChild("ChatRequest") as RemoteFunction;
const localPlayer = Players.LocalPlayer;

// Test: fire a prompt when the character loads
localPlayer.Chatted.Connect((message) => {
	const res = chatRemote.InvokeServer(message);
	ChatService.Chat(Head, res, Enum.ChatColor.Blue);
});