const ReplicatedStorage = game.GetService("ReplicatedStorage");
const ChatService = game.GetService("Chat");
const Players = game.GetService("Players");

const ChatRBX = game.Workspace.WaitForChild("ChatRBX") as Model;
const Head = ChatRBX.WaitForChild("Head") as BasePart;

const chatRemote = ReplicatedStorage.WaitForChild("ChatRequest") as RemoteFunction;
const localPlayer = Players.LocalPlayer;

localPlayer.Chatted.Connect((message) => {
	const res = chatRemote.InvokeServer(message);
	if (typeIs(res, "string")) {
		ChatService.Chat(Head, res, Enum.ChatColor.Blue);
	} else {
		ChatService.Chat(Head, "[server error]", Enum.ChatColor.Red);
	}
});