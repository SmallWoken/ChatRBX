const ReplicatedStorage = game.GetService("ReplicatedStorage");
const TextChatService = game.GetService("TextChatService");
const channel = TextChatService.WaitForChild("TextChannels").WaitForChild("RBXSystem") as TextChannel;
const ChatService = game.GetService("Chat");
const Players = game.GetService("Players");

const ChatRBX = game.Workspace.WaitForChild("ChatRBX") as Model;
const Head = ChatRBX.WaitForChild("Head") as BasePart;

const chatRemote = ReplicatedStorage.WaitForChild("ChatRequest") as RemoteFunction;
const localPlayer = Players.LocalPlayer;

function splitByEOS(s: string): string[] {
	const parts: string[] = [];
	let pos = 1;
	while (true) {
		const [start, stop] = string.find(s, "<EOS>", pos, true);
		if (start === undefined) {
			const remaining = s.sub(pos);
			if (remaining.size() > 0) parts.push(remaining);
			break;
		}
		const segment = s.sub(pos, (start as number) - 1);
		if (segment.size() > 0) parts.push(segment);
		pos = (stop as number) + 1;
	}
	return parts;
}

const thinkingPhrases = [
	"pondering...",
	"hmm...",
	"let me think...",
	"one moment...",
	"considering...",
	"thinking...",
	"processing...",
];

localPlayer.Chatted.Connect((message) => {
	let done = false;

	const thinkingThread = task.spawn(() => {
		let i = 0;
		while (!done) {
			ChatService.Chat(Head, thinkingPhrases[i % thinkingPhrases.size()], Enum.ChatColor.Green);
			i++;
			task.wait(2.5);
		}
	});

	const res = chatRemote.InvokeServer(message);
	done = true;
	task.cancel(thinkingThread);

	if (typeIs(res, "string")) {
		const parts = splitByEOS(res);
		for (const part of parts) {
			ChatService.Chat(Head, part, Enum.ChatColor.Blue);
			channel.DisplaySystemMessage(part);
			task.wait(math.random() * 0.5 + 0.3);
		}
	} else {
		ChatService.Chat(Head, "[server error]", Enum.ChatColor.Red);
	}
});