from datasets import load_dataset

ds = load_dataset("elricwan/dailydialog", split="train")

with open("chat_pairs.txt", "w", encoding="utf-8") as f:
    count = 0

    for row in ds:
        convo = row["conversation"]

        if not isinstance(convo, list) or len(convo) < 2:
            continue

        for i in range(len(convo) - 1):
            user = str(convo[i]).strip()
            bot = str(convo[i + 1]).strip()

            if not user or not bot:
                continue

            # remove Person A / Person B labels
            if user.startswith("Person A:"):
                user = user[len("Person A:"):].strip()
            elif user.startswith("Person B:"):
                user = user[len("Person B:"):].strip()

            if bot.startswith("Person A:"):
                bot = bot[len("Person A:"):].strip()
            elif bot.startswith("Person B:"):
                bot = bot[len("Person B:"):].strip()

            if user and bot:
                f.write(f"<BOS>User: {user}\nBot: {bot}<EOS>\n\n")
                count += 1

print("Wrote pairs:", count)