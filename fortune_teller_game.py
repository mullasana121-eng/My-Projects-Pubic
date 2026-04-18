import random

def fortune_teller_game():
    print("=" * 50)
    print("  🔮  Welcome to the Fortune Teller's Game  🔮")
    print("=" * 50)
    print("\n\"Ah, seeker... I sense a number in your future.\"")
    print("\"It lies between 1 and 100. Can you feel it?\"\n")

    magic_number = random.randint(1, 100)
    attempts = 0
    max_attempts = 10

    while attempts < max_attempts:
        remaining = max_attempts - attempts
        print(f"🌟 Attempts remaining: {remaining}")

        try:
            guess = int(input("Enter your guess: "))
        except ValueError:
            print("⚠️  The spirits are confused... Please enter a valid number.\n")
            continue

        attempts += 1

        if guess < 1 or guess > 100:
            print("⚠️  The number lies between 1 and 100, seeker!\n")
            attempts -= 1  # Don't count out-of-range as a valid attempt
            continue

        if guess == magic_number:
            print(f"\n✨ \"YES! The stars align!\" ✨")
            print(f"🎉 You guessed the magic number {magic_number} in {attempts} attempt(s)!")
            if attempts <= 3:
                print("🏆 Incredible! You truly have the gift of foresight!")
            elif attempts <= 6:
                print("👏 Well done! Your intuition is strong.")
            else:
                print("😅 You got there in the end — the spirits are pleased.")
            break

        elif guess < magic_number:
            print("🔻 \"Too low, seeker... reach higher into the cosmos.\"\n")
        else:
            print("🔺 \"Too high, seeker... come back down to earth.\"\n")

    else:
        print(f"\n💀 \"Your vision fails you today, seeker...\"")
        print(f"   The magic number was: {magic_number}")
        print("   Perhaps the stars will favour you next time.\n")


def play_again_loop():
    while True:
        fortune_teller_game()
        print("\n" + "-" * 50)
        again = input("Do you wish to consult the oracle again? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\n🔮 \"Until we meet again, seeker. Farewell...\"")
            print("=" * 50)
            break
        print()


if __name__ == "__main__":
    play_again_loop()
