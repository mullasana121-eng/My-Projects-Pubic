import random
import os

# ─── Word List ───────────────────────────────────────────────────────────────
WORDS = [
    ("DRAGON",    "Mythical Creature"),
    ("WIZARD",    "Magical Being"),
    ("CASTLE",    "Place"),
    ("POTION",    "Alchemy"),
    ("SHIELD",    "Armor"),
    ("THRONE",    "Royalty"),
    ("GOBLIN",    "Creature"),
    ("SCROLL",    "Artifact"),
    ("KNIGHT",    "Title"),
    ("CHALICE",   "Artifact"),
    ("DUNGEON",   "Place"),
    ("PHOENIX",   "Mythical Creature"),
    ("SORCERER",  "Magical Being"),
    ("CRYSTAL",   "Artifact"),
    ("AMULET",    "Artifact"),
    ("SPECTER",   "Supernatural"),
    ("GRIMOIRE",  "Artifact"),
    ("BASILISK",  "Creature"),
    ("BANSHEE",   "Supernatural"),
    ("WARLOCK",   "Magical Being"),
]

# ─── Hangman ASCII Art (0 = safe ... 7 = dead) ───────────────────────────────
HANGMAN_STAGES = [
    # 0 wrong
    """
   +-------+
   |       |
   |
   |
   |
   |
  ===
    """,
    # 1 wrong
    """
   +-------+
   |       |
   |       O
   |
   |
   |
  ===
    """,
    # 2 wrong
    """
   +-------+
   |       |
   |       O
   |       |
   |
   |
  ===
    """,
    # 3 wrong
    """
   +-------+
   |       |
   |       O
   |      /|
   |
   |
  ===
    """,
    # 4 wrong
    """
   +-------+
   |       |
   |       O
   |      /|\\
   |
   |
  ===
    """,
    # 5 wrong
    """
   +-------+
   |       |
   |       O
   |      /|\\
   |      /
   |
  ===
    """,
    # 6 wrong
    """
   +-------+
   |       |
   |       O
   |      /|\\
   |      / \\
   |
  ===
    """,
    # 7 wrong — dead
    """
   +-------+
   |       |
   |      (X)
   |      /|\\
   |      / \\
   |
  ===  R.I.P.
    """,
]

MAX_WRONG = len(HANGMAN_STAGES) - 1   # 7


# ─── Helpers ─────────────────────────────────────────────────────────────────
def clear():
    os.system("cls" if os.name == "nt" else "clear")


def display_word(word: str, guessed: set) -> str:
    """Return word with unguessed letters as underscores."""
    return "  ".join(letter if letter in guessed else "_" for letter in word)


def display_wrong(wrong: list) -> str:
    return ", ".join(wrong) if wrong else "None"


def print_game(word, category, guessed, wrong):
    clear()
    print("=" * 45)
    print("        ✦  THE WIZARD'S CURSE  ✦")
    print("=" * 45)
    print(HANGMAN_STAGES[len(wrong)])
    print(f"  Category : {category}")
    print(f"  Word     : {display_word(word, guessed)}")
    print(f"  Lives    : {'❤ ' * (MAX_WRONG - len(wrong))}{'☠ ' * len(wrong)}")
    print(f"  Wrong    : {display_wrong(wrong)}")
    print("-" * 45)


# ─── Main Game Loop ───────────────────────────────────────────────────────────
def play():
    word, category = random.choice(WORDS)
    guessed: set  = set()
    wrong:   list = []

    while True:
        print_game(word, category, guessed, wrong)

        # ── Win check ──
        if all(letter in guessed for letter in word):
            print(f"\n  ✦ YOU SAVED THE PRISONER! ✦")
            print(f"  The secret word was: {word}\n")
            break

        # ── Lose check ──
        if len(wrong) >= MAX_WRONG:
            print(f"\n  ✗ THE WIZARD WINS... ✗")
            print(f"  The secret word was: {word}\n")
            break

        # ── Get input ──
        guess = input("  Guess a letter: ").strip().upper()

        if len(guess) != 1 or not guess.isalpha():
            print("  ⚠  Please enter a single letter.")
            input("  Press Enter to continue...")
            continue

        if guess in guessed or guess in wrong:
            print("  ⚠  You already guessed that letter!")
            input("  Press Enter to continue...")
            continue

        # ── Process guess ──
        if guess in word:
            guessed.add(guess)
        else:
            wrong.append(guess)

    # ── Play again? ──
    again = input("  Play again? (Y/N): ").strip().upper()
    if again == "Y":
        play()
    else:
        print("\n  Farewell, brave soul. May the kingdom prosper.\n")


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    play()
