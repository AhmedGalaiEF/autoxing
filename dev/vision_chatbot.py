#!/usr/bin/env python3
import os
import sys
import ollama

# -----------------------------
# Config
# -----------------------------
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")
# examples: "llava", "llava:13b", "llama3.2-vision"


def build_user_message(text: str, image_paths: list[str] | None = None) -> dict:
    """
    Build a user message for ollama.chat.
    image_paths: list of local image paths (jpg/png/etc). If empty/None -> text-only.
    """
    msg: dict = {
        "role": "user",
        "content": text,
    }
    if image_paths:
        # The ollama Python client accepts file paths directly in 'images'
        # It will handle reading/encoding them. 
        msg["images"] = image_paths
    return msg


def main():
    print(f"Using vision model: {VISION_MODEL}")
    print("Make sure it's installed with:  ollama pull", VISION_MODEL)
    print("Type 'quit' to exit.\n")

    # Chat history (kept client-side)
    history: list[dict] = []

    # Optional system message to steer behaviour
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that can reason over images and text. "
            "When an image is provided, refer to it explicitly in your answer."
        ),
    }
    history.append(system_msg)

    while True:
        try:
            user_text = input("User (text) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print("Bye.")
            break

        # Ask for optional image(s)
        img_input = input(
            "Image path(s) (comma-separated, empty for none) > "
        ).strip()

        image_paths: list[str] = []
        if img_input:
            for raw in img_input.split(","):
                p = raw.strip().strip('"').strip("'")
                if not p:
                    continue
                if not os.path.isfile(p):
                    print(f"  [WARN] File not found: {p} (skipping)")
                    continue
                image_paths.append(p)

        # Build and append user message
        user_msg = build_user_message(user_text, image_paths or None)
        history.append(user_msg)

        # Call Ollama
        try:
            response = ollama.chat(
                model=VISION_MODEL,
                messages=history,
                stream=False,  # set True if you want token streaming
            )
        except Exception as e:
            print(f"[ERROR] Ollama call failed: {e}")
            # remove last user message from history so we can retry safely
            history.pop()
            continue

        assistant_msg = response["message"]["content"]
        print("\nAssistant >")
        print(assistant_msg)
        print()

        # Append assistant reply to history so the conversation continues
        history.append(
            {
                "role": "assistant",
                "content": assistant_msg,
            }
        )


if __name__ == "__main__":
    # Optional: allow model override via CLI:  python vision_chatbot.py llava
    if len(sys.argv) > 1:
        VISION_MODEL = sys.argv[1]
    main()

