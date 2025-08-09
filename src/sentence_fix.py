import re
import argparse
from pathlib import Path


def read_text_file(file_path: Path) -> str:
    """Read the text from a file."""
    with file_path.open('r', encoding='utf-8') as file:
        return file.read()


def clean_and_split_sentences(text: str) -> list[str]:
    """Clean the text and split it into sentences, preserving abbreviations and filtering short ones."""
    # Step 1: Replace newlines and normalize spaces
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Step 2: Temporary replacement for abbreviations to avoid false splits
    abbreviation_map = {
        "Dr.": "Dr<ABBR>",
        "Mr.": "Mr<ABBR>",
        "Mrs.": "Mrs<ABBR>",
        "Ms.": "Ms<ABBR>",
        "Prof.": "Prof<ABBR>",
        "Sr.": "Sr<ABBR>",
        "Jr.": "Jr<ABBR>",
        "St.": "St<ABBR>",
        "vs.": "vs<ABBR>",
        "etc.": "etc<ABBR>",
    }

    for abbr, token in abbreviation_map.items():
        text = text.replace(abbr, token)

    # Step 3: Sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Step 4: Restore abbreviations and filter out short sentences (<5 words)
    restored_sentences = []
    for sentence in sentences:
        for token, abbr in {v: k for k, v in abbreviation_map.items()}.items():
            sentence = sentence.replace(token, abbr)
        sentence = sentence.strip()
        if len(sentence.split()) >= 5:
            restored_sentences.append(sentence)

    return restored_sentences


def write_sentences_to_file(sentences: list[str], file_path: Path):
    """Write one sentence per line to a file."""
    with file_path.open('w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(sentence + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Split text into one sentence per line.")
    parser.add_argument('input', type=Path, help='Path to the input .txt file')
    parser.add_argument('output', type=Path,
                        help='Path to the output .txt file')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Input file does not exist: {args.input}")
        return

    print(f"📥 Reading from: {args.input}")
    text = read_text_file(args.input)

    print("🔍 Cleaning and splitting text...")
    sentences = clean_and_split_sentences(text)

    print(f"📤 Writing {len(sentences)} sentences to: {args.output}")
    write_sentences_to_file(sentences, args.output)
    print("✅ Done.")


if __name__ == '__main__':
    main()
