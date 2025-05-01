import sys
import argparse
import traceback

from pd2pl import translate_code

def main():
    parser = argparse.ArgumentParser(description="Translate Pandas code to Polars code.")
    parser.add_argument('--format', action='store_true', help='Format the output code')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    try:
        input_code = sys.stdin.read()
        config = {"format": args.format, "debug": args.debug}
        result = translate_code(input_code, config=config)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 