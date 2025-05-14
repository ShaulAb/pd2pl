import sys
import argparse
import traceback

from pd2pl import translate_code
from pd2pl.imports_postprocess import ImportStrategy

def main():
    parser = argparse.ArgumentParser(description="Translate Pandas code to Polars code.")
    parser.add_argument('--format', action='store_true', help='Format the output code')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--import-strategy', choices=['always', 'never', 'auto', 'preserve'], default='auto',
                        help='Import handling strategy: always, never, auto, preserve (default: auto). '
                             'Note: Import strategies are only applied if translation is performed. If no translation is needed, the code is left unchanged.')
    args = parser.parse_args()

    try:
        input_code = sys.stdin.read()
        # Map CLI string to ImportStrategy enum
        strategy_map = {
            'always': ImportStrategy.ALWAYS,
            'never': ImportStrategy.NEVER,
            'auto': ImportStrategy.AUTO,
            'preserve': ImportStrategy.PRESERVE
        }
        config = {
            "format": args.format,
            "debug": args.debug,
            "import_strategy": strategy_map[args.import_strategy]
        }
        result = translate_code(input_code, postprocess_imports=True, config=config)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 