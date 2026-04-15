#!/usr/bin/env python3
"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

Run the inference server or generate from CLI.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turboxinf import TurboXInfConfig, TurboXInfEngine


def main():
    parser = argparse.ArgumentParser(description="TurboXInf Inference Engine")
    sub = parser.add_subparsers(dest="command")

    # Server
    srv = sub.add_parser("serve", help="Start the API server")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--no-compile", action="store_true")
    srv.add_argument("--quantize", choices=["none", "int8_triton", "int8_bnb", "int4_bnb"], default="int8_triton")

    # Generate
    gen = sub.add_parser("generate", help="Generate text from a prompt")
    gen.add_argument("prompt", nargs="?", default="Hello, how are you?")
    gen.add_argument("--max-tokens", type=int, default=256)
    gen.add_argument("--no-compile", action="store_true")
    gen.add_argument("--quantize", choices=["none", "int8_triton", "int8_bnb", "int4_bnb"], default="int8_triton")
    gen.add_argument("--stream", action="store_true")
    gen.add_argument("--no-sample", action="store_true")

    # Benchmark
    bench = sub.add_parser("benchmark", help="Run the benchmark suite")
    bench.add_argument("--quantize", choices=["none", "int8_triton"], default="int8_triton")
    bench.add_argument("--no-compile", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    config = TurboXInfConfig()
    if hasattr(args, "quantize"):
        config.quantize_weights = args.quantize
    if hasattr(args, "no_compile") and args.no_compile:
        config.use_torch_compile = False

    if args.command == "serve":
        from turboxinf.server import run_server
        # Engine is created inside the FastAPI startup event
        run_server(host=args.host, port=args.port)

    elif args.command == "generate":
        engine = TurboXInfEngine(config)
        engine.load()
        engine.warmup(runs=10)

        if args.stream:
            for chunk in engine.generate_stream(
                args.prompt,
                max_new_tokens=args.max_tokens,
                do_sample=not args.no_sample,
            ):
                if not chunk["done"]:
                    print(chunk["token"], end="", flush=True)
                else:
                    print(f"\n\n[{chunk['tokens_per_sec']} tok/s, {chunk['usage']['output_tokens']} tokens]")
        else:
            result = engine.generate(
                args.prompt,
                max_new_tokens=args.max_tokens,
                do_sample=not args.no_sample,
            )
            if result["reasoning"]:
                print(f"<think>\n{result['reasoning']}\n</think>\n")
            print(result["content"])
            print(f"\n[{result['tokens_per_sec']} tok/s, {result['usage']['output_tokens']} tokens]")

    elif args.command == "benchmark":
        engine = TurboXInfEngine(config)
        engine.load()
        engine.warmup(runs=10)

        from turboxinf.benchmark import run_benchmark
        run_benchmark(engine)


if __name__ == "__main__":
    main()
