import argparse
from pathlib import Path

from trust_eval.config import EvaluationConfig, ResponseGeneratorConfig
from trust_eval.evaluator import Evaluator
from trust_eval.logging_config import logger
from trust_eval.response_generator import ResponseGenerator

# Root of the trust_eval project (directory containing this script)
_PROJECT_ROOT = Path(__file__).parent
_DEFAULT_PROMPTS_DIR = _PROJECT_ROOT / "docs" / "experiments" / "prompts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate LLM responses for trust alignment benchmarks"
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="qampari",
        choices=["asqa", "qampari", "eli5", "expertqa"],
        help="Benchmark dataset type",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--local-model-path",
        type=str,
        default=None,
        help="Path to a local model directory (overrides --model)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism in vllm",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save generated responses and evaluation results",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to input benchmark data JSON (default: data/<data_type>_eval_top100_calibrated.json)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to prompt template JSON (default: prompts/<data_type>_refusal.json)",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default=None,
        help="Path to save/load generated responses JSON (default: <output_dir>/eval_data.json)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip response generation and only run evaluation (eval-file must exist)",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Disable vllm and use HuggingFace model directly",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum context length for the model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature (default: 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_file = args.eval_file or str(output_dir / "eval_data.json")
    result_path = str(output_dir / "metrics.json")

    if not args.skip_generation:
        logger.info("=== Generating responses ===")

        model = args.local_model_path if args.local_model_path else args.model
        if args.local_model_path:
            logger.info("Using local model: %s", args.local_model_path)

        prompt_file = args.prompt_file
        if prompt_file is None:
            prompt_file = str(_DEFAULT_PROMPTS_DIR / f"{args.data_type}_refusal.json")
            if not Path(prompt_file).exists():
                logger.warning(
                    "Default prompt file not found: %s. "
                    "Pass --prompt-file to specify a custom path.",
                    prompt_file,
                )

        generator_config = ResponseGeneratorConfig(
            yaml_path="",
            data_type=args.data_type,
            model=model,
            eval_file=eval_file,
            quick_test=args.num_samples,
            tensor_parallel_size=args.concurrency,
            vllm=not args.no_vllm,
            max_length=args.max_length,
            data_file=args.data_file,
            prompt_file=prompt_file,
            temperature=args.temperature,
        )
        for key, value in vars(generator_config).items():
            logger.info("  %s: %s", key, value)

        generator = ResponseGenerator(generator_config)
        generator.generate_responses()
        generator.save_responses()
        logger.info("Responses saved to %s", eval_file)

    logger.info("=== Evaluating responses ===")
    evaluation_config = EvaluationConfig(
        yaml_path="",
        data_type=args.data_type,
        eval_file=eval_file,
        result_path=result_path,
    )
    for key, value in vars(evaluation_config).items():
        logger.info("  %s: %s", key, value)

    evaluator = Evaluator(evaluation_config)
    evaluator.compute_metrics()
    evaluator.save_results()
    logger.info("Results saved to %s", result_path)


if __name__ == "__main__":
    main()
