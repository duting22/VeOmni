import sys
from pathlib import Path


def main():
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    from npu_cpu_affinity import apply_from_environment

    apply_from_environment()

    from veomni.arguments import parse_args
    from veomni.trainer.vlm_trainer import VeOmniVLMArguments, VLMTrainer

    args = parse_args(VeOmniVLMArguments)
    trainer = VLMTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
