import argparse
import os

from transformers import SchedulerType

root_path = os.path.abspath(os.path.dirname('../'))

model_name_or_path = "/home/models/Qwen2.5-1.5B-Instruct"
# model_name_or_path = "/home/models/Llama-3.2-1B-Instruct"
# model_name_or_path = "/home/models/Meta-Llama-3-8B-Instruct"
train_file = "/home/moon_shop/data/ECInstruct.json"
mmlu_data_dir = "/home/moon_shop/mmlu_data"


def parse_args():
    parser = argparse.ArgumentParser(description="MEAL args")

    parser.add_argument("--device", default='cuda', type=str, help="device")

    # Train
    parser.add_argument("--seed", default=1234567809, type=int, help="seed")
    parser.add_argument("--model_name_or_path", default=model_name_or_path, type=str, help="model_name_or_path")
    parser.add_argument("--train_file", default=train_file, type=str, help="train_file")
    parser.add_argument("--mmlu_data_dir", default=mmlu_data_dir, type=str, help="mmlu_data_dir")
    parser.add_argument("--disable_actor_dropout", action="store_true", help="disable_actor_dropout")

    parser.add_argument("--enable_ema", action="store_true", help="enable_ema")
    parser.add_argument("--actor_lora_dim", default=0, type=int, help="actor_lora_dim")
    parser.add_argument("--actor_lora_learning_rate", default=5e-4, type=int, help="actor_lora_learning_rate")
    parser.add_argument("--actor_weight_decay", default=0, type=float, help="actor_weight_decay")
    parser.add_argument("--actor_learning_rate", default=1e-6, type=float, help="actor_learning_rate")  # 9.65e-6
    parser.add_argument("--num_warmup_steps", default=100, type=int, help="num_warmup_steps")
    parser.add_argument("--num_padding_at_beginning", type=int, default=1, help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now.",)
    parser.add_argument("--disable_reward_dropout", action="store_true", help="Disable the dropout of the reward model.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--mab_warmup_steps", default=5, type=int, help="mab_warmup_steps")
    parser.add_argument("--mab_e", default=4, type=int, help="mab_e")
    parser.add_argument("--mab_d", default=128, type=int, help="mab_d")
    parser.add_argument("--tau", default=0, type=float, help="tau")
    parser.add_argument("--batch_size", default=1, type=int, help="batch_size")
    parser.add_argument("--num_workers", default=8, type=int, help="num_workers")

    #
    parser.add_argument("--actor_zero_stage", type=int, default=0, help="ZeRO optimization stage for Actor model.",)
    parser.add_argument("--reference_zero_stage", type=int, default=0, help="ZeRO optimization stage for Reference model.",)
    parser.add_argument("--kl_ctl", type=float, default=0.1, help="KL penalty coefficient.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor for MC return estimate.",)
    parser.add_argument("--delta", type=bool, default=False, help="delta",)
    parser.add_argument("--weight_avg", type=bool, default=False, help="weight_avg",)
    parser.add_argument("--simple_baseline1", type=bool, default=True, help="weight=1/reward")
    
    # Generation
    parser.add_argument("--max_new_tokens", default=300, type=str, help="max_new_tokens")
    parser.add_argument("--max_length", type=int, default=4096, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=0.8, help="Nucleus sampling top-p")

    #
    parser.add_argument("--shot", default=1, type=int, help="shot")
    parser.add_argument("--use_task_specific_prompt", default=True, type=bool, help="use_task_specific_prompt")
    parser.add_argument("--use_letter_choices", default=True, type=bool, help="use_letter_choices")
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
    parser.add_argument("--ndoc", default=3, type=int, help="ndoc")
    parser.add_argument("--no_doc_in_demo", default=False, type=bool, help="no_doc_in_demo")
    parser.add_argument("--fewer_doc_in_demo", default=False, type=bool, help="fewer_doc_in_demo")
    parser.add_argument("--use_shorter", default=None, type=str, help="use_shorter")
    parser.add_argument("--tag", default='gtr', type=str, help="tag")
    parser.add_argument("--Epoch", type=int, default=7)
    parser.add_argument("--retrieve_in_all_docs", type=bool, default=False, help="Retrieve in all documents instead of just top ndoc")
    parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")

    return parser.parse_args()
