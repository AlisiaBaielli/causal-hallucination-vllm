from transformers import AutoConfig


def cache_has_contents(past_key_values) -> bool:
    """True when KV cache already holds tokens (transformers 4 tuple or 5 DynamicCache)."""
    if past_key_values is None:
        return False
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        return past_key_values.get_seq_length() > 0
    try:
        return len(past_key_values) > 0
    except TypeError:
        return False


def cache_seq_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        return past_key_values.get_seq_length()
    return past_key_values[-1][-1].shape[-2]


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)
