"""Resolve only settings implemented by the selected Ringmaster strategy."""


def resolve_settings(
    cp,
    resolved,
    *,
    num_kv_heads,
    contiguous_reason=None,
    sliding_window=False,
    glm_dsa=False,
    inner_attn="flash_attention_2",
    dropout=0.0,
):
    """Validate explicit choices and return effective balancing/communication."""
    import ringmaster as rm

    explicit = cp.model_fields_set
    if num_kv_heads and num_kv_heads % resolved.ulysses_size:
        raise ValueError("ulysses_size must divide the model's num_key_value_heads")
    if glm_dsa:
        if (
            cp.backend != "auto"
            or cp.ulysses_size is not None
            or cp.ring_size is not None
            or cp.ring_impl != "auto"
            or "rotate_method" in explicit
            or cp.load_balance not in ("auto", "none")
        ):
            raise ValueError(
                "GLM DSA owns CP attention; only size and load_balance: none/auto apply"
            )
        resolved.load_balance = rm.LoadBalance.NONE
        return "glm_dsa"

    pure_ring = resolved.ring_size > 1 and resolved.ulysses_size == 1
    balance = cp.load_balance
    if balance == "auto":
        balance = (
            "head_tail"
            if pure_ring
            and not contiguous_reason
            and not sliding_window
            and "rotate_method" not in explicit
            and inner_attn == "flash_attention_2"
            and not dropout
            else "none"
        )
    if balance != "none" and not pure_ring:
        raise ValueError(f"load_balance: {balance} requires pure Ring, not Ulysses/USP")
    if balance == "head_tail" and contiguous_reason:
        raise ValueError(
            f"load_balance: head_tail is incompatible with {contiguous_reason}; use none or auto"
        )
    if balance != "none" and sliding_window:
        raise ValueError(
            f"load_balance: {balance} does not support sliding-window attention; use none"
        )
    if resolved.ring_size == 1:
        if "rotate_method" in explicit or cp.ring_impl != "auto":
            raise ValueError(
                "rotate_method and ring_impl only apply when the resolved ring_size > 1"
            )
        communication = "all_to_all"
    elif balance != "none":
        if "rotate_method" in explicit:
            raise ValueError(
                f"load_balance: {balance} owns its P2P schedule; omit rotate_method or use load_balance: none"
            )
        communication = "p2p"
    else:
        communication = "allgather" if cp.rotate_method == "allgather" else "p2p"
    if communication == "p2p" and (
        inner_attn != "flash_attention_2" or dropout or sliding_window
    ):
        raise ValueError(
            "Ring P2P schedules require flash_attention_2, zero attention dropout, "
            "and no sliding window; use load_balance: none with rotate_method: allgather"
        )
    resolved.load_balance = rm.LoadBalance(balance)
    return communication


def check_model_capability(model_type):
    """Unknown architectures use generic detection; descriptors may override support."""
    from axolotl.model_support import check_capability, get_model_support

    support = get_model_support(model_type) if model_type else None
    check_capability(
        support,
        "context_parallel",
        model_type,
        feature="Ringmaster context parallelism",
    )
