_axolotl_completions() {
    local cur prev
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # If we're completing the first argument (the command)
    if [[ $COMP_CWORD -eq 1 ]]; then
        COMPREPLY=($(compgen -W "delinearize-llama4 fetch lm-eval merge-sharded-fsdp-weights quantize vllm-serve evaluate inference merge-lora preprocess train" -- "$cur"))
        return 0
    fi

    # If the previous word was "train" or "preprocess"
    if [[ "$prev" == "train" || "$prev" == "preprocess" ]]; then
        # Complete with directories and YAML files
        COMPREPLY=($(compgen -d -- "$cur"))  # directories
        COMPREPLY+=($(compgen -f -X "!*.yaml" -- "$cur"))  # .yaml files
        COMPREPLY+=($(compgen -f -X "!*.yml" -- "$cur"))   # .yml files
        return 0
    fi

    # Default: no completion
    return 0
}

complete -F _axolotl_completions axolotl
