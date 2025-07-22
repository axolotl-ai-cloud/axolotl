#!/bin/bash

_axolotl_completions() {
    local cur prev
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # If we're completing the first argument (the command)
    if [[ $COMP_CWORD -eq 1 ]]; then
        mapfile -t COMPREPLY < <(compgen -W "delinearize-llama4 fetch lm-eval merge-sharded-fsdp-weights quantize vllm-serve evaluate inference merge-lora preprocess train" -- "$cur")
        return 0
    fi

    # Commands that should complete with directories and YAML files
    local -a yaml_commands=("merge-sharded-fsdp-weights" "quantize" "vllm-serve" "evaluate" "inference" "merge-lora" "preprocess" "train")

    # Check if previous word is in our list
    if [[ " ${yaml_commands[*]} " =~ (^|[[:space:]])$prev($|[[:space:]]) ]]; then
        # Use filename completion which handles directories properly
        compopt -o filenames
        mapfile -t COMPREPLY < <(compgen -f -- "$cur")

        # Filter to only include directories and YAML files
        local -a filtered=()
        for item in "${COMPREPLY[@]}"; do
            if [[ -d "$item" ]] || [[ "$item" == *.yaml ]] || [[ "$item" == *.yml ]]; then
                filtered+=("$item")
            fi
        done
        COMPREPLY=("${filtered[@]}")

        return 0
    fi

    # Default: no completion
    return 0
}

# Remove the -o nospace option - let filenames handle it
complete -F _axolotl_completions axolotl
