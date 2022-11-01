#!/bin/bash -e

results=()
current_dataset=''
if [[ -f graphs/dataset.txt ]]; then
    current_dataset="$(<graphs/dataset.txt)"
fi

load_dataset ()
{
    if [[ "$1" != "$current_dataset" ]]; then
        printf 'Preparing to extract dataset %s...\n' "$1"
        rm -rf graphs/graph_bin/ graphs/graph_info/ DGN/eig/g*.txt graphs/dataset.txt common/includes/dataset/dataset_size.txt 2>/dev/null
        total="$(unzip -Z1 "$1" | grep -c -v '/$')"
        i=1
        printf 'Extracting dataset %s: %d/%d\r' "$1" 0 "$total"
        unzip -o "$1.zip" | while IFS=$'\n' read; do
            if [[ "$(("$i" % 100))" -eq 0 ]]; then
                printf 'Extracting dataset %s: %d/%d\r' "$1" "$i" "$total"
            fi
            i="$((i + 1))"
        done
        printf 'Extracting dataset %s: %d/%d\n' "$1" "$total" "$total"
        current_dataset="$1"
    fi
}

run_case ()
{
    dataset="$(tr '[:upper:]' '[:lower:]' <<< "$1")"
    model="$(tr '[:lower:]+' '[:upper:]-' <<<"$2")"

    tput bold 2>/dev/null || true
    tput setaf 2 2>/dev/null || true
    printf '******* Running %s on %s *******\n' "$model" "$dataset"
    tput sgr0 2>/dev/null || true

    load_dataset "$1"
    (
        cd "$model"
        make host
        ./host ./build_dir.hw.*/*_compute_graphs.xclbin
    )
    total="$(grep -FxA2 'Kernel Execution' "$model"/summary.csv | tail -n 1 | cut -d',' -f5)"
    graphs="$(<common/includes/dataset/dataset_size.txt)"
    result="$(awk '{ print $1, "on", $2 ":", $3 / $4, "ms" }' <<<"$model $1 $total $graphs")"
    results+=("$result")
    printf '%s\n\n' "$result"
}

datasets=(molhiv molpcba hep10k)
models=(GIN GIN-VN GCN GAT PNA DGN)
run_all ()
{
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            run_case "$dataset" "$model"
        done
    done
}

usage ()
{
    printf 'Usage: %s <experiments...>\n' "$0"
    printf '\n'
    printf 'Experiments:\n'
    printf '  all               Run all experiments for all models and datasets\n'
    printf '  <dataset>         Run experiments for all models on one dataset\n'
    printf '  <model>           Run experiments for all datasets with one model\n'
    printf '  <dataset>:<model> Run a specific experiment\n'
    printf '\n'
    printf 'Available datasets: %s\n' "${datasets[*]}"
    printf '  Available models: %s\n' "${models[*]}"
}

if [[ "$#" -eq 0 ]]; then
    usage
    exit 1
fi

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            usage
            exit
            ;;
    esac
done

for arg in "$@"; do
    case "$arg" in
        all)
            run_all
            ;;
        *:*)
            run_case "${arg%%:*}" "${arg#*:}"
            ;;
        *)
            if [[ " ${datasets[*]} " == *" $arg "* ]]; then
                for model in "${models[@]}"; do
                    run_case "$arg" "$model"
                done
            elif [[ " ${models[*]} " == *" $arg "* ]]; then
                for dataset in "${datasets[@]}"; do
                    run_case "$dataset" "$arg"
                done
            else
                printf 'Unknown dataset or model: %s\n' "$arg"
                printf 'Run with --help for more information.\n'
                exit 1
            fi
            ;;
    esac
done

tput bold 2>/dev/null || true
tput setaf 2 2>/dev/null || true
printf '******* All results *******\n'
tput sgr0 2>/dev/null || true
for result in "${results[@]}"; do
    printf '%s\n' "$result"
done
