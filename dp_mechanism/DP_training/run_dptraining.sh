#!/bin/bash
cd DPMLBench || exit 1

echo "Generating training scripts for all algorithms..."
for algo_dir in algorithms/*/; do
    algo_name=$(basename "$algo_dir")
    if [ -f "${algo_dir}/gen_scripts.py" ]; then
        echo "Generating scripts for: ${algo_name}"
        python "${algo_dir}/gen_scripts.py"
    fi
done

echo "Executing training tasks..."
for script in scripts/*.sh; do
    echo "Running script: ${script}"
    bash "${script}"
done

echo "All training tasks have been executed."