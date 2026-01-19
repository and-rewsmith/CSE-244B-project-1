# set -e

# # remove everything in the analysis folder except for weights dir
# find artifacts -mindepth 1 -not -path "artifacts/weights*" -delete

# # iterate over all  and run them
# # but they need to be run with module syntax (i.e. python -m FF_RL.analysis.src.file_name)
# source env/bin/activate
# for file in FF_RL/analysis/src/*.py; do
#     echo "Running $file"
#     # convert file name to module name
#     module_name=$(echo "$file" | sed 's/\.py$//' | sed 's/\//./g')
#     echo "Module name: $module_name"
#     python -m "$module_name" --no-input
# done
