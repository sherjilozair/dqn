set -ex

sbatch --gres=gpu:1 --time=2:59:00 --mem=4gb --job-name=Pong --account=rpp-bengioy dqn.sh --game=Pong
