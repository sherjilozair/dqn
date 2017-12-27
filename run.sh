set -ex

sbatch --gres=gpu:1 --time=11:59:00 --mem=4gb --job-name=Pong-semi --account=rpp-bengioy dqn.sh --game=Pong
