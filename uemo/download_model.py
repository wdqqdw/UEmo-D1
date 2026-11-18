"""Download an explicit model snapshot; evaluation and training then run locally."""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='wudq/UEmo-D1')
    p.add_argument('--output', default='models/UEmo-D1')
    p.add_argument('--revision')
    args = p.parse_args()
    from huggingface_hub import snapshot_download
    revision = args.revision
    if revision is None and args.repo == 'wudq/UEmo-D1':
        revision = 'cf17117f0d0539a40b52bb607b5d4d575cd4f8bd'
    print(snapshot_download(args.repo, revision=revision, local_dir=args.output, max_workers=4))


if __name__ == '__main__':
    main()
