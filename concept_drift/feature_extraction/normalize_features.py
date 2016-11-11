import click
import pandas as pd
from scipy import stats


@click.command()
@click.argument('data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('out-path', type=click.Path(exists=False, dir_okay=False))
def main(data_path, out_path):
    data = pd.read_csv(data_path, header=None, dtype='float32')
    data_normalized = data.apply(stats.zscore)
    data_normalized.to_csv(out_path, header=False, index=False)


if __name__ == '__main__':
    main()
