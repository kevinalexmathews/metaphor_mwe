import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_dir",
                        default="datasets/relocar/",
                        choices=["datasets/wimcor/five-cls/", "datasets/relocar/"],
                        help="path to directory containing pickle files")

    parser.add_argument('--train_batch_size',
                        default=8,
                        type=int,
                        help="batch size in training")

    parser.add_argument('--test_batch_size',
                        default=1,
                        type=int,
                        help="batch size in testing")

    parser.add_argument('--n_splits',
                        default=2,
                        type=int,
                        help="for cross-validation")

    parser.add_argument('-e', '--epochs',
                        default=5,
                        type=int,
                        help="number of epochs in training")

    parser.add_argument('--num_total_steps',
                        default=500,
                        type=int)

    parser.add_argument('--num_warmup_steps',
                        default=100,
                        type=int)

    parser.add_argument('-m', '--maxlen',
                        default=96,
                        type=int,
                        help="maximum sequence length")

    parser.add_argument('--dropout',
                        default=0.6,
                        type=float)

    parser.add_argument('--plm',
                        default='bert',
                        choices=['bert', 'roberta', 'xlnet', 'distilbert'],
                        help="pre-trained language model")

    parser.add_argument('--trim_texts',
                        dest='trim_texts',
                        action='store_true')

    parser.add_argument('--no_trim_texts',
                        dest='trim_texts',
                        action='store_false')

    parser.add_argument('--debug_mode',
                        dest='debug_mode',
                        action='store_true')

    parser.add_argument('--no_debug_mode',
                        dest='debug_mode',
                        action='store_false')

    parser.add_argument('--distort_context',
                        dest='distort_context',
                        action='store_true')

    parser.add_argument('--no_distort_context',
                        dest='distort_context',
                        action='store_false')

    parser.set_defaults(trim_texts=True)
    parser.set_defaults(debug_mode=True)
    parser.set_defaults(distort_context=False)
    args = parser.parse_args()
    print(args)

    return args

