import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_dir",
                        default="datasets/wimcor/five-cls/",
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

    parser.add_argument('--expt_model_choice',
                        default='BertWithGCNAndMWE',
                        choices=['BertWithGCNAndMWE', 'BertWithPreWin'],
                        help="model to experiment with")

    parser.add_argument('--n_splits',
                        default=2,
                        type=int,
                        help="for cross-validation")

    parser.add_argument('--layer_no',
                        default=-1, # last_hidden_layer
                        type=int,
                        help="which PLM layer to use")

    parser.add_argument('-e', '--epochs',
                        default=5,
                        type=int,
                        help="number of epochs in training")

    parser.add_argument('-w', '--window_size',
                        default=5,
                        type=int,
                        help="size of context window")

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

    parser.add_argument('--plm_choice',
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

    parser.add_argument('--oracle',
                        dest='oracle',
                        action='store_true')

    parser.add_argument('--no_oracle',
                        dest='oracle',
                        action='store_false')

    parser.set_defaults(trim_texts=True)
    parser.set_defaults(debug_mode=False)
    parser.set_defaults(distort_context=False)
    parser.set_defaults(oracle=False)
    args = parser.parse_args()
    print(args)

    return args

