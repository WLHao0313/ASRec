import os
import setproctitle

setproctitle.setproctitle("EXP@DCRec")
import argparse
from logging import getLogger
from recbole.config import Config
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color
from recbole.data.utils import create_dataset, data_preparation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='ASRec', help='Experiment Description.')
    parser.add_argument('--model', '-m', type=str, default='ASRec', help='Model for session-based rec.')
    # amazon-sports-outdoors   amazon-beauty
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='Benchmarks for session-based rec.')
    parser.add_argument('--log', '-l', type=int, default=0, help='record logs or not')
    parser.add_argument('--log_name', '-ln', type=str, default=None)
    parser.add_argument('--save', '-s', type=int, default=1, help='save models or not')
    parser.add_argument('--validation', action='store_true',
                        help='Whether evaluating on validation set (split from train set), otherwise on test set.')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='ratio of validation set.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--graphcl_enable', type=int, default=1)
    parser.add_argument('--ablation', type=str, default="full")

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    args = get_args()

    # configurations initialization
    config_dict = {
        "reproducibility": 0,
        'USER_ID_FIELD': 'user_id',
        "ITEM_ID_FIELD": 'item_id',
        "RATING_FIELD": 'rating',
        "TIME_FIELD": 'timestamp',
        "load_col": {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},

        'benchmark_filename': None,  # ['train', 'test']
        'topk': [1, 5, 10, 20],
        'metrics': ['Hit', 'NDCG', 'MRR'],
        'valid_metric': 'Hit@20',
        'eval_args': {
            'group_by': 'user',
            'order': 'TO',
            'split': {'LS': 'valid_and_test'},
            'mode': 'full'
        },
        'gpu_id': args.gpu_id,
        "MAX_ITEM_LIST_LENGTH": 50,
        "train_batch_size": args.batch_size,
        "eval_batch_size": 256,
        "stopping_step": 20,
        "fast_sample_eval": 1,

        "hidden_dropout_prob": 0.3,
        "attn_dropout_prob": 0.3,

        # Graph Args:
        "graph_dropout_prob": 0.3,
        "graphcl_enable": args.graphcl_enable,
        "graphcl_coefficient": 1e-4,
        "cl_ablation": args.ablation,
        "graph_view_fusion": 1,
        "cl_temp": 1,
        "long_tail_rate": 0.95,
        "data_augmentation": True,
        "user_inter_num_interval": '(4, inf)',
        "item_inter_num_interval": '(4, inf)',
        "initializer_range": 0.02,
        "scheduler": False,
        "step_size": 4,
        "gamma": 0.1,

        "our_att_drop_out": 0.3,
        "our_ae_drop_out": 0.2,

        "gumbel_temperature": 0.5,
        "is_gumbel_tau_anneal": True,
        'neg_sampling': None,
        "load_pre_train_emb": False,
    }
    # BEST SETTINGS
    if args.dataset == "ml-100k":
        config_dict["schedule_step"] = 30
        config_dict["stopping_step"] = 10
        config_dict["hidden_dropout_prob"] = 0.5
        config_dict["attn_dropout_prob"] = 0.5
        config_dict['train_batch_size'] = 256
        config_dict["weight_mean"] = 0.5
        config_dict["cl_temp"] = 1
        config_dict["n_layers"] = 2
        config_dict["n_heads"] = 4
        config_dict["hidden_size"] = 256
        config_dict["inner_size"] = 256
        config_dict["A"] = 1
    if args.dataset == "amazon-beauty":
        config_dict["schedule_step"] = 30
        config_dict["stopping_step"] = 10
        config_dict["hidden_dropout_prob"] = 0.5
        config_dict["attn_dropout_prob"] = 0.5
        config_dict['train_batch_size'] = 256
        config_dict["weight_mean"] = 0.5
        config_dict["cl_temp"] = 1
        config_dict["n_layers"] = 2
        config_dict["n_heads"] = 4
        config_dict["hidden_size"] = 256
        config_dict["inner_size"] = 256
        config_dict["A"] = 1.0
    elif args.dataset == "amazon-sports-outdoors":
        config_dict["schedule_step"] = 30
        config_dict["stopping_step"] = 10
        config_dict["hidden_dropout_prob"] = 0.5
        config_dict["attn_dropout_prob"] = 0.5
        config_dict['train_batch_size'] = 256
        config_dict["weight_mean"] = 0.4
        config_dict["cl_temp"] = 1
        config_dict["n_layers"] = 2
        config_dict["n_heads"] = 4
        config_dict["hidden_size"] = 256
        config_dict["inner_size"] = 256
        config_dict["A"] = 1.0
    elif args.dataset == "yelp":
        config_dict["schedule_step"] = 30
        config_dict["stopping_step"] = 10
        config_dict["hidden_dropout_prob"] = 0.5
        config_dict["attn_dropout_prob"] = 0.5
        config_dict['train_batch_size'] = 1024
        config_dict["weight_mean"] = 0.5
        config_dict["cl_temp"] = 1
        config_dict["n_layers"] = 2
        config_dict["n_heads"] = 1
        config_dict["hidden_size"] = 256
        config_dict["inner_size"] = 256
        config_dict["A"] = 0.0
        config_dict['load_col']={'inter': ['user_id', 'business_id', 'stars', 'date']}

        config_dict['ITEM_ID_FIELD']='business_id'
        config_dict['RATING_FIELD'] = 'stars'
        config_dict['TIME_FIELD']='date'
        config_dict['val_interval'] = {'date': '[1546272000, inf]'}
    elif args.dataset == "ml-1m":
        config_dict["MAX_ITEM_LIST_LENGTH"] = 200
        config_dict["schedule_step"] = 30
        config_dict["stopping_step"] = 10
        config_dict["hidden_dropout_prob"] = 0.5
        config_dict["attn_dropout_prob"] = 0.5
        config_dict['train_batch_size'] = 256
        config_dict["weight_mean"] = 0.5
        config_dict["cl_temp"] = 1
        config_dict["n_layers"] = 2
        config_dict["n_heads"] = 1
        config_dict["hidden_size"] = 64
        config_dict["inner_size"] = 64
        config_dict["A"] = 0.4

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    config = Config(model=args.model, dataset=f'{args.dataset}', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config, args.log, logfilename=args.log_name)
    logger = getLogger()
    logger.info(f"PID: {os.getpid()}")
    logger.info(args.desc)
    logger.info("\n")

    prior_args = dict()
    keywords = ["graph", "weight_mean", "kl", "sim_group", "dup"]
    for c in config_dict:
        for k in keywords:
            if c.startswith(k):
                prior_args[c] = config_dict[c]
        else:
            if c == "eval_args":
                prior_args[c] = config_dict[c]["mode"]
    prior_args = "\n".join([k + ": " + str(v) for k, v in prior_args.items()]) + "\n"
    logger.info(prior_args)

    logger.info(config)

    try:
        # dataset filtering
        dataset = create_dataset(config)
        logger.info(dataset)

        # dataset splitting
        train_dataset, valid_dataset, test_dataset = data_preparation(config, dataset)

        model = get_model(config['model'])(config, train_dataset.dataset).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer(config['MODEL_TYPE'], config["model"])(config, model)
        # model training and evaluation
        valid_score, valid_result = trainer.fit(
            train_dataset, valid_dataset, saved=args.save, show_progress=config['show_progress']
        )
        logger.info(set_color('valid result', 'yellow') + f': {valid_result}')

        # model evaluation
        test_result,test_feature = trainer.evaluate(test_dataset)
        logger.info('test result: {}'.format(test_result))

        logger.info('best valid result: {}'.format(valid_result))
        logger.info('test result: {}'.format(test_result))

        rst_dic = {
            'best_valid_score': valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': valid_result,
            'test_result': test_result
        }


    except Exception as e:
        logger.exception(e)
        raise e
